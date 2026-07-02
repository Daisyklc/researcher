# -*- coding: utf-8 -*-
"""
reference_miner.py
==================
参考项目挖掘 Agent 主管线 (对应 DESIGN.md 全文)。

输入一份 Markdown(项目结构 / 调研需求), Agent 做意图识别, 联网寻找开源与闭源项目:
    - 开源项目下载到 references/<owner>__<repo>/, 逐个用子 Agent 自主分析,
      每个项目产出一份 analysis_<owner>__<repo>.md 放工作区根目录;
    - 闭源项目仅记录名称与链接(crawl4ai 抓主页 + LLM 抽取);
    - 最终汇总成一份 SUMMARY.md。

管线阶段:
    1  recognize_intent      (INTENT_LLM, openai SDK)
    2  discover_candidates   (browser-use + DISCOVERY_LLM)
    3  classify_candidates   (host 启发式 + clone 探测)
    4a fetch_project_code + ProjectAnalysisAgent(逐个)
    4b enrich_closed_source  (crawl4ai + CLOSED_LLM)
    5  write_summary         → SUMMARY.md

错误处理哲学 (DESIGN.md 第 8 节):
    - 主干出错即停 (意图/发现/汇总失败 → 返回 FAILED)。
    - 单个开源项目 clone/分析失败 → 记入 failed_projects 后继续下一个。
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from model_config import ModelConfig
from project_analysis_agent import ProjectAnalysisAgent
from prompts import (
    INTENT_PROMPT,
    DISCOVERY_TASK,
    CLOSED_PROMPT,
    SUMMARY_PROMPT,
)

# ---- 第三方依赖 (延迟/防御式导入, 保证缺库时模块仍可导入) ----
try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None


# =============================================================================
# 日志配置
# =============================================================================
logger = logging.getLogger("ReferenceMiner")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _console = logging.StreamHandler()
    _console.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_console)


# 可 clone 的代码托管 host (host 启发式判定开源)
_CODE_HOSTS = ("github.com", "gitlab.com", "gitee.com", "bitbucket.org", "codeberg.org")


class ReferenceMiner:
    """参考项目挖掘与分析管线。"""

    def __init__(
        self,
        workspace_path: str,
        model_config: Optional[ModelConfig] = None,
        discovery_mode: str = "llm",
    ):
        """
        参数
        ----
        workspace_path : str            工作区根目录 (产物落此)
        model_config   : ModelConfig    多模型分工配置, 缺省从 config.json 载入(不存在则用内置默认)
        discovery_mode : str            发现搜索模式:
            "llm"     (默认) 轻量 LLM 直接发现, 不启动浏览器, 快;
            "browser" 用 browser-use 真实浏览器联网检索, 失败/被反爬拦截时自动回退到 LLM。
        """
        self.workspace_path = Path(workspace_path).resolve()
        self.references_dir = self.workspace_path / "references"
        self.references_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = model_config or ModelConfig.load()
        self.discovery_mode = (discovery_mode or "llm").lower()
        self._client: Optional["AsyncOpenAI"] = None

        logger.info("ReferenceMiner 初始化 | 工作区: %s", self.workspace_path)
        logger.info("  ├─ references: %s", self.references_dir)
        logger.info("  ├─ 发现模式: %s", self.discovery_mode)
        logger.info(
            "  └─ 模型: intent=%s discovery=%s closed=%s analysis=%s summary=%s",
            self.cfg.intent_model, self.cfg.discovery_model, self.cfg.closed_model,
            self.cfg.analysis_model, self.cfg.summary_model,
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _get_client(self) -> "AsyncOpenAI":
        """惰性构造 openai 兼容异步客户端。"""
        if AsyncOpenAI is None:
            raise RuntimeError("未安装 openai 库, 请执行: pip install openai")
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.cfg.resolve_api_key(),
                base_url=self.cfg.base_url,
            )
        return self._client

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """兜底剥离 Markdown 代码块围栏, 便于 JSON 解析。"""
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> Any:
        """从模型返回中尽力提取 JSON(对象或数组)。"""
        cleaned = ReferenceMiner._strip_code_fence(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # 回退: 抓第一个 { ... } 或 [ ... ] 片段
        for pattern in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pattern, cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"无法从模型返回中解析 JSON。原始返回(截断):\n{cleaned[:500]}")

    @staticmethod
    def _has_command(cmd: str) -> bool:
        from shutil import which
        return which(cmd) is not None

    @staticmethod
    def _safe_dir_name(full_name: str) -> str:
        return re.sub(r"[^0-9A-Za-z_.-]", "_", full_name.replace("/", "__"))

    # ================================================================== #
    # 1) 意图识别 (INTENT_LLM)
    # ================================================================== #
    async def recognize_intent(self, input_md_path: str) -> dict:
        """
        读取输入 Markdown, 调用 INTENT_LLM 做意图识别, 返回结构化意图字典并落盘。
        字段: task_summary, knowledge_domains[], tech_stack[], keywords[],
              github_queries[], web_queries[]
        """
        logger.info("[意图] 读取输入: %s", input_md_path)
        try:
            with open(input_md_path, "r", encoding="utf-8") as f:
                input_md = f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"输入 Markdown 不存在: {input_md_path}") from exc

        client = self._get_client()
        user_prompt = "== 输入 Markdown ==\n" + input_md + "\n\n请据此输出意图识别 JSON。"
        logger.info("[意图] 调用 %s 进行意图识别 ...", self.cfg.intent_model)
        try:
            resp = await client.chat.completions.create(
                model=self.cfg.intent_model,
                messages=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"意图识别调用失败: {exc}") from exc

        intent = self._extract_json(resp.choices[0].message.content or "")
        if not isinstance(intent, dict):
            raise ValueError("意图识别返回的不是 JSON 对象。")

        # 归一化字段, 保证下游安全
        for key in ("knowledge_domains", "tech_stack", "keywords", "github_queries", "web_queries"):
            intent.setdefault(key, [])
        intent.setdefault("task_summary", "")

        if not intent["github_queries"] and not intent["web_queries"]:
            raise RuntimeError("意图识别未产出任何检索线索(github_queries / web_queries), 无法继续。")

        logger.info("[意图] task_summary: %s", intent.get("task_summary"))
        logger.info("[意图] github_queries=%s", intent.get("github_queries"))
        logger.info("[意图] web_queries=%s", intent.get("web_queries"))

        try:
            with open(self.references_dir / "intent.json", "w", encoding="utf-8") as f:
                json.dump(intent, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("意图 JSON 落盘失败: %s", exc)
        return intent

    # ================================================================== #
    # 2) 发现搜索 (browser-use + DISCOVERY_LLM)
    # ================================================================== #
    async def discover_candidates(self, intent: dict, max_candidates: int = 12) -> list[dict]:
        """
        按 github_queries + web_queries 检索候选 [{name, url, kind, note}, ...],
        并落盘 references/candidates.json。发现方式由 discovery_mode 决定:
          - "llm"(默认): 轻量 LLM 直接发现, 不启动浏览器;
          - "browser":  browser-use 真实浏览器检索, 失败/被拦截时回退 LLM。
        """
        task = DISCOVERY_TASK.format(
            task_summary=intent.get("task_summary", ""),
            knowledge_domains=intent.get("knowledge_domains", []),
            keywords=intent.get("keywords", []),
            github_queries=intent.get("github_queries", []),
            web_queries=intent.get("web_queries", []),
            max=max_candidates,
        )

        if self.discovery_mode == "browser":
            candidates = await self._run_browser_discovery(task, max_candidates)
        else:
            logger.info("[发现] 使用轻量 LLM 发现模式(discovery_mode=llm)。")
            candidates = await self._llm_discovery(task, max_candidates)

        # 去重(按 url) + 归一化
        seen: set[str] = set()
        cleaned: list[dict] = []
        for c in candidates:
            url = (c.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            cleaned.append({
                "name": (c.get("name") or url).strip(),
                "url": url,
                "kind": (c.get("kind") or "").strip() or "unknown",
                "note": (c.get("note") or "").strip(),
            })
        cleaned = cleaned[:max_candidates]

        logger.info("[发现] 候选项目 %d 个。", len(cleaned))
        try:
            with open(self.references_dir / "candidates.json", "w", encoding="utf-8") as f:
                json.dump(cleaned, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("候选 JSON 落盘失败: %s", exc)

        if not cleaned:
            raise RuntimeError("发现搜索未返回任何候选项目。")
        return cleaned

    async def _run_browser_discovery(self, task: str, max_candidates: int) -> list[dict]:
        """
        发现搜索(带弹性回退, 对齐 DESIGN §9):
          1. 优先用 browser-use 真实联网检索;
          2. 未装 browser-use / 运行异常 / 被反爬拦截返回空 → 退化为轻量 LLM 发现(不启动浏览器)。
        由 clone 做开源/闭源最终判定。
        """
        try:
            import browser_use  # noqa: F401
        except ImportError:
            logger.warning("[发现] 未安装 browser-use, 退化为 LLM 直接发现(不启动浏览器)。")
            return await self._llm_discovery(task, max_candidates)

        try:
            candidates = await self._browser_use_discover(task, max_candidates)
        except Exception as exc:
            logger.warning("[发现] browser-use 运行失败(%s), 退化为 LLM 发现。", exc)
            return await self._llm_discovery(task, max_candidates)

        if not candidates:
            logger.warning("[发现] browser-use 未提取到候选(疑似被搜索引擎验证码/反爬拦截), 退化为 LLM 发现。")
            return await self._llm_discovery(task, max_candidates)
        return candidates

    async def _browser_use_discover(self, task: str, max_candidates: int) -> list[dict]:
        """驱动 browser-use 真实浏览器联网检索, 返回原始候选列表。"""
        from browser_use import Agent

        try:
            from pydantic import BaseModel
        except ImportError as exc:
            raise RuntimeError("browser-use 需要 pydantic, 请: pip install pydantic") from exc

        class _Candidate(BaseModel):
            name: str
            url: str
            kind: str = "unknown"
            note: str = ""

        class _CandidateList(BaseModel):
            candidates: list[_Candidate]

        # DeepSeek 对 OpenAI 的 response_format=json_schema 及 JSON 围栏解析都不兼容,
        # browser-use 自带 ChatDeepSeek 专门处理这些怪癖; 其它 provider 用通用 ChatOpenAI。
        api_key = self.cfg.resolve_api_key()
        if "deepseek" in self.cfg.base_url:
            from browser_use.llm import ChatDeepSeek
            llm = ChatDeepSeek(
                model=self.cfg.discovery_model,
                base_url=self.cfg.base_url,
                api_key=api_key,
            )
        else:
            from browser_use.llm import ChatOpenAI
            llm = ChatOpenAI(
                model=self.cfg.discovery_model,
                base_url=self.cfg.base_url,
                api_key=api_key,
                dont_force_structured_output=True,
                add_schema_to_system_prompt=True,
            )
        logger.info("[发现] 启动 browser-use 联网搜索 (model=%s) ...", self.cfg.discovery_model)

        # headless + 关闭默认扩展(uBlock 等), 避免首次启动下载扩展拖垮浏览器启动超时
        agent_kwargs: dict = {"task": task, "llm": llm, "output_model_schema": _CandidateList}
        try:
            from browser_use import BrowserProfile
            agent_kwargs["browser_profile"] = BrowserProfile(
                headless=True,
                enable_default_extensions=False,
            )
        except Exception as exc:  # BrowserProfile 不可用则回退默认行为
            logger.warning("[发现] BrowserProfile 不可用, 使用默认浏览器配置: %s", exc)

        agent = Agent(**agent_kwargs)
        history = await agent.run(max_steps=20)

        # 结构化输出: 不同版本 API 略有差异, 逐一兜底
        structured = None
        for getter in ("structured_output",):
            structured = getattr(history, getter, None)
            if structured is not None:
                break
        if structured is None and hasattr(history, "final_result"):
            raw = history.final_result()
            if raw:
                try:
                    structured = _CandidateList.model_validate(self._extract_json(raw))
                except Exception:
                    parsed = self._extract_json(raw)
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict) and "candidates" in parsed:
                        return parsed["candidates"]

        if structured is None:
            raise RuntimeError("browser-use 未返回可解析的结构化候选结果。")
        return [c.model_dump() for c in structured.candidates]

    async def _llm_discovery(self, task: str, max_candidates: int) -> list[dict]:
        """
        轻量发现(无浏览器): 直接让 discovery_model 基于知识列出真实、知名的可参考项目。
        开源项目要求给出真实仓库 URL; kind 取值 open_source/closed_source。
        """
        client = self._get_client()
        system = (
            "你是资深技术调研员。基于给定调研意图, 列出【真实存在、知名】的可参考项目"
            "(开源与闭源/商业各若干)。\n"
            "只输出 JSON 数组, 不要代码围栏, 每个元素形如:\n"
            '{"name": "...", "url": "...", "kind": "open_source|closed_source", "note": "一句话说明"}\n'
            "要求: open_source 必须给出真实的 GitHub/GitLab/Gitee 仓库 URL(形如 https://github.com/owner/repo); "
            "closed_source 给出官网/产品页 URL。去重, 优先相关度与知名度高的。"
        )
        user = task + f"\n\n最多返回 {max_candidates} 个。"
        logger.info("[发现] 使用 %s 进行轻量 LLM 发现 ...", self.cfg.discovery_model)
        try:
            resp = await client.chat.completions.create(
                model=self.cfg.discovery_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM 发现调用失败: {exc}") from exc

        parsed = self._extract_json(resp.choices[0].message.content or "")
        if isinstance(parsed, dict):
            parsed = parsed.get("candidates") or parsed.get("projects") or []
        if not isinstance(parsed, list):
            raise RuntimeError("LLM 发现未返回候选数组。")
        return parsed

    # ================================================================== #
    # 3) 分类 (host 启发式 + browser-use 标注; clone 成功与否为最终判定)
    # ================================================================== #
    def classify_candidates(self, candidates: list[dict]) -> tuple[list, list]:
        """
        依据 host 启发式(github/gitlab/gitee/bitbucket/codeberg + 仓库路径) 与
        browser-use 的 kind 标注, 分成 open_source[] 与 closed_source[]。
        (clone 成功与否作为最终判定, 在 4a 阶段完成。)
        """
        open_source: list[dict] = []
        closed_source: list[dict] = []
        for c in candidates:
            url = c.get("url", "")
            if self._looks_like_repo(url) or c.get("kind") == "open_source":
                entry = dict(c)
                entry["full_name"] = self._repo_full_name(url) or c.get("name", url)
                open_source.append(entry)
            else:
                closed_source.append(dict(c))
        logger.info("[分类] 开源候选 %d 个 | 闭源候选 %d 个。", len(open_source), len(closed_source))
        return open_source, closed_source

    @staticmethod
    def _looks_like_repo(url: str) -> bool:
        """host 属于代码托管平台, 且路径形如 owner/repo, 视为可 clone 仓库。"""
        u = (url or "").lower()
        if not any(h in u for h in _CODE_HOSTS):
            return False
        m = re.search(r"(?:github|gitlab|gitee|bitbucket|codeberg)\.[a-z]+/([^/]+)/([^/#?]+)", u)
        if not m:
            return False
        owner, repo = m.group(1), m.group(2)
        # 排除非仓库路径 (如 github.com/topics, /features 等)
        reserved = {"topics", "features", "about", "explore", "marketplace", "sponsors", "orgs"}
        return owner not in reserved and bool(repo)

    @staticmethod
    def _repo_full_name(url: str) -> Optional[str]:
        m = re.search(
            r"(?:github|gitlab|gitee|bitbucket|codeberg)\.[a-z]+/([^/]+)/([^/#?]+)",
            (url or ""),
            re.IGNORECASE,
        )
        if not m:
            return None
        owner, repo = m.group(1), m.group(2)
        repo = re.sub(r"\.git$", "", repo)
        return f"{owner}/{repo}"

    # ================================================================== #
    # 4a) 开源拉取 (git clone --depth 1)
    # ================================================================== #
    def fetch_project_code(self, repo: dict, clone_timeout: int = 300) -> Optional[str]:
        """
        将单个候选仓库浅克隆到 references/<owner>__<repo>/。
        成功返回本地路径, 失败/超时返回 None。
        """
        import subprocess

        if not self._has_command("git"):
            raise RuntimeError("未检测到 git, 无法拉取项目代码。")

        full_name = repo.get("full_name") or repo.get("name") or repo.get("url")
        url = repo.get("url")
        dest = self.references_dir / self._safe_dir_name(full_name)

        if dest.exists() and any(dest.iterdir()):
            logger.info("[拉取] 已存在, 跳过: %s", dest)
            return str(dest)

        logger.info("[拉取] git clone --depth 1 %s -> %s", url, dest)
        try:
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                capture_output=True, text=True, timeout=clone_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("[拉取] 超时(%ss), 跳过: %s", clone_timeout, full_name)
            return None
        if proc.returncode == 0:
            logger.info("[拉取] 完成: %s", full_name)
            return str(dest)
        logger.error("[拉取] 失败(%s): %s", full_name, proc.stderr.strip()[:200])
        return None

    # ================================================================== #
    # 4b) 闭源抓取 (crawl4ai + CLOSED_LLM)
    # ================================================================== #
    async def enrich_closed_source(self, closed: list[dict]) -> list[dict]:
        """
        对每个闭源项目用 crawl4ai 抓取主页, 用 CLOSED_LLM 抽取:
        {name, url, description, commercial_note}。只存链接与信息, 不下载。
        """
        if not closed:
            return []

        crawler_cls = None
        try:
            from crawl4ai import AsyncWebCrawler
            crawler_cls = AsyncWebCrawler
        except ImportError:
            logger.warning("[闭源] 未安装 crawl4ai, 将跳过网页抓取, 仅保留基础链接信息。")

        results: list[dict] = []
        client = self._get_client()

        if crawler_cls is not None:
            async with crawler_cls() as crawler:
                for item in closed:
                    results.append(await self._enrich_one_closed(crawler, client, item))
        else:
            for item in closed:
                results.append({
                    "name": item.get("name", item.get("url", "")),
                    "url": item.get("url", ""),
                    "description": item.get("note", ""),
                    "commercial_note": "(未抓取: 缺少 crawl4ai)",
                })
        logger.info("[闭源] 已处理闭源项目 %d 个。", len(results))
        return results

    async def _enrich_one_closed(self, crawler, client, item: dict) -> dict:
        url = item.get("url", "")
        base = {
            "name": item.get("name", url),
            "url": url,
            "description": item.get("note", ""),
            "commercial_note": "",
        }
        try:
            res = await crawler.arun(url=url)
            page_md = (getattr(res, "markdown", None) or "")
            if hasattr(page_md, "raw_markdown"):  # 某些版本 markdown 是对象
                page_md = page_md.raw_markdown
            page_md = str(page_md)[:6000]
        except Exception as exc:
            logger.warning("[闭源] 抓取失败 %s: %s", url, exc)
            base["commercial_note"] = f"(抓取失败: {exc})"
            return base

        if not page_md.strip():
            return base

        try:
            resp = await client.chat.completions.create(
                model=self.cfg.closed_model,
                messages=[
                    {"role": "system", "content": CLOSED_PROMPT},
                    {"role": "user", "content": f"URL: {url}\n\n正文(Markdown):\n{page_md}"},
                ],
                temperature=0.1,
                stream=False,
            )
            extracted = self._extract_json(resp.choices[0].message.content or "")
            if isinstance(extracted, dict):
                base["name"] = extracted.get("name") or base["name"]
                base["description"] = extracted.get("description") or base["description"]
                base["commercial_note"] = extracted.get("commercial_note") or base["commercial_note"]
        except Exception as exc:
            logger.warning("[闭源] 信息抽取失败 %s: %s", url, exc)
            base["commercial_note"] = base["commercial_note"] or f"(抽取失败: {exc})"
        return base

    # ================================================================== #
    # 5) 汇总落盘 (SUMMARY_LLM)
    # ================================================================== #
    async def write_summary(
        self,
        intent: dict,
        open_results: list[dict],
        closed: list[dict],
        failed: list[dict],
    ) -> str:
        """
        产出 SUMMARY.md 并返回其路径。开源项目摘要指向对应 analysis_<repo>.md 的相对链接。
        """
        client = self._get_client()

        open_blocks = []
        for r in open_results:
            open_blocks.append(
                f"- full_name: {r.get('full_name')}\n"
                f"  url: {r.get('url')}\n"
                f"  analysis_md: {r.get('analysis_md')}\n"
                f"  分析摘要(节选): {(r.get('analysis_excerpt') or '')[:600]}"
            )
        closed_blocks = [
            f"- {c.get('name')} | {c.get('url')} | {c.get('description')} | {c.get('commercial_note')}"
            for c in closed
        ]
        failed_blocks = [
            f"- {f.get('full_name')} | {f.get('url')} | stage={f.get('stage')} | error={f.get('error')}"
            for f in failed
        ]

        user_prompt = (
            "== 调研意图 ==\n" + json.dumps(intent, ensure_ascii=False, indent=2) +
            "\n\n== 开源项目(含分析摘要与 analysis_md 路径) ==\n" + ("\n".join(open_blocks) or "(无)") +
            "\n\n== 闭源项目 ==\n" + ("\n".join(closed_blocks) or "(无)") +
            "\n\n== 失败项目 ==\n" + ("\n".join(failed_blocks) or "(无)") +
            "\n\n请据此产出 SUMMARY.md。"
        )

        logger.info("[汇总] 调用 %s 生成 SUMMARY.md ...", self.cfg.summary_model)
        try:
            resp = await client.chat.completions.create(
                model=self.cfg.summary_model,
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"汇总生成调用失败: {exc}") from exc

        body = self._strip_code_fence(resp.choices[0].message.content or "")
        summary_path = self.workspace_path / "SUMMARY.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(body + "\n")
        logger.info("[汇总] SUMMARY.md 已保存: %s", summary_path)
        return str(summary_path)

    # ================================================================== #
    # 主控编排
    # ================================================================== #
    async def discover_and_analyze(
        self,
        input_md_path: str,
        max_open_source: int = 3,
        max_closed_source: int = 5,
    ) -> dict:
        """
        主干出错即停; 但 4a 中单个开源项目分析失败时跳过该项目并记录, 不中断整条管线。
        """
        logger.info("########## 参考项目挖掘管线启动 ##########")

        # 1) 意图识别 (主干)
        try:
            intent = await self.recognize_intent(input_md_path)
        except Exception as exc:
            logger.error("意图识别失败, 管线终止: %s", exc)
            return {"status": "FAILED", "stage": "recognize_intent", "error": str(exc)}

        # 2) 发现搜索 (主干)
        try:
            candidates = await self.discover_candidates(
                intent, max_candidates=max_open_source + max_closed_source + 4
            )
        except Exception as exc:
            logger.error("发现搜索失败, 管线终止: %s", exc)
            return {"status": "FAILED", "stage": "discover_candidates", "error": str(exc), "intent": intent}

        # 3) 分类
        open_cands, closed_cands = self.classify_candidates(candidates)
        open_cands = open_cands[:max_open_source]
        closed_cands = closed_cands[:max_closed_source]

        # 4a) 开源: 逐个 clone + 子 Agent 分析 (单项目失败即跳过)
        open_results: list[dict] = []
        failed_projects: list[dict] = []
        client = self._get_client()

        for repo in open_cands:
            full_name = repo.get("full_name") or repo.get("name")
            url = repo.get("url")
            try:
                local_path = self.fetch_project_code(repo)
            except Exception as exc:
                logger.error("[4a] clone 阶段异常, 记为闭源候选: %s", exc)
                local_path = None
            if not local_path:
                # clone 失败 → 按 DESIGN 归为闭源(只存链接), 同时记入 failed
                failed_projects.append({
                    "full_name": full_name, "url": url,
                    "stage": "fetch_project_code", "error": "clone 失败或非仓库 URL",
                })
                closed_cands.append({"name": full_name, "url": url, "note": repo.get("note", "")})
                continue

            try:
                agent = ProjectAnalysisAgent(
                    repo_path=local_path,
                    model=self.cfg.analysis_model,
                    client=client,
                )
                markdown = await agent.analyze(intent)
            except Exception as exc:
                logger.error("[4a] 分析失败, 跳过 %s: %s", full_name, exc)
                failed_projects.append({
                    "full_name": full_name, "url": url,
                    "stage": "analyze", "error": str(exc),
                })
                continue

            analysis_name = f"analysis_{self._safe_dir_name(full_name)}.md"
            analysis_path = self.workspace_path / analysis_name
            header = f"# 开源项目分析: {full_name}\n\n> URL: {url}\n\n"
            with open(analysis_path, "w", encoding="utf-8") as f:
                f.write(header + markdown + "\n")
            logger.info("[4a] 分析报告已保存: %s", analysis_path)

            open_results.append({
                "full_name": full_name,
                "url": url,
                "stars": repo.get("stars"),
                "local_path": local_path,
                "analysis_md": analysis_name,
                "analysis_excerpt": markdown[:800],
            })

        # 4b) 闭源: crawl4ai 抓取 + 抽取 (主干; 失败即停)
        try:
            closed_results = await self.enrich_closed_source(closed_cands[:max_closed_source])
        except Exception as exc:
            logger.error("闭源抓取失败, 管线终止: %s", exc)
            return {
                "status": "FAILED", "stage": "enrich_closed_source", "error": str(exc),
                "intent": intent, "open_source": open_results, "failed_projects": failed_projects,
            }

        # 5) 汇总 (主干)
        try:
            summary_path = await self.write_summary(intent, open_results, closed_results, failed_projects)
        except Exception as exc:
            logger.error("汇总失败, 管线终止: %s", exc)
            return {
                "status": "FAILED", "stage": "write_summary", "error": str(exc),
                "intent": intent, "open_source": open_results,
                "closed_source": closed_results, "failed_projects": failed_projects,
            }

        result = {
            "status": "SUCCESS",
            "intent": intent,
            "open_source": [
                {k: v for k, v in r.items() if k != "analysis_excerpt"} for r in open_results
            ],
            "failed_projects": failed_projects,
            "closed_source": closed_results,
            "summary_path": summary_path,
        }
        logger.info("########## 挖掘完成: SUMMARY -> %s ##########", summary_path)
        return result

    # ---- 同步兼容入口 ----
    def discover_and_analyze_sync(self, *args, **kwargs) -> dict:
        return asyncio.run(self.discover_and_analyze(*args, **kwargs))
