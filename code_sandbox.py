# -*- coding: utf-8 -*-
"""
code_sandbox.py
================
受控 Kaggle 竞赛 Agent 的核心执行引擎: CodeSandbox

设计哲学 —— “绝对服从的单步执行器”:
    - 只负责【生成代码】并【执行一次】。
    - 代码报错时, 只负责【捕获日志】并【停止】。
    - 绝对不含任何 while True / 递归 来自动修复 Bug, 保持直接的手动控制感。

运行约束:
    - 16G 统一内存 Mac, PyTorch 走 MPS 加速。
    - 所有代码 / 数据 / 日志强制读写于外接硬盘 /Volumes/MySSD/Kaggle_Agent/ 下。
    - 大模型使用 DeepSeek API (通过 openai SDK 调用)。
"""

import os
import json
import time
import shlex
import signal
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# ---- 第三方依赖 (延迟/防御式导入, 保证纯本地流程即使缺 paramiko 也能跑) ----
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import paramiko
except ImportError:  # pragma: no cover
    paramiko = None


# =============================================================================
# 全局日志配置
# =============================================================================
logger = logging.getLogger("CodeSandbox")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _console = logging.StreamHandler()
    _console.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_console)


# =============================================================================
# DeepSeek 代码生成的 System Prompt (强约束)
# =============================================================================
_SYSTEM_PROMPT_TEMPLATE = """你是一位顶级的 Kaggle Grandmaster 兼 Python 工程师。
你的任务: 根据【策略蓝图】和【文献报告】, 生成一份可以直接运行的完整 Python 实验脚本。

必须严格遵守以下硬性约束 (违反任意一条都视为失败):
1. 只输出一份【完整的、可直接运行的】Python 脚本, 不要输出任何解释性文字、
   不要使用 Markdown 代码块围栏 (```), 第一行起就是可执行的 Python 代码。
2. 所有数据读取路径必须【硬编码】指向外接硬盘的数据目录:
       DATA_DIR = "{data_dir}"
   例如 pd.read_csv(os.path.join(DATA_DIR, "train.csv"))。
3. 如果使用 PyTorch, 必须包含 Mac 硬件加速判定, 并把模型与张量放到该 device 上:
       device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
4. 脚本必须自包含: 包含全部 import、数据加载、训练/推理、以及最终结果的清晰打印 (print)。
5. 注意运行环境为 16G 统一内存的 Mac, 请控制 batch_size / 内存占用, 避免 OOM。
"""


# =============================================================================
# 参考项目挖掘: 意图识别 Prompt (要求严格输出 JSON)
# =============================================================================
_INTENT_PROMPT = """你是一位资深的机器学习竞赛研究员。
请阅读下面的【研究报告(Markdown)】(以及可选的策略蓝图), 提炼出这份研究的核心意图,
并给出可用于在 GitHub 上检索【相似/可参考开源项目】的搜索线索。

只输出一个合法的 JSON 对象, 不要输出任何解释文字, 不要使用 Markdown 代码块围栏。
JSON 结构必须严格如下:
{{
  "task_summary": "一句话概括这份研究要解决的核心任务",
  "domain": "所属领域, 例如 tabular-classification / cv / nlp / time-series",
  "frameworks": ["涉及或推荐的框架, 如 pytorch, sklearn, lightgbm"],
  "keywords": ["3-6 个核心技术关键词"],
  "github_queries": ["2-4 条精炼的 GitHub 仓库搜索语句, 英文, 每条尽量短且高命中"]
}}
"""

# =============================================================================
# 参考项目挖掘: 设计对齐 Prompt
# =============================================================================
_ALIGN_PROMPT = """你是一位资深的算法架构师。
下面给出【本项目的研究意图】以及【若干个已拉取到本地的开源参考项目】(含 README 摘要与目录结构)。
请对齐分析: 每个参考项目在【设计思路 / 数据处理 / 模型结构 / 工程实践】上有哪些值得本项目借鉴的点,
以及哪些点不适用或需要注意的坑。

请输出结构化的 Markdown 报告, 面向工程落地, 简洁务实。对每个参考项目给出:
- 一句话定位
- 3-5 条【可直接参考/借鉴】的点(越具体越好, 指出对应文件或模块更佳)
- 1-2 条【需注意/不适用】的点
最后给出一段【对本项目的整合建议】, 说明应优先采纳哪些设计。
"""


class CodeSandbox:
    """代码生成与多端(本地 Mac / 远程服务器)执行沙盒。"""

    # DeepSeek API 默认端点
    _DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        workspace_path: str,
        ssh_config: Optional[Dict[str, Any]] = None,
        deepseek_api_key: Optional[str] = None,
        model: str = "deepseek-chat",
    ):
        """
        参数
        ----
        workspace_path : str
            外接硬盘工作区根目录, 例如 /Volumes/MySSD/Kaggle_Agent/
        ssh_config : dict, 可选
            远程算力路由配置, 需包含: host, user, port, key_filepath。
        deepseek_api_key : str, 可选
            DeepSeek API Key。缺省时读取环境变量 DEEPSEEK_API_KEY。
        model : str
            生成代码使用的模型, "deepseek-chat" 或 "deepseek-coder"。
        """
        self.workspace_path = Path(workspace_path).resolve()
        self.src_dir = self.workspace_path / "src"
        self.logs_dir = self.workspace_path / "logs"
        self.data_dir = self.workspace_path / "data"
        self.references_dir = self.workspace_path / "references"

        # ---- 校验外接硬盘挂载点是否存在 (SSD 未插时给出明确报错) ----
        mount_root = Path("/Volumes")
        if str(self.workspace_path).startswith("/Volumes") and not mount_root.exists():
            raise RuntimeError(
                "未检测到 /Volumes 挂载点, 外接硬盘可能未连接。请先插入 SSD 再运行。"
            )

        # ---- 确保工作区子目录存在 ----
        for d in (self.src_dir, self.logs_dir, self.data_dir, self.references_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"无法创建工作区目录 {d} (外接硬盘是否已挂载并可写?): {exc}"
                ) from exc

        # ---- SSH 配置 ----
        self.ssh_config = ssh_config
        if self.ssh_config:
            missing = [k for k in ("host", "user", "port", "key_filepath") if k not in self.ssh_config]
            if missing:
                logger.warning("ssh_config 缺少字段: %s, 远程执行可能失败。", missing)

        # ---- DeepSeek 客户端 (惰性构造, 避免无 Key 时直接崩) ----
        self._api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self._client: Optional["OpenAI"] = None

        logger.info("CodeSandbox 初始化完成 | 工作区: %s", self.workspace_path)
        logger.info("  ├─ src : %s", self.src_dir)
        logger.info("  ├─ logs: %s", self.logs_dir)
        logger.info("  ├─ data: %s", self.data_dir)
        logger.info("  └─ references: %s", self.references_dir)
        logger.info("  远程算力: %s", "已配置" if self.ssh_config else "未配置(仅本地)")

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _get_client(self) -> "OpenAI":
        """惰性构造 DeepSeek(openai 兼容) 客户端。"""
        if OpenAI is None:
            raise RuntimeError("未安装 openai 库, 请执行: pip install openai")
        if not self._api_key:
            raise RuntimeError(
                "缺少 DeepSeek API Key, 请设置环境变量 DEEPSEEK_API_KEY 或在构造时传入。"
            )
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._DEEPSEEK_BASE_URL)
        return self._client

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """兜底: 若模型仍返回了 Markdown 代码块围栏, 剥离之。"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # 去掉首行 ```python / ```
            lines = lines[1:]
            # 去掉末行 ```
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip() + "\n"

    # ------------------------------------------------------------------ #
    # 1) 代码生成
    # ------------------------------------------------------------------ #
    def generate_script(self, blueprint_path: str, research_md_path: str) -> str:
        """
        读取蓝图与文献报告, 调用 DeepSeek 生成实验脚本并保存。

        返回: 生成脚本的绝对路径 workspace/src/baseline_v1.py
        """
        logger.info("[生成] 读取策略蓝图: %s", blueprint_path)
        try:
            with open(blueprint_path, "r", encoding="utf-8") as f:
                blueprint_raw = f.read()
            blueprint = json.loads(blueprint_raw)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"蓝图文件不存在: {blueprint_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"蓝图 JSON 解析失败: {exc}") from exc

        logger.info("[生成] 读取文献报告: %s", research_md_path)
        try:
            with open(research_md_path, "r", encoding="utf-8") as f:
                research_md = f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"文献报告不存在: {research_md_path}") from exc

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(data_dir=str(self.data_dir))
        user_prompt = (
            "== 策略蓝图 (blueprint.json) ==\n"
            f"{json.dumps(blueprint, ensure_ascii=False, indent=2)}\n\n"
            "== 文献报告 (research_report.md) ==\n"
            f"{research_md}\n\n"
            "请据此生成完整可运行的 Python 实验脚本。"
        )

        logger.info("[生成] 调用 DeepSeek 模型: %s ...", self.model)
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                stream=False,
            )
        except Exception as exc:  # openai 各类网络/鉴权异常统一兜底
            raise RuntimeError(f"DeepSeek API 调用失败: {exc}") from exc

        code = resp.choices[0].message.content or ""
        code = self._strip_code_fence(code)
        if not code.strip():
            raise RuntimeError("DeepSeek 返回空代码, 请检查蓝图/报告内容或 API 配额。")

        script_path = self.src_dir / "baseline_v1.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info("[生成] 脚本已保存: %s (%d 字符)", script_path, len(code))
        return str(script_path)

    # ================================================================== #
    #  参考项目挖掘管线 (意图识别 -> 搜索 -> 拉取 -> 设计对齐)
    #  同样遵循【单步执行】: 出错即停, 不做任何自动修复重试。
    # ================================================================== #

    # ---- 1) 意图识别 ----
    def recognize_intent(
        self, research_md_path: str, blueprint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        读取研究报告 (可选蓝图), 调用 DeepSeek 做意图识别, 返回结构化意图字典。

        返回示例:
          {"task_summary", "domain", "frameworks", "keywords", "github_queries"}
        """
        logger.info("[意图] 读取研究报告: %s", research_md_path)
        try:
            with open(research_md_path, "r", encoding="utf-8") as f:
                research_md = f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"研究报告不存在: {research_md_path}") from exc

        blueprint_ctx = ""
        if blueprint_path and os.path.isfile(blueprint_path):
            try:
                with open(blueprint_path, "r", encoding="utf-8") as f:
                    blueprint_ctx = "\n\n== 策略蓝图(可选参考) ==\n" + f.read()
            except OSError:
                pass

        user_prompt = (
            "== 研究报告(Markdown) ==\n" + research_md + blueprint_ctx +
            "\n\n请据此输出意图识别 JSON。"
        )

        logger.info("[意图] 调用 DeepSeek 进行意图识别 ...")
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _INTENT_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"DeepSeek 意图识别调用失败: {exc}") from exc

        raw = self._strip_code_fence(resp.choices[0].message.content or "")
        try:
            intent = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"意图识别返回非法 JSON: {exc}\n原始返回:\n{raw[:500]}") from exc

        if not intent.get("github_queries"):
            raise RuntimeError("意图识别未产出任何 github_queries, 无法继续搜索。")

        logger.info("[意图] task_summary: %s", intent.get("task_summary"))
        logger.info("[意图] domain=%s | frameworks=%s", intent.get("domain"), intent.get("frameworks"))
        logger.info("[意图] github_queries=%s", intent.get("github_queries"))

        # 落盘意图, 便于人工审阅
        try:
            with open(self.references_dir / "intent.json", "w", encoding="utf-8") as f:
                json.dump(intent, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("意图 JSON 落盘失败: %s", exc)
        return intent

    # ---- 2) 搜索合适的开源项目 ----
    def search_projects(self, intent: Dict[str, Any], max_repos: int = 3) -> list:
        """
        依据意图中的 github_queries, 使用 gh CLI 搜索仓库, 按 star 排序去重取 top-N。

        返回: [{"full_name", "url", "stars", "description", "query"}...]
        依赖已认证的 gh CLI; 未安装/未登录时抛出 RuntimeError。
        """
        if not self._has_command("gh"):
            raise RuntimeError("未检测到 gh CLI, 无法搜索项目。请安装并 `gh auth login`。")

        queries = intent.get("github_queries", [])
        seen = set()
        candidates: list = []

        for q in queries:
            logger.info("[搜索] gh search repos: %s", q)
            try:
                proc = subprocess.run(
                    [
                        "gh", "search", "repos", q,
                        "--limit", "10",
                        "--sort", "stars",
                        "--json", "fullName,url,stargazersCount,description",
                    ],
                    capture_output=True, text=True, timeout=60,
                )
            except subprocess.TimeoutExpired:
                logger.warning("[搜索] 查询超时, 跳过: %s", q)
                continue
            if proc.returncode != 0:
                logger.warning("[搜索] gh 返回错误 (%s), 跳过: %s", proc.stderr.strip()[:200], q)
                continue
            try:
                rows = json.loads(proc.stdout or "[]")
            except json.JSONDecodeError:
                continue
            for r in rows:
                name = r.get("fullName")
                if not name or name in seen:
                    continue
                seen.add(name)
                candidates.append({
                    "full_name": name,
                    "url": r.get("url"),
                    "stars": r.get("stargazersCount", 0),
                    "description": r.get("description") or "",
                    "query": q,
                })

        if not candidates:
            raise RuntimeError("未搜索到任何候选项目, 请检查网络/gh 登录或调整意图关键词。")

        candidates.sort(key=lambda x: x["stars"], reverse=True)
        top = candidates[:max_repos]
        logger.info("[搜索] 命中 %d 个候选, 选取 top-%d:", len(candidates), len(top))
        for c in top:
            logger.info("  * %s (★%d) %s", c["full_name"], c["stars"], c["url"])
        return top

    # ---- 3) 拉取项目代码到工作区 ----
    def fetch_project_code(self, repos: list, clone_timeout: int = 300) -> list:
        """
        将候选仓库浅克隆 (--depth 1) 到 references/<repo>。已存在则跳过重复克隆。
        返回带 local_path 的仓库列表 (克隆失败的条目 local_path=None)。
        """
        if not self._has_command("git"):
            raise RuntimeError("未检测到 git, 无法拉取项目代码。")

        fetched = []
        for repo in repos:
            name = repo["full_name"]
            safe_dir = name.replace("/", "__")
            dest = self.references_dir / safe_dir
            entry = dict(repo)

            if dest.exists() and any(dest.iterdir()):
                logger.info("[拉取] 已存在, 跳过: %s", dest)
                entry["local_path"] = str(dest)
                fetched.append(entry)
                continue

            logger.info("[拉取] git clone --depth 1 %s -> %s", repo["url"], dest)
            try:
                proc = subprocess.run(
                    ["git", "clone", "--depth", "1", repo["url"], str(dest)],
                    capture_output=True, text=True, timeout=clone_timeout,
                )
                if proc.returncode == 0:
                    entry["local_path"] = str(dest)
                    logger.info("[拉取] 完成: %s", name)
                else:
                    entry["local_path"] = None
                    logger.error("[拉取] 失败(%s): %s", name, proc.stderr.strip()[:200])
            except subprocess.TimeoutExpired:
                entry["local_path"] = None
                logger.error("[拉取] 超时(%ss), 跳过: %s", clone_timeout, name)
            fetched.append(entry)

        return fetched

    # ---- 4) 设计对齐, 输出可参考点报告 ----
    def align_references(self, research_md_path: str, repos: list) -> str:
        """
        汇总每个已拉取项目的 README 摘要 + 目录结构, 调用 DeepSeek 产出对齐报告。
        报告保存到 references/reference_report.md, 返回其绝对路径。
        """
        valid = [r for r in repos if r.get("local_path")]
        if not valid:
            raise RuntimeError("没有可用的已拉取项目, 无法进行设计对齐。")

        try:
            with open(research_md_path, "r", encoding="utf-8") as f:
                research_md = f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"研究报告不存在: {research_md_path}") from exc

        blocks = []
        for r in valid:
            readme = self._read_readme(r["local_path"])
            tree = self._list_tree(r["local_path"], max_entries=40)
            blocks.append(
                f"### 参考项目: {r['full_name']} (★{r.get('stars', 0)})\n"
                f"- URL: {r.get('url')}\n"
                f"- 描述: {r.get('description')}\n"
                f"- 目录结构(节选):\n{tree}\n"
                f"- README(节选):\n{readme[:3000]}\n"
            )

        user_prompt = (
            "== 本项目研究意图(Markdown) ==\n" + research_md[:4000] +
            "\n\n== 已拉取的参考项目 ==\n" + "\n\n".join(blocks) +
            "\n\n请输出设计对齐与可参考点的 Markdown 报告。"
        )

        logger.info("[对齐] 调用 DeepSeek 生成设计对齐报告 ...")
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _ALIGN_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"DeepSeek 设计对齐调用失败: {exc}") from exc

        report = resp.choices[0].message.content or ""
        report_path = self.references_dir / "reference_report.md"
        header = (
            "# 参考项目设计对齐报告\n\n"
            f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"> 参考项目数: {len(valid)}\n\n"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(header + report + "\n")

        logger.info("[对齐] 报告已保存: %s", report_path)
        return str(report_path)

    # ---- 主控: 一键完成参考项目挖掘 ----
    def discover_references(
        self,
        research_md_path: str,
        blueprint_path: Optional[str] = None,
        max_repos: int = 3,
    ) -> Dict[str, Any]:
        """
        参考项目挖掘主控管线 (单步、出错即停):
          1. recognize_intent  意图识别
          2. search_projects   搜索合适项目
          3. fetch_project_code 拉取代码到 references/
          4. align_references  对齐设计并产出可参考点报告
        返回汇总字典。
        """
        logger.info("########## 参考项目挖掘管线启动 ##########")
        try:
            intent = self.recognize_intent(research_md_path, blueprint_path)
        except Exception as exc:
            logger.error("意图识别阶段失败, 管线终止: %s", exc)
            return {"status": "FAILED", "stage": "recognize_intent", "error": str(exc)}

        try:
            repos = self.search_projects(intent, max_repos=max_repos)
        except Exception as exc:
            logger.error("项目搜索阶段失败, 管线终止: %s", exc)
            return {"status": "FAILED", "stage": "search_projects", "error": str(exc), "intent": intent}

        try:
            fetched = self.fetch_project_code(repos)
        except Exception as exc:
            logger.error("代码拉取阶段失败, 管线终止: %s", exc)
            return {"status": "FAILED", "stage": "fetch_project_code", "error": str(exc), "repos": repos}

        try:
            report_path = self.align_references(research_md_path, fetched)
        except Exception as exc:
            logger.error("设计对齐阶段失败, 管线终止: %s", exc)
            return {"status": "FAILED", "stage": "align_references", "error": str(exc), "repos": fetched}

        result = {
            "status": "SUCCESS",
            "intent": intent,
            "repos": fetched,
            "report_path": report_path,
            "references_dir": str(self.references_dir),
        }
        logger.info("########## 参考项目挖掘完成: 报告 -> %s ##########", report_path)
        return result

    # ---- 参考挖掘用到的小工具 ----
    @staticmethod
    def _has_command(cmd: str) -> bool:
        from shutil import which
        return which(cmd) is not None

    @staticmethod
    def _read_readme(repo_dir: str) -> str:
        """读取仓库根目录的 README(大小写/后缀不敏感), 找不到返回占位串。"""
        p = Path(repo_dir)
        for item in p.iterdir():
            if item.is_file() and item.name.lower().startswith("readme"):
                try:
                    return item.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    break
        return "(该仓库未找到可读的 README)"

    @staticmethod
    def _list_tree(repo_dir: str, max_entries: int = 40) -> str:
        """列出仓库主要文件结构(忽略 .git 等), 截断到 max_entries 条。"""
        p = Path(repo_dir)
        ignore = {".git", "__pycache__", ".github", "node_modules", ".idea"}
        lines = []
        count = 0
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in ignore]
            rel_root = os.path.relpath(root, p)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            if depth > 2:  # 只看前 3 层, 避免报告过长
                dirs[:] = []
                continue
            for name in sorted(files):
                rel = os.path.normpath(os.path.join(rel_root, name))
                lines.append(f"  {rel}")
                count += 1
                if count >= max_entries:
                    lines.append("  ... (更多文件省略)")
                    return "\n".join(lines)
        return "\n".join(lines) if lines else "  (空)"

    # ------------------------------------------------------------------ #
    # 2) 本地执行 (Mac / MPS)
    # ------------------------------------------------------------------ #
    def execute_local(self, script_path: str, timeout_seconds: int = 3600) -> Dict[str, Any]:
        """
        在本地拉起 Python 子进程执行脚本, 严格超时掐断, 全量落盘日志。

        为应对【本地进程死锁】, 使用独立进程组 (start_new_session=True),
        超时后对整个进程组发送 SIGKILL, 避免残留僵尸子进程。
        """
        log_path = self.logs_dir / "run_local.log"
        script_path = str(Path(script_path).resolve())

        logger.info("=" * 60)
        logger.info(">>> 路由走向: 【本地 Mac 执行】 (LOCAL / MPS)")
        logger.info(">>> 脚本: %s", script_path)
        logger.info(">>> 超时: %ss | 日志: %s", timeout_seconds, log_path)
        logger.info("=" * 60)

        if not os.path.isfile(script_path):
            msg = f"待执行脚本不存在: {script_path}"
            logger.error(msg)
            self._write_log(log_path, header="LOCAL EXECUTION - SCRIPT NOT FOUND", body=msg)
            return {"status": "FAILED", "log_path": str(log_path), "error": msg}

        start_ts = time.time()
        proc = None
        try:
            proc = subprocess.Popen(
                ["python3", "-u", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # 合并 stderr 到 stdout, 保证日志时序完整
                cwd=str(self.workspace_path),
                start_new_session=True,      # 独立进程组, 便于整组掐断
                text=True,
            )
            stdout_data, _ = proc.communicate(timeout=timeout_seconds)
            returncode = proc.returncode
            elapsed = time.time() - start_ts

            if returncode == 0:
                status = "SUCCESS"
                logger.info("<<< 本地执行成功, 耗时 %.1fs", elapsed)
            else:
                status = "FAILED"
                logger.error("<<< 本地执行失败, returncode=%s, 耗时 %.1fs", returncode, elapsed)

            self._write_log(
                log_path,
                header=f"LOCAL EXECUTION | status={status} | returncode={returncode} | elapsed={elapsed:.1f}s",
                body=stdout_data or "",
            )
            return {
                "status": status,
                "log_path": str(log_path),
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 1),
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_ts
            logger.error("<<< 本地执行【超时】(%ss), 强制掐断进程组...", timeout_seconds)
            self._kill_process_group(proc)
            # 尽力回收已产生的输出
            partial = ""
            try:
                partial, _ = proc.communicate(timeout=10)
            except Exception:
                pass
            self._write_log(
                log_path,
                header=f"LOCAL EXECUTION | status=TIMEOUT | limit={timeout_seconds}s | elapsed={elapsed:.1f}s",
                body=(partial or "") + f"\n\n[SANDBOX] 进程超过 {timeout_seconds}s 未结束, 已被强制终止。",
            )
            return {"status": "TIMEOUT", "log_path": str(log_path), "elapsed_seconds": round(elapsed, 1)}

        except Exception as exc:
            elapsed = time.time() - start_ts
            logger.error("<<< 本地执行发生异常: %s", exc)
            self._kill_process_group(proc)
            self._write_log(
                log_path,
                header=f"LOCAL EXECUTION | status=FAILED | elapsed={elapsed:.1f}s",
                body=f"[SANDBOX EXCEPTION] {type(exc).__name__}: {exc}",
            )
            return {"status": "FAILED", "log_path": str(log_path), "error": str(exc)}

    @staticmethod
    def _kill_process_group(proc: Optional[subprocess.Popen]) -> None:
        """向子进程所在进程组发送 SIGKILL, 清理死锁/残留子进程。"""
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 3) 远程执行 (SSH 服务器)
    # ------------------------------------------------------------------ #
    def execute_remote(self, script_path: str, timeout_seconds: int = 7200) -> Dict[str, Any]:
        """
        通过 SSH/SFTP 把脚本(以及缺失的数据)推送到远程服务器执行,
        拉回控制台输出到本地 logs/run_remote.log。
        """
        log_path = self.logs_dir / "run_remote.log"

        logger.info("=" * 60)
        logger.info(">>> 路由走向: 【远程服务器执行】 (REMOTE / SSH)")
        logger.info("=" * 60)

        if paramiko is None:
            msg = "未安装 paramiko, 无法远程执行: pip install paramiko"
            logger.error(msg)
            self._write_log(log_path, header="REMOTE EXECUTION - NO PARAMIKO", body=msg)
            return {"status": "FAILED", "log_path": str(log_path), "error": msg}

        if not self.ssh_config:
            msg = "未提供 ssh_config, 无法进行远程路由。"
            logger.error(msg)
            self._write_log(log_path, header="REMOTE EXECUTION - NO SSH CONFIG", body=msg)
            return {"status": "FAILED", "log_path": str(log_path), "error": msg}

        host = self.ssh_config.get("host")
        user = self.ssh_config.get("user")
        port = int(self.ssh_config.get("port", 22))
        key_filepath = self.ssh_config.get("key_filepath")
        # 远程工作区根目录 (可在 ssh_config 中覆盖)
        remote_root = self.ssh_config.get("remote_workspace", f"/home/{user}/Kaggle_Agent")
        remote_src_dir = f"{remote_root}/src"
        remote_data_dir = f"{remote_root}/data"

        script_name = os.path.basename(script_path)
        remote_script = f"{remote_src_dir}/{script_name}"

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        start_ts = time.time()

        try:
            logger.info(">>> 正在连接远程主机 %s@%s:%s ...", user, host, port)
            client.connect(
                hostname=host,
                port=port,
                username=user,
                key_filename=key_filepath,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
            )
            logger.info(">>> SSH 连接建立成功。")

            sftp = client.open_sftp()

            # ---- 确保远程目录存在 ----
            self._remote_mkdirs(sftp, remote_src_dir)
            self._remote_mkdirs(sftp, remote_data_dir)

            # ---- 推送脚本 ----
            logger.info(">>> SCP 推送脚本 -> %s", remote_script)
            sftp.put(script_path, remote_script)

            # ---- 数据集: 远程缺失则上传本地 data/ ----
            if not self._remote_dir_has_files(sftp, remote_data_dir):
                logger.info(">>> 远程数据目录为空, 开始上传本地数据集 (可能较慢)...")
                self._sftp_upload_dir(sftp, str(self.data_dir), remote_data_dir)
            else:
                logger.info(">>> 远程已存在数据集, 跳过数据上传。")

            sftp.close()

            # ---- 远程执行 (在 remote_root 下运行, -u 保证实时输出) ----
            remote_cmd = (
                f"cd {shlex.quote(remote_root)} && "
                f"python3 -u {shlex.quote(remote_script)}"
            )
            logger.info(">>> 远程执行: %s", remote_cmd)

            stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=timeout_seconds, get_pty=True)
            channel = stdout.channel

            # ---- 带超时的输出收集 (防止远程 hang 死) ----
            collected = []
            deadline = time.time() + timeout_seconds
            channel.settimeout(1.0)
            timed_out = False

            while True:
                if channel.exit_status_ready() and not channel.recv_ready():
                    break
                if time.time() > deadline:
                    timed_out = True
                    logger.error("<<< 远程执行【超时】(%ss), 关闭通道。", timeout_seconds)
                    break
                try:
                    while channel.recv_ready():
                        collected.append(channel.recv(4096).decode("utf-8", errors="replace"))
                except Exception:
                    time.sleep(0.5)

            # 收尾: 抓取残余输出
            try:
                while channel.recv_ready():
                    collected.append(channel.recv(4096).decode("utf-8", errors="replace"))
            except Exception:
                pass

            output = "".join(collected)
            elapsed = time.time() - start_ts

            if timed_out:
                try:
                    channel.close()
                except Exception:
                    pass
                self._write_log(
                    log_path,
                    header=f"REMOTE EXECUTION | status=TIMEOUT | host={host} | limit={timeout_seconds}s | elapsed={elapsed:.1f}s",
                    body=output + f"\n\n[SANDBOX] 远程进程超过 {timeout_seconds}s 未结束。",
                )
                return {"status": "TIMEOUT", "log_path": str(log_path), "elapsed_seconds": round(elapsed, 1)}

            exit_status = channel.recv_exit_status()
            status = "SUCCESS" if exit_status == 0 else "FAILED"
            logger.info("<<< 远程执行结束: status=%s, exit=%s, 耗时 %.1fs", status, exit_status, elapsed)

            self._write_log(
                log_path,
                header=f"REMOTE EXECUTION | status={status} | host={host} | exit={exit_status} | elapsed={elapsed:.1f}s",
                body=output,
            )
            return {
                "status": status,
                "log_path": str(log_path),
                "returncode": exit_status,
                "elapsed_seconds": round(elapsed, 1),
            }

        except paramiko.AuthenticationException as exc:
            return self._remote_fail(log_path, host, f"SSH 鉴权失败(检查 key_filepath/user): {exc}")
        except paramiko.SSHException as exc:
            return self._remote_fail(log_path, host, f"SSH 协议/连接异常: {exc}")
        except (OSError, TimeoutError) as exc:
            return self._remote_fail(log_path, host, f"网络/连接错误(可能断连): {exc}")
        except Exception as exc:
            return self._remote_fail(log_path, host, f"远程执行未知异常: {type(exc).__name__}: {exc}")
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _remote_fail(self, log_path: Path, host: Any, msg: str) -> Dict[str, Any]:
        logger.error("<<< %s", msg)
        self._write_log(log_path, header=f"REMOTE EXECUTION | status=FAILED | host={host}", body=msg)
        return {"status": "FAILED", "log_path": str(log_path), "error": msg}

    # ---- SFTP 辅助 ----
    @staticmethod
    def _remote_mkdirs(sftp, remote_dir: str) -> None:
        """递归创建远程目录 (类似 mkdir -p)。"""
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for p in parts:
            cur += "/" + p
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)

    @staticmethod
    def _remote_dir_has_files(sftp, remote_dir: str) -> bool:
        try:
            return len(sftp.listdir(remote_dir)) > 0
        except IOError:
            return False

    def _sftp_upload_dir(self, sftp, local_dir: str, remote_dir: str) -> None:
        """递归上传本地目录到远程。"""
        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.warning(">>> 本地数据目录不存在, 跳过上传: %s", local_dir)
            return
        self._remote_mkdirs(sftp, remote_dir)
        for item in local_dir.iterdir():
            remote_item = f"{remote_dir}/{item.name}"
            if item.is_dir():
                self._sftp_upload_dir(sftp, str(item), remote_item)
            else:
                logger.info(">>>   上传 %s", item.name)
                sftp.put(str(item), remote_item)

    # ------------------------------------------------------------------ #
    # 日志落盘
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_log(log_path: Path, header: str, body: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"===== {header} =====\n")
                f.write(f"===== timestamp: {ts} =====\n\n")
                f.write(body)
                f.write("\n")
        except OSError as exc:
            logger.error("写入日志失败 %s: %s", log_path, exc)

    # ------------------------------------------------------------------ #
    # 5) 主控调度
    # ------------------------------------------------------------------ #
    def run_experiment(self, blueprint_path: str, research_md_path: str) -> Dict[str, Any]:
        """
        主控管线:
          1. generate_script 生成代码
          2. 解析 blueprint.json 的 target_hardware 字段
          3. 路由到 execute_local / execute_remote
        单步执行, 出错即停, 不做任何自动修复重试。
        """
        logger.info("########## 实验管线启动 ##########")

        # Step 1: 生成脚本
        try:
            script_path = self.generate_script(blueprint_path, research_md_path)
        except Exception as exc:
            logger.error("代码生成阶段失败, 管线终止: %s", exc)
            return {"status": "FAILED", "stage": "generate_script", "error": str(exc)}

        # Step 2: 解析目标硬件
        try:
            with open(blueprint_path, "r", encoding="utf-8") as f:
                blueprint = json.load(f)
            target_hardware = blueprint.get("target_hardware", "local_mac")
        except Exception as exc:
            logger.error("蓝图 target_hardware 解析失败: %s", exc)
            return {"status": "FAILED", "stage": "parse_blueprint", "error": str(exc)}

        logger.info("目标硬件路由: target_hardware = %s", target_hardware)

        # Step 3: 路由分发
        if target_hardware == "local_mac":
            result = self.execute_local(script_path)
        elif target_hardware == "remote_server":
            if not self.ssh_config:
                msg = "target_hardware=remote_server 但未配置 ssh_config, 拒绝执行。"
                logger.error(msg)
                return {"status": "FAILED", "stage": "route", "error": msg, "script_path": script_path}
            result = self.execute_remote(script_path)
        else:
            msg = f"未知的 target_hardware 取值: {target_hardware} (应为 local_mac / remote_server)"
            logger.error(msg)
            return {"status": "FAILED", "stage": "route", "error": msg, "script_path": script_path}

        result["script_path"] = script_path
        result["target_hardware"] = target_hardware
        logger.info("########## 实验管线结束: status=%s ##########", result.get("status"))
        return result


# =============================================================================
# 使用示例 (直接运行本文件时的最小演示)
# =============================================================================
if __name__ == "__main__":
    WORKSPACE = "/Volumes/MySSD/Kaggle_Agent/"

    # 远程算力(可选): 不需要远程时置为 None
    SSH_CONFIG = {
        "host": "your.server.ip",
        "user": "ubuntu",
        "port": 22,
        "key_filepath": os.path.expanduser("~/.ssh/id_rsa"),
        # "remote_workspace": "/home/ubuntu/Kaggle_Agent",  # 可选覆盖
    }

    sandbox = CodeSandbox(
        workspace_path=WORKSPACE,
        ssh_config=None,  # 示例默认仅本地; 需要远程时传入 SSH_CONFIG
    )

    blueprint = os.path.join(WORKSPACE, "blueprint.json")
    research = os.path.join(WORKSPACE, "research_report.md")

    # ---- 功能A: 参考项目挖掘 (意图识别 -> 搜索 -> 拉取 -> 设计对齐) ----
    refs = sandbox.discover_references(research, blueprint, max_repos=3)
    print("\n参考项目挖掘结果:")
    print(json.dumps(
        {k: v for k, v in refs.items() if k != "intent"},
        ensure_ascii=False, indent=2,
    ))

    # ---- 功能B: 生成并执行实验 ----
    final = sandbox.run_experiment(blueprint, research)
    print("\n最终执行状态:")
    print(json.dumps(final, ensure_ascii=False, indent=2))
