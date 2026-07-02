# -*- coding: utf-8 -*-
"""
code_sandbox.py
================
项目分析 Agent 的核心执行引擎: CodeSandbox

主链路 (run_analysis_pipeline):
    需求表格/文档 → 截图识别 → 寻找项目 → 加载代码 → 闭环整理 → 输出行动报告 .md

设计哲学 —— “绝对服从的单步执行器”:
    - 每个阶段只执行一次, 出错即停, 不做自动修复重试。
    - 输出 .md 记录阅读路径、开源软件、网上讨论与项目逻辑, 供下一轮续接。
"""

import os
import re
import json
import time
import shlex
import signal
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

# ---- 第三方依赖 (延迟/防御式导入) ----
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import paramiko
except ImportError:  # pragma: no cover
    paramiko = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    from duckduckgo_search import DDGS
except ImportError:  # pragma: no cover
    DDGS = None


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
_INTENT_PROMPT = """你是一位资深的机器学习竞赛研究员与软件架构师。
请阅读下面的【需求文档】(以及可选的补充材料), 提炼核心意图,
并给出可用于检索【开源项目】与【网上讨论】的搜索线索。

只输出一个合法的 JSON 对象, 不要输出任何解释文字, 不要使用 Markdown 代码块围栏。
JSON 结构必须严格如下:
{{
  "task_summary": "一句话概括要解决的核心任务",
  "domain": "所属领域, 例如 tabular-classification / cv / nlp / web-backend",
  "frameworks": ["涉及或可能用到的框架, 如 pytorch, sklearn, fastapi"],
  "keywords": ["3-6 个核心技术关键词"],
  "github_queries": ["2-4 条精炼的 GitHub 仓库搜索语句, 英文, 每条尽量短且高命中"],
  "search_queries": ["2-4 条用于检索网上正在讨论的相关话题的搜索语句, 中英文均可"]
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

# =============================================================================
# 项目分析管线: 闭环整理 Prompt
# =============================================================================
_REFINE_CONTEXT_PROMPT = """你是一位资深的软件架构师。
根据【需求文档】与【已加载的项目代码上下文】, 深入整理该项目的:
- 整体业务流程与模块职责
- 关键入口与核心调用链
- 已阅读文件路径及其作用
- 当前最值得继续深入阅读的文件

输出结构化的 Markdown, 面向工程落地, 简洁务实。"""

# =============================================================================
# 项目分析管线: 行动报告 Prompt
# =============================================================================
_ACTION_REPORT_PROMPT = """你是一位技术文档工程师。
请根据提供的全部上下文, 生成一份完整的【项目行动报告】Markdown。

报告必须包含以下章节 (按顺序, 使用二级标题):
## 需求摘要
## 开源软件名称
列出项目依赖的第三方库/框架/工具, 注明来源文件 (如 requirements.txt)。
## 网上正在讨论的
汇总与该项目/技术栈相关的外部讨论要点 (来自提供的搜索结果)。
## 项目逻辑
说明业务流程、模块划分、入口文件、关键调用链, 并列出【已阅读的路径】。
## 当前阶段提示词
给出一段可直接用于下一轮 Agent/人工继续深入的行动提示词 (可续接历史上下文)。

要求: 内容具体、可执行, 路径写全, 不要空泛套话。"""

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".rst"}
_DEPENDENCY_FILENAMES = (
    "requirements.txt", "requirements-dev.txt", "pyproject.toml",
    "setup.py", "setup.cfg", "Pipfile", "environment.yml",
    "package.json", "go.mod", "Cargo.toml",
)
_ENTRY_CANDIDATES = (
    "main.py", "app.py", "run.py", "manage.py", "cli.py",
    "index.js", "index.ts", "main.go", "lib.rs",
)


class CodeSandbox:
    """项目分析 + 代码执行沙盒 (主链路: 需求 → 项目理解 → 行动报告)。"""

    _DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    _MAX_KEY_FILE_BYTES = 12_000
    _MAX_KEY_FILES = 8

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
            工作区根目录, 例如 ./kaggle_agent_workspace
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
        self.reports_dir = self.workspace_path / "reports"

        for d in (self.src_dir, self.logs_dir, self.data_dir, self.references_dir, self.reports_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"无法创建工作区目录 {d} (请检查路径是否存在且可写): {exc}"
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
        logger.info("  ├─ references: %s", self.references_dir)
        logger.info("  └─ reports: %s", self.reports_dir)
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

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """统一 LLM 调用入口。"""
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"DeepSeek API 调用失败: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    # ================================================================== #
    #  项目分析管线: 需求解析 → 找项目 → 加载代码 → 报告
    # ================================================================== #

    def parse_requirement(self, requirement_path: str) -> Dict[str, Any]:
        """
        解析需求输入: 支持图片(OCR)或文本/JSON/CSV 文件。

        返回: {"source_path", "source_type", "text"}
        """
        path = Path(requirement_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"需求文件不存在: {requirement_path}")

        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS:
            text = self._ocr_image(path)
            source_type = "image_ocr"
        elif suffix in _TEXT_EXTENSIONS or suffix == "":
            text = path.read_text(encoding="utf-8", errors="replace")
            source_type = "text"
        elif suffix == ".json":
            raw = path.read_text(encoding="utf-8", errors="replace")
            try:
                text = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                text = raw
            source_type = "json"
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            source_type = "text"

        text = text.strip()
        if not text:
            raise ValueError(f"需求文件内容为空: {requirement_path}")

        parsed_path = self.reports_dir / "requirement_parsed.txt"
        parsed_path.write_text(text, encoding="utf-8")
        logger.info("[需求] 已解析 (%s): %d 字符 -> %s", source_type, len(text), parsed_path)
        return {"source_path": str(path), "source_type": source_type, "text": text}

    def _ocr_image(self, image_path: Path) -> str:
        """对需求表格截图执行 OCR。"""
        if Image is None or pytesseract is None:
            raise RuntimeError(
                "图片识别需要安装 Pillow 与 pytesseract, 且系统需安装 tesseract 二进制。"
                " 也可改用 .md / .txt 文本需求文件。"
            )
        if not self._has_command("tesseract"):
            raise RuntimeError(
                "未检测到 tesseract 可执行文件。请安装 tesseract, 或改用文本需求文件。"
            )
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        if not text.strip():
            raise ValueError(f"OCR 未识别到有效文字: {image_path}")
        return text

    def recognize_intent_from_text(
        self,
        requirement_text: str,
        supplemental_path: Optional[str] = None,
        require_github: bool = True,
    ) -> Dict[str, Any]:
        """从需求文本做意图识别, 返回结构化 intent 字典。"""
        supplemental = ""
        if supplemental_path and os.path.isfile(supplemental_path):
            supplemental = (
                "\n\n== 补充材料 ==\n"
                + Path(supplemental_path).read_text(encoding="utf-8", errors="replace")
            )

        user_prompt = (
            "== 需求文档 ==\n" + requirement_text + supplemental +
            "\n\n请据此输出意图识别 JSON。"
        )
        logger.info("[意图] 调用 DeepSeek 进行意图识别 ...")
        raw = self._strip_code_fence(self._call_llm(_INTENT_PROMPT, user_prompt, temperature=0.1))
        try:
            intent = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"意图识别返回非法 JSON: {exc}\n原始返回:\n{raw[:500]}") from exc

        if require_github and not intent.get("github_queries"):
            raise RuntimeError("意图识别未产出 github_queries, 无法搜索开源项目。")
        if not intent.get("search_queries"):
            intent["search_queries"] = intent.get("keywords", [])[:3]

        intent_path = self.references_dir / "intent.json"
        intent_path.write_text(json.dumps(intent, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[意图] task_summary: %s", intent.get("task_summary"))
        return intent

    def organize_local_project(self, project_path: str) -> Dict[str, Any]:
        """闭源分支: 整理本地项目路径与基本信息。"""
        root = Path(project_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"本地项目路径不存在: {project_path}")
        if not root.is_dir():
            raise ValueError(f"本地项目路径不是目录: {project_path}")

        project = {
            "project_type": "closed_source",
            "full_name": root.name,
            "url": None,
            "stars": 0,
            "description": f"本地项目: {root}",
            "local_path": str(root),
        }
        logger.info("[闭源] 已整理本地项目: %s", root)
        return project

    def find_project(
        self,
        intent: Dict[str, Any],
        local_project_path: Optional[str] = None,
        max_repos: int = 1,
    ) -> Dict[str, Any]:
        """
        寻找项目: 优先使用本地路径(闭源), 否则搜索并拉取开源仓库。
        """
        if local_project_path:
            return self.organize_local_project(local_project_path)

        repos = self.search_projects(intent, max_repos=max_repos)
        fetched = self.fetch_project_code(repos)
        valid = [r for r in fetched if r.get("local_path")]
        if not valid:
            raise RuntimeError("未找到可用的开源项目 (搜索或克隆失败)。")

        primary = valid[0]
        primary["project_type"] = "open_source"
        logger.info("[寻找项目] 选定开源项目: %s", primary.get("full_name"))
        return primary

    def load_project_code(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """深入加载项目: 目录树、README、依赖文件、入口与关键源码。"""
        local_path = project.get("local_path")
        if not local_path or not os.path.isdir(local_path):
            raise RuntimeError("项目 local_path 无效, 无法加载代码。")

        root = Path(local_path)
        read_paths: List[str] = []
        dependency_files: Dict[str, str] = {}
        key_files: Dict[str, str] = {}

        readme = self._read_readme(local_path)
        if readme != "(该仓库未找到可读的 README)":
            read_paths.append(str(root / "README*"))

        tree = self._list_tree(local_path, max_entries=80)
        for fname in _DEPENDENCY_FILENAMES:
            fpath = root / fname
            if fpath.is_file():
                content = fpath.read_text(encoding="utf-8", errors="replace")
                dependency_files[fname] = content[: self._MAX_KEY_FILE_BYTES]
                read_paths.append(str(fpath))

        for entry_name in _ENTRY_CANDIDATES:
            fpath = root / entry_name
            if fpath.is_file():
                content = fpath.read_text(encoding="utf-8", errors="replace")
                key_files[entry_name] = content[: self._MAX_KEY_FILE_BYTES]
                read_paths.append(str(fpath))

        for rel in self._discover_key_source_files(root):
            fpath = root / rel
            if str(fpath) in read_paths:
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            key_files[rel] = content[: self._MAX_KEY_FILE_BYTES]
            read_paths.append(str(fpath))
            if len(key_files) >= self._MAX_KEY_FILES:
                break

        context = {
            "project": project,
            "local_path": local_path,
            "readme": readme[:4000],
            "tree": tree,
            "dependency_files": dependency_files,
            "key_files": key_files,
            "read_paths": read_paths,
        }
        ctx_path = self.reports_dir / "project_context.json"
        ctx_path.write_text(
            json.dumps({**context, "key_files": list(key_files.keys())}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("[加载代码] 已阅读 %d 个路径", len(read_paths))
        return context

    def _discover_key_source_files(self, root: Path) -> List[str]:
        """发现值得阅读的关键源码文件 (优先 src/ 与顶层 .py)。"""
        ignore = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
        candidates: List[str] = []
        for pattern in ("src/**/*.py", "**/*.py"):
            for fpath in root.glob(pattern):
                if any(part in ignore for part in fpath.parts):
                    continue
                if fpath.name.startswith("test_") or fpath.name.endswith("_test.py"):
                    continue
                rel = str(fpath.relative_to(root))
                if rel in candidates:
                    continue
                candidates.append(rel)
                if len(candidates) >= self._MAX_KEY_FILES:
                    return candidates
        return candidates

    def extract_dependencies(self, project_path: str) -> List[str]:
        """从依赖清单与 Python import 中提取开源软件名称。"""
        root = Path(project_path)
        deps: set = set()

        req = root / "requirements.txt"
        if req.is_file():
            for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = re.split(r"[<>=!~\[]", line)[0].strip()
                if name:
                    deps.add(name)

        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            for match in re.finditer(r'["\']([a-zA-Z0-9_-]+)["\']', pyproject.read_text(encoding="utf-8", errors="replace")):
                val = match.group(1)
                if val not in {"dependencies", "dev-dependencies", "optional-dependencies"}:
                    deps.add(val)

        pkg = root / "package.json"
        if pkg.is_file():
            try:
                data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
                for section in ("dependencies", "devDependencies"):
                    for name in (data.get(section) or {}):
                        deps.add(name)
            except json.JSONDecodeError:
                pass

        import_re = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)")
        for py_file in list(root.glob("**/*.py"))[:30]:
            if ".venv" in py_file.parts or "venv" in py_file.parts:
                continue
            try:
                for line in py_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    m = import_re.match(line)
                    if m:
                        deps.add(m.group(1).split(".")[0])
            except OSError:
                continue

        stdlib = {"os", "sys", "json", "re", "pathlib", "typing", "datetime", "logging", "time", "math"}
        return sorted(d for d in deps if d and d not in stdlib)

    def fetch_online_discussions(
        self, intent: Dict[str, Any], max_results: int = 5
    ) -> List[Dict[str, str]]:
        """检索网上正在讨论的相关话题。"""
        if DDGS is None:
            logger.warning("未安装 duckduckgo-search, 跳过网上讨论检索。")
            return []

        queries = intent.get("search_queries") or intent.get("keywords", [])[:3]
        results: List[Dict[str, str]] = []
        seen_urls: set = set()

        try:
            with DDGS() as ddgs:
                for q in queries:
                    logger.info("[讨论] 搜索: %s", q)
                    try:
                        rows = list(ddgs.text(q, max_results=max_results))
                    except Exception as exc:
                        logger.warning("[讨论] 查询失败, 跳过: %s | %s", q, exc)
                        continue
                    for row in rows:
                        url = row.get("href") or row.get("link") or ""
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        results.append({
                            "query": q,
                            "title": row.get("title") or "",
                            "url": url,
                            "snippet": row.get("body") or row.get("snippet") or "",
                        })
        except Exception as exc:
            logger.warning("网上讨论检索异常: %s", exc)

        disc_path = self.reports_dir / "online_discussions.json"
        disc_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[讨论] 收集 %d 条结果", len(results))
        return results

    def build_stage_prompt(
        self,
        requirement_text: str,
        intent: Dict[str, Any],
        project_context: Dict[str, Any],
        dependencies: List[str],
        discussions: List[Dict[str, str]],
        history_prompt: Optional[str] = None,
    ) -> str:
        """生成当前阶段提示词, 可续接历史上下文。"""
        history_block = ""
        if history_prompt:
            history_block = f"\n\n== 历史提示词(可续接) ==\n{history_prompt}\n"

        discussion_block = "\n".join(
            f"- [{d.get('title')}] {d.get('url')}\n  {d.get('snippet', '')[:200]}"
            for d in discussions[:8]
        ) or "(未检索到外部讨论)"

        key_file_names = list(project_context.get("key_files", {}).keys())
        prompt = (
            f"任务: {intent.get('task_summary', '')}\n"
            f"领域: {intent.get('domain', '')}\n"
            f"项目路径: {project_context.get('local_path', '')}\n"
            f"项目类型: {project_context.get('project', {}).get('project_type', '')}\n"
            f"已阅读路径数: {len(project_context.get('read_paths', []))}\n"
            f"关键文件: {', '.join(key_file_names) or '(无)'}\n"
            f"开源依赖: {', '.join(dependencies[:20]) or '(未识别)'}\n"
            f"外部讨论摘要:\n{discussion_block}\n"
            f"{history_block}\n"
            "请基于以上上下文, 继续深入阅读代码并完善项目逻辑理解。"
        )
        prompt_path = self.reports_dir / "stage_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt

    def refine_project_context(
        self,
        requirement_text: str,
        project_context: Dict[str, Any],
        history_prompt: Optional[str] = None,
    ) -> str:
        """闭环整理项目流程与信息。"""
        history_block = ""
        if history_prompt:
            history_block = f"\n\n== 历史上下文 ==\n{history_prompt}\n"

        key_blocks = []
        for name, content in project_context.get("key_files", {}).items():
            key_blocks.append(f"### 文件: {name}\n```\n{content[:2000]}\n```")

        dep_blocks = []
        for name, content in project_context.get("dependency_files", {}).items():
            dep_blocks.append(f"### {name}\n```\n{content[:1500]}\n```")

        user_prompt = (
            "== 需求文档 ==\n" + requirement_text[:3000] +
            "\n\n== 目录结构 ==\n" + project_context.get("tree", "") +
            "\n\n== README ==\n" + project_context.get("readme", "")[:3000] +
            "\n\n== 依赖文件 ==\n" + "\n".join(dep_blocks) +
            "\n\n== 关键源码 ==\n" + "\n\n".join(key_blocks) +
            "\n\n== 已阅读路径 ==\n" + "\n".join(project_context.get("read_paths", [])) +
            history_block
        )
        logger.info("[闭环] 整理项目流程与信息 ...")
        refined = self._call_llm(_REFINE_CONTEXT_PROMPT, user_prompt, temperature=0.3)
        refined_path = self.reports_dir / "refined_context.md"
        refined_path.write_text(refined + "\n", encoding="utf-8")
        return refined

    def generate_action_report(
        self,
        requirement_text: str,
        intent: Dict[str, Any],
        project: Dict[str, Any],
        project_context: Dict[str, Any],
        dependencies: List[str],
        discussions: List[Dict[str, str]],
        refined_context: str,
        stage_prompt: str,
    ) -> str:
        """生成最终行动报告 .md (含开源软件、网上讨论、逻辑、阶段提示词)。"""
        discussion_text = "\n".join(
            f"- **{d.get('title', '无标题')}** ({d.get('url', '')})\n  {d.get('snippet', '')}"
            for d in discussions
        ) or "- 未检索到相关外部讨论 (可安装 duckduckgo-search 或检查网络)"

        user_prompt = (
            "== 需求文档 ==\n" + requirement_text[:3000] +
            "\n\n== 意图识别 ==\n" + json.dumps(intent, ensure_ascii=False, indent=2) +
            "\n\n== 项目信息 ==\n" + json.dumps(project, ensure_ascii=False, indent=2) +
            "\n\n== 已阅读路径 ==\n" + "\n".join(project_context.get("read_paths", [])) +
            "\n\n== 识别到的开源软件 ==\n" + "\n".join(f"- {d}" for d in dependencies) +
            "\n\n== 网上讨论(raw) ==\n" + discussion_text +
            "\n\n== 闭环整理结果 ==\n" + refined_context +
            "\n\n== 阶段提示词 ==\n" + stage_prompt
        )

        logger.info("[报告] 生成行动报告 ...")
        body = self._call_llm(_ACTION_REPORT_PROMPT, user_prompt, temperature=0.3)
        report_path = self.reports_dir / "action_report.md"
        header = (
            "# 项目行动报告\n\n"
            f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"> 项目: {project.get('full_name', '')} ({project.get('project_type', '')})\n"
            f"> 路径: {project.get('local_path', '')}\n\n"
        )
        report_path.write_text(header + body + "\n", encoding="utf-8")
        logger.info("[报告] 已保存: %s", report_path)
        return str(report_path)

    def run_analysis_pipeline(
        self,
        requirement_path: str,
        local_project_path: Optional[str] = None,
        supplemental_path: Optional[str] = None,
        history_prompt: Optional[str] = None,
        max_repos: int = 1,
    ) -> Dict[str, Any]:
        """
        项目分析主控管线 (单步、出错即停):
          1. parse_requirement      需求表格/文档/截图识别
          2. recognize_intent       意图识别
          3. find_project           寻找项目 (开源/闭源)
          4. load_project_code      加载当前代码
          5. extract_dependencies   提取开源软件名称
          6. fetch_online_discussions 检索网上讨论
          7. refine_project_context 闭环整理项目信息
          8. build_stage_prompt     生成阶段提示词
          9. generate_action_report 输出行动报告 .md
        """
        logger.info("########## 项目分析管线启动 ##########")

        try:
            parsed = self.parse_requirement(requirement_path)
        except Exception as exc:
            logger.error("需求解析失败: %s", exc)
            return {"status": "FAILED", "stage": "parse_requirement", "error": str(exc)}

        requirement_text = parsed["text"]

        try:
            intent = self.recognize_intent_from_text(
                requirement_text,
                supplemental_path=supplemental_path,
                require_github=not bool(local_project_path),
            )
        except Exception as exc:
            logger.error("意图识别失败: %s", exc)
            return {"status": "FAILED", "stage": "recognize_intent", "error": str(exc), "requirement": parsed}

        try:
            project = self.find_project(intent, local_project_path=local_project_path, max_repos=max_repos)
        except Exception as exc:
            logger.error("寻找项目失败: %s", exc)
            return {"status": "FAILED", "stage": "find_project", "error": str(exc), "intent": intent}

        try:
            project_context = self.load_project_code(project)
        except Exception as exc:
            logger.error("加载代码失败: %s", exc)
            return {"status": "FAILED", "stage": "load_project_code", "error": str(exc), "project": project}

        try:
            dependencies = self.extract_dependencies(project["local_path"])
        except Exception as exc:
            logger.error("依赖提取失败: %s", exc)
            return {"status": "FAILED", "stage": "extract_dependencies", "error": str(exc), "project": project}

        try:
            discussions = self.fetch_online_discussions(intent)
        except Exception as exc:
            logger.error("网上讨论检索失败: %s", exc)
            discussions = []

        try:
            refined_context = self.refine_project_context(
                requirement_text, project_context, history_prompt=history_prompt
            )
        except Exception as exc:
            logger.error("闭环整理失败: %s", exc)
            return {"status": "FAILED", "stage": "refine_project_context", "error": str(exc), "project": project}

        try:
            stage_prompt = self.build_stage_prompt(
                requirement_text, intent, project_context, dependencies, discussions, history_prompt
            )
        except Exception as exc:
            logger.error("阶段提示词生成失败: %s", exc)
            return {"status": "FAILED", "stage": "build_stage_prompt", "error": str(exc), "project": project}

        try:
            report_path = self.generate_action_report(
                requirement_text, intent, project, project_context,
                dependencies, discussions, refined_context, stage_prompt,
            )
        except Exception as exc:
            logger.error("行动报告生成失败: %s", exc)
            return {"status": "FAILED", "stage": "generate_action_report", "error": str(exc), "project": project}

        result = {
            "status": "SUCCESS",
            "requirement": parsed,
            "intent": intent,
            "project": project,
            "dependencies": dependencies,
            "discussions_count": len(discussions),
            "read_paths": project_context.get("read_paths", []),
            "refined_context_path": str(self.reports_dir / "refined_context.md"),
            "stage_prompt_path": str(self.reports_dir / "stage_prompt.txt"),
            "report_path": report_path,
        }
        logger.info("########## 项目分析完成: %s ##########", report_path)
        return result

    # ------------------------------------------------------------------ #
    # 1) 代码生成 (遗留实验管线)
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
        兼容旧接口; 新管线请使用 recognize_intent_from_text / run_analysis_pipeline。
        """
        logger.info("[意图] 读取研究报告: %s", research_md_path)
        try:
            with open(research_md_path, "r", encoding="utf-8") as f:
                research_md = f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"研究报告不存在: {research_md_path}") from exc

        supplemental = blueprint_path if blueprint_path and os.path.isfile(blueprint_path) else None
        return self.recognize_intent_from_text(research_md, supplemental_path=supplemental, require_github=True)

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
    WORKSPACE = os.environ.get(
        "KAGGLE_AGENT_WORKSPACE",
        str(Path(__file__).resolve().parent / "kaggle_agent_workspace"),
    )

    sandbox = CodeSandbox(workspace_path=WORKSPACE, ssh_config=None)

    examples_dir = Path(__file__).resolve().parent / "examples"
    requirement = os.environ.get(
        "REQUIREMENT_PATH",
        str(examples_dir / "requirement_table.md"),
    )
    local_project = os.environ.get("LOCAL_PROJECT_PATH")

    result = sandbox.run_analysis_pipeline(
        requirement_path=requirement,
        local_project_path=local_project,
        supplemental_path=str(examples_dir / "blueprint.json") if (examples_dir / "blueprint.json").exists() else None,
    )
    print("\n项目分析结果:")
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("intent",)},
        ensure_ascii=False,
        indent=2,
    ))
    if result.get("status") == "SUCCESS":
        print(f"\n行动报告: {result.get('report_path')}")
