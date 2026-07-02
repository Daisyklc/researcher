# 参考项目挖掘 Agent 设计方案

> 状态：设计阶段（待确认后实现）
> 目标：输入一个 Markdown 文件（项目结构 / 调研需求），Agent 做意图识别，联网寻找开源与闭源项目；开源项目下载到指定目录并逐个自主分析、每个项目产出一份分析 md 放根目录；闭源项目仅记录名称与链接；最终汇总成一份总结文件。

## 0. 设计选型（已确认）

- 联网搜索抓取：复用工作区内的 **crawl4ai / browser-use**
- 开源项目分析：用**子 Agent 逐个自主遍历分析**
- 模型策略：**多模型分工**
- 本文档范围：仅设计（含接口签名 + prompt 草稿），确认后再实现

## 1. 总体架构

因为选了「复用 crawl4ai/browser-use」+「子 Agent 分析」+「多模型」，整套管线从原来的纯同步 `subprocess + 单模型` 演进为 **异步 + 多 Agent + 多模型** 结构。新管线建议做成 `async`，在现有 `CodeSandbox` 里加异步方法（或独立类），再用 `asyncio.run` 包一个同步入口，兼容现有调用习惯。

```text
input.md
   │
   ▼
[1] IntentAgent (INTENT_LLM, 便宜/结构化)
     └─→ intent.json  {task, knowledge_domains, tech_stack, github_queries, web_queries}
   │
   ▼
[2] DiscoveryAgent (browser-use + DISCOVERY_LLM)
     └─→ candidates.json  [{name, url, kind, note}, ...]   # 联网搜索，初步标注 开源/闭源
   │
   ▼
[3] Classifier (host 启发式 + clone 探测)
     ├─ open_source[]  (可 clone 的 git 仓库)
     └─ closed_source[] (商业/产品页/文档站)
   │
   ├──────────────────────────────┐
   ▼                              ▼
[4a] 开源分支                    [4b] 闭源分支
  fetch_project_code (git clone)   crawl4ai 抓产品页 → CLOSED_LLM 抽取
  ProjectAnalysisAgent            {name, url, description, commercial_note}
  (每个项目一个子Agent自主遍历)     (不下载)
  → analysis_<repo>.md (根目录)
   │                              │
   └──────────────┬───────────────┘
                  ▼
[5] SummaryWriter (SUMMARY_LLM)
     └─→ SUMMARY.md  (闭源清单 + 每个开源项目摘要)
```

## 2. 多模型分工

用一个配置对象声明每个角色用什么模型，默认都走 DeepSeek，但允许按角色替换：

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 意图识别：要求稳定输出 JSON，便宜即可
    intent_model: str = "deepseek-chat"
    # 发现搜索：驱动 browser-use 的决策模型（需 OpenAI 兼容）
    discovery_model: str = "deepseek-chat"
    # 闭源页信息抽取
    closed_model: str = "deepseek-chat"
    # 开源项目深度分析（子Agent，建议用更强的推理模型）
    analysis_model: str = "deepseek-reasoner"
    # 最终总结
    summary_model: str = "deepseek-chat"

    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
```

> browser-use 走它自己的 `ChatOpenAI`（可设 `base_url` 指向 DeepSeek）；crawl4ai 的 LLM 抽取走 LiteLLM（`deepseek/deepseek-chat`）；子 Agent 用现有的 openai SDK 客户端。三处都能统一到 DeepSeek，也能各自替换。

## 3. 模块与接口签名

### 3.1 意图识别（复用 + 扩展）

复用现有 `recognize_intent`，扩展输出字段，兼容「项目结构」与「调研需求」两种输入，并额外产出通用网页搜索线索。

```python
async def recognize_intent(self, input_md_path: str) -> dict:
    """
    返回:
    {
      "task_summary": "...",
      "knowledge_domains": ["...", "..."],   # 新增: 涉及/需要的知识领域
      "tech_stack": ["pytorch", "fastapi"],
      "keywords": ["..."],
      "github_queries": ["...", "..."],      # 开源检索线索
      "web_queries": ["...", "..."]          # 新增: 通用网页/商业产品检索线索
    }
    """
```

### 3.2 发现搜索（browser-use）

用 browser-use 的 Agent 联网搜索，直接返回结构化候选清单（browser-use 支持 `output_model_schema` 结构化输出）。

```python
async def discover_candidates(
    self, intent: dict, max_candidates: int = 12
) -> list[dict]:
    """
    用 browser-use Agent 按 github_queries + web_queries 联网搜索,
    返回候选:
    [
      {"name": "...", "url": "...", "kind": "open_source|closed_source", "note": "..."},
      ...
    ]
    落盘 references/candidates.json
    """
```

browser-use 用法大致：

```python
from browser_use import Agent
from browser_use.llm import ChatOpenAI

llm = ChatOpenAI(model=cfg.discovery_model, base_url=cfg.base_url, api_key=...)
agent = Agent(task=DISCOVERY_TASK.format(...), llm=llm, output_model_schema=CandidateList)
history = await agent.run()
```

### 3.3 分类 + 闭源抓取（crawl4ai）

```python
def classify_candidates(self, candidates: list[dict]) -> tuple[list, list]:
    """
    host 启发式(github/gitlab/gitee/bitbucket/codeberg + 仓库路径) + browser-use 的 kind 标注,
    分成 open_source[] 与 closed_source[]。clone 成功与否作为最终判定。
    """

async def enrich_closed_source(self, closed: list[dict]) -> list[dict]:
    """
    对每个闭源项目用 crawl4ai 抓取其主页, 用 CLOSED_LLM 抽取:
    {"name", "url", "description", "commercial_note"}
    只存链接与信息, 不下载。
    """
```

crawl4ai 抓取部分：

```python
from crawl4ai import AsyncWebCrawler
async with AsyncWebCrawler() as crawler:
    res = await crawler.arun(url=item["url"])
    page_md = res.markdown  # 交给 CLOSED_LLM 抽取名称/简介/商业属性
```

### 3.4 开源拉取（复用）

复用现有 `fetch_project_code`：`git clone --depth 1` + 去重 + 超时逻辑已够用。

### 3.5 项目分析子 Agent（核心新增）

设计成一个**带文件工具的 ReAct 循环**，作用域锁死在单个 clone 下来的仓库目录内，用强模型自主决定读哪些文件，最后产出一份 md。

```python
class ProjectAnalysisAgent:
    """针对单个已 clone 的开源项目, 自主遍历并产出分析报告。"""

    def __init__(self, repo_path: str, model: str, client, max_steps: int = 15):
        ...

    async def analyze(self, intent: dict) -> str:
        """
        自主循环: 模型通过工具调用探索仓库, 直到给出最终 Markdown。
        返回 markdown 文本, 由外层写入 analysis_<repo>.md
        """
```

暴露给子 Agent 的工具（全部限制在 `repo_path` 内，防目录穿越）：

```python
list_dir(path=".", depth=1)     # 列目录
read_file(path, max_bytes=8000) # 读文件(截断)
find_files(glob="**/*.py")      # 按模式找文件
submit_analysis(markdown)       # 结束并提交最终报告
```

单项目分析产出的 md（放 workspace 根目录，命名 `analysis_<owner>__<repo>.md`）：
- 一句话定位
- 技术栈 / 依赖
- 代码架构（目录 + 模块职责表）
- 核心执行流程
- 可参考 / 可借鉴点（具体到文件或模块）
- 不适用 / 需注意的坑
- 与本项目意图的契合度

### 3.6 汇总落盘

```python
async def write_summary(
    self, intent: dict, open_results: list[dict],
    closed: list[dict], failed: list[dict]
) -> str:
    """
    产出 SUMMARY.md:
    - 顶部: 本次调研意图 & 知识领域
    - 闭源项目表: 名称 | 链接 | 简介
    - 开源项目: 逐个摘要 + 指向对应 analysis_<repo>.md 的相对链接
    - 失败项目: 列出 clone/分析失败的开源项目及原因(便于后续人工补看)
    返回 SUMMARY.md 路径
    """
```

### 3.7 主控编排

```python
async def discover_and_analyze(
    self,
    input_md_path: str,
    max_open_source: int = 3,
    max_closed_source: int = 5,
) -> dict:
    """
    主干出错即停; 但 4a 中单个开源项目分析失败时跳过该项目并记录, 不中断整条管线。
      1  recognize_intent
      2  discover_candidates            (browser-use)
      3  classify_candidates
      4a fetch_project_code + ProjectAnalysisAgent(逐个)  → analysis_*.md
         (单项目 clone/分析失败 → 记入 failed_projects, 继续下一个)
      4b enrich_closed_source          (crawl4ai)
      5  write_summary                  → SUMMARY.md
    """

# 同步兼容入口
def discover_and_analyze_sync(self, *args, **kwargs) -> dict:
    return asyncio.run(self.discover_and_analyze(*args, **kwargs))
```

## 4. 关键产物数据结构

```jsonc
// references/intent.json
{ "task_summary": "...", "knowledge_domains": [...], "tech_stack": [...],
  "keywords": [...], "github_queries": [...], "web_queries": [...] }

// references/candidates.json
[ { "name": "...", "url": "...", "kind": "open_source", "note": "..." } ]

// discover_and_analyze 返回
{ "status": "SUCCESS",
  "intent": {...},
  "open_source": [ {"full_name","url","stars","local_path","analysis_md"} ],
  "failed_projects": [ {"full_name","url","stage","error"} ],   // clone/分析失败的开源项目
  "closed_source": [ {"name","url","description","commercial_note"} ],
  "summary_path": "<workspace>/SUMMARY.md" }
```

## 5. 目录与产物结构

```text
<workspace>/
├── input.md                      # 输入
├── references/
│   ├── intent.json               # 意图识别结果
│   ├── candidates.json           # 搜索到的全部候选(含分类)
│   ├── <owner>__<repo>/          # 开源项目 clone 的代码
│   └── ...
├── analysis_<owner>__<repo>.md   # 每个开源项目一份分析(放根目录)
├── analysis_<...>.md
└── SUMMARY.md                    # 最终总结(闭源链接 + 开源摘要)
```

## 6. Prompt 草稿

**意图识别（INTENT_LLM）**
```text
你是资深技术调研员。输入是一份 Markdown，可能是【项目结构】或【调研需求】。
请识别其核心意图，并给出用于联网检索"可参考项目"的线索。
只输出合法 JSON（无代码围栏），字段：
task_summary, knowledge_domains[], tech_stack[], keywords[],
github_queries[](英文,精炼,适合GitHub仓库搜索),
web_queries[](适合搜索商业/闭源产品与技术方案)。
```

**发现搜索（DISCOVERY_LLM / browser-use task）**
```text
目标：围绕以下调研意图，联网检索既有的【开源】与【闭源/商业】项目各若干。
意图：{task_summary}；领域：{knowledge_domains}；关键词：{keywords}。
对每个项目输出 {name, url, kind, note}：
- kind=open_source：有公开可获取源码的仓库（GitHub/GitLab/Gitee 等）
- kind=closed_source：商业产品/SaaS/仅有官网或文档、无公开完整源码
最多返回 {max} 个，去重，优先相关度高、知名度高的。
```

**闭源信息抽取（CLOSED_LLM）**
```text
下面是某产品官网抓取的正文(Markdown)。请抽取 JSON：
{name, description(一句话), commercial_note(收费/闭源/授权等关键信息)}。
只输出 JSON，无代码围栏。
```

**项目分析子 Agent（ANALYSIS_LLM，system prompt）**
```text
你是资深代码架构分析师。你可以调用工具遍历一个开源项目的本地副本，
自主决定读取哪些文件（README、依赖清单、入口、核心模块）。
探索完成后调用 submit_analysis 提交 Markdown 报告，需包含：
一句话定位 / 技术栈 / 代码架构(目录+模块职责) / 核心执行流程 /
可借鉴点(具体到文件或模块) / 需注意的坑 / 与本调研意图 {task_summary} 的契合度。
约束：只读该仓库目录内文件；单文件读取会被截断；步数上限 {max_steps}。
```

**总结（SUMMARY_LLM）**
```text
根据调研意图、闭源项目清单、各开源项目分析摘要，产出 SUMMARY.md：
1) 调研意图与知识领域概述
2) 闭源项目表：名称 | 链接 | 简介
3) 开源项目：逐个摘要并链接到对应 analysis_<repo>.md
4) 失败项目：列出 clone/分析失败的开源项目及原因
5) 整合建议：应优先参考哪些项目/设计。
```

## 7. 依赖与集成注意点

- 新增依赖：`crawl4ai`、`browser-use`（都需 `playwright install chromium`）。现有 `requirements.txt` 只有 `openai`、`paramiko`，需补充。
- browser-use / crawl4ai 都是 **async**，主管线建议 async 化，再给一个 `asyncio.run` 同步包装，保持 `__main__` 调用风格。
- browser-use 用 Chromium 抓搜索引擎，比纯 API 重、也慢；好处是能应付需要渲染/反爬的页面，并能"像人一样"判断开源/闭源。需设 `max_steps`、超时，避免卡死（与"出错即停"哲学一致）。
- 三处模型入口不同（browser-use 的 `ChatOpenAI` / crawl4ai 的 LiteLLM / 自己的 openai client），`ModelConfig` 统一 `base_url` 和 key，分别适配。
- 子 Agent 的文件工具必须做**路径越界校验**（`resolve()` 后确认在 `repo_path` 下），否则模型可能读到仓库外文件。
- 开源/闭源最终判定：以 `git clone` 是否成功为准 —— clone 成功归开源并分析，失败或非仓库 URL 归闭源只存链接。

## 8. 错误处理策略（已确认）

- **主干出错即停**：意图识别、发现搜索、汇总等主干阶段失败则终止管线并返回 `FAILED`（沿用现有哲学）。
- **单项目失败即跳过**：4a 中某个开源项目 `clone` 或分析失败时，记入 `failed_projects`（含 `stage` 与 `error`）后继续处理下一个，不中断整条管线。
- 失败项目会在 `SUMMARY.md` 单独列出，便于后续人工补看。

## 9. 取舍与风险

- browser-use 做发现搜索**质量最高但最重**：依赖浏览器、耗时耗 token。若后续觉得慢，可退化成"crawl4ai 抓搜索结果页 + 链接抽取"的轻量路径，接口不变。
- 子 Agent 逐个分析**最深入但最贵**：N 个开源项目 = N 个多步 Agent。默认 `max_open_source=3`、`max_steps=15` 控成本（默认值已确认合适）。

## 10. 与现有代码的关系（已确认）

- **移除** `code_sandbox.py` 中的代码生成/执行相关部分（`generate_script`、`execute_local`、`execute_remote`、`run_experiment` 及其 SSH/日志辅助方法），本项目只聚焦"参考项目挖掘与分析"。
- **保留并复用**：`recognize_intent`（扩展字段）、`fetch_project_code`、`_read_readme` / `_list_tree` / `_has_command` 等工具方法。
- 建议将挖掘管线重构为独立模块（如 `reference_miner.py`）或重写后的 `CodeSandbox`，去掉与 Kaggle 实验执行相关的约束（外接硬盘挂载校验、MPS、blueprint 等）。
