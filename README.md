# 参考项目挖掘 Agent

输入一份 Markdown（项目结构 / 调研需求），Agent 做意图识别，联网寻找**开源**与**闭源**项目：

- 开源项目 `git clone` 到 `references/`，逐个用**子 Agent 自主遍历分析**，每个产出一份 `analysis_<owner>__<repo>.md`；
- 闭源项目用 `crawl4ai` 抓主页 + LLM 抽取，仅记录名称/链接/简介，不下载；
- 最终汇总成一份 `SUMMARY.md`。

实现依据见 [`DESIGN.md`](./DESIGN.md)。

## 架构与管线

```text
input.md
  → [1] recognize_intent      (INTENT_LLM, openai SDK)        → references/intent.json
  → [2] discover_candidates   (browser-use + DISCOVERY_LLM)   → references/candidates.json
  → [3] classify_candidates   (host 启发式 + clone 探测)
  → [4a] fetch_project_code + ProjectAnalysisAgent(逐个)       → analysis_*.md
  → [4b] enrich_closed_source (crawl4ai + CLOSED_LLM)
  → [5] write_summary         (SUMMARY_LLM)                    → SUMMARY.md
```

错误处理：**主干出错即停**；单个开源项目 clone/分析失败则记入 `failed_projects` 并跳过，不中断整条管线。

## 模块

| 文件 | 职责 |
| --- | --- |
| `config.json` | 集中存放 `api_key` / `base_url` / 各角色模型（本地文件，已 gitignore；结构见 `config.example.json`） |
| `model_config.py` | 多模型分工配置（`ModelConfig`），从 `config.json` 载入，统一 `base_url` / API Key |
| `prompts.py` | 各阶段 Prompt 草稿 |
| `project_analysis_agent.py` | 开源项目分析子 Agent（带 `list_dir`/`read_file`/`find_files`/`submit_analysis` 工具的 ReAct 循环，路径越界校验） |
| `reference_miner.py` | 主管线 `ReferenceMiner`（async，含同步入口 `discover_and_analyze_sync`） |
| `run_miner.py` | 命令行入口 |
| `input.md` | 输入示例 |

## 安装

```bash
pip install -r requirements.txt
playwright install chromium   # browser-use / crawl4ai 都依赖浏览器
```

## 配置

所有 `api_key` / `base_url` / 各角色模型集中在 `config.json`（默认与代码同目录），采用**多 provider** 结构，内置 **DeepSeek** 与 **千问(Qwen / 阿里云百炼 DashScope)** 两套预设。首次使用：

```bash
cp config.example.json config.json   # Windows: copy config.example.json config.json
```

结构说明：

- `active_provider`：当前生效的 provider（`deepseek` 或 `qwen`），切换厂商只改这一行。
- `providers.<name>`：每个厂商的 `base_url` / `api_key_env` / `api_key` 和它按**复杂度分档**的**模型目录** `models`（档位别名 → 真实模型名）。
- `roles`：五个管线角色分别用哪个**档位**，也可直接写真实模型名。

### 按任务复杂度分配模型

模型目录按复杂度分档（`fast` / `standard` / `strong` / `reasoning`），角色按其任务复杂度选档，做到**简单任务用便宜省 token 的小模型、复杂任务用性能更强的模型**：

| 角色 | 任务 | 复杂度 | 档位 | DeepSeek | 千问 |
| --- | --- | --- | --- | --- | --- |
| `intent_model` | 意图识别（结构化 JSON） | 简单 | `fast` | `deepseek-chat` | `qwen-turbo` |
| `closed_model` | 闭源页信息抽取 | 简单 | `fast` | `deepseek-chat` | `qwen-turbo` |
| `summary_model` | 汇总总结 | 中等 | `standard` | `deepseek-chat` | `qwen-plus` |
| `discovery_model` | browser-use 多步搜索决策 | 中等偏复杂 | `standard` | `deepseek-chat` | `qwen-plus` |
| `analysis_model` | 子 Agent 深度代码分析 | 复杂 | `reasoning` | `deepseek-reasoner` | `qwq-plus` |

> 想让某个角色更强/更省，只改 `roles` 里对应的档位即可（如把 `summary_model` 从 `standard` 提到 `strong`）。若某 provider 未定义某档位，代码会按降级链自动回退到可用档位。

千问额外的多模态档位：`vision`→`qwen-vl-max`（图像）、`video`→`qwen-vl-max-latest`（**视频处理**）、`omni`→`qwen-omni-turbo`（全模态），可通过 `cfg.models_catalog["video"]` 取用，供后续扩展。

切换到千问：把 `config.json` 的 `"active_provider"` 改为 `"qwen"`（或运行时 `--provider qwen`）。

Key 解析优先级：`providers.<active>.api_key` > 其 `api_key_env` 指向的环境变量。留空则读环境变量：

```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-..."      # 或千问: $env:DASHSCOPE_API_KEY="sk-..."
```

```bash
# macOS / Linux
export DEEPSEEK_API_KEY="sk-..."    # 或千问: export DASHSCOPE_API_KEY="sk-..."
```

> `config.json` 已加入 `.gitignore`，避免误提交密钥；换厂商/模型/端点只需改这个文件（或用 `--config` / `--provider` 指定）。

## 使用

运行（默认轻量 LLM 发现，快）：

```bash
python run_miner.py --input input.md --workspace . --max-open 3 --max-closed 5
```

也可在代码中调用：

```python
from model_config import ModelConfig
from reference_miner import ReferenceMiner

miner = ReferenceMiner(workspace_path=".", model_config=ModelConfig())
result = miner.discover_and_analyze_sync("input.md", max_open_source=3, max_closed_source=5)
print(result["summary_path"])
```

### 发现模式（`--discovery-mode`）

发现搜索有两种模式，通过 `--discovery-mode` 或 `ReferenceMiner(discovery_mode=...)` 选择：

| 模式 | 说明 | 速度 |
| --- | --- | --- |
| `llm`（默认） | 由发现模型基于自身知识直接列出真实、知名的可参考项目，不启动浏览器 | 快（秒级~分钟级） |
| `browser` | 用 `browser-use` 驱动真实 Chromium 联网检索；被搜索引擎验证码/反爬拦截或异常时**自动回退到 `llm`** | 慢（可能 10+ 分钟） |

```bash
# 真实浏览器联网检索(需已装 browser-use + playwright chromium)
python run_miner.py --input input.md --discovery-mode browser
```

> 无论哪种模式，开源/闭源的最终判定都以 `git clone` 是否成功为准。日常调研建议用默认 `llm`；需要发现最新/长尾项目再用 `browser`。

## 产物结构

```text
<workspace>/
├── input.md
├── references/
│   ├── intent.json               # 意图识别结果
│   ├── candidates.json           # 全部候选(含分类)
│   └── <owner>__<repo>/          # 开源项目 clone 的代码
├── analysis_<owner>__<repo>.md   # 每个开源项目一份分析
└── SUMMARY.md                    # 最终总结(闭源链接 + 开源摘要 + 失败清单)
```

## 备注

- 三处模型入口不同：browser-use 用 `ChatOpenAI`、crawl4ai 用 LiteLLM、其余用 openai `AsyncOpenAI`，`ModelConfig` 统一 `base_url` 和 Key 后分别适配。
- 子 Agent 默认 `analysis_model="deepseek-reasoner"`。若所选模型不支持 function calling，请改用支持工具调用的模型（如 `--analysis-model deepseek-chat`）。
- 子 Agent 的文件工具做了**路径越界校验**（`resolve()` 后确认仍在 `repo_path` 下），防止读到仓库外文件。
