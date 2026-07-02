# -*- coding: utf-8 -*-
"""
prompts.py
==========
参考项目挖掘 Agent 的全部 Prompt 草稿 (对应 DESIGN.md 第 6 节)。
集中管理, 便于统一调参与替换。
"""

# 意图识别 (INTENT_LLM) —— 要求严格输出 JSON
INTENT_PROMPT = """你是资深技术调研员。输入是一份 Markdown, 可能是【项目结构】或【调研需求】。
请识别其核心意图, 并给出用于联网检索"可参考项目"的线索。
只输出合法 JSON(无代码围栏), 字段:
task_summary, knowledge_domains[], tech_stack[], keywords[],
github_queries[](英文, 精炼, 适合 GitHub 仓库搜索),
web_queries[](适合搜索商业/闭源产品与技术方案)。"""


# 发现搜索 (DISCOVERY_LLM / browser-use task)
DISCOVERY_TASK = """目标: 围绕以下调研意图, 联网检索既有的【开源】与【闭源/商业】项目各若干。
意图: {task_summary}
领域: {knowledge_domains}
关键词: {keywords}
GitHub 检索线索: {github_queries}
网页检索线索: {web_queries}

对每个项目输出 {{name, url, kind, note}}:
- kind=open_source: 有公开可获取源码的仓库(GitHub/GitLab/Gitee 等)
- kind=closed_source: 商业产品/SaaS/仅有官网或文档、无公开完整源码
最多返回 {max} 个, 去重, 优先相关度高、知名度高的。
请把结果整理为结构化列表返回。"""


# 闭源信息抽取 (CLOSED_LLM)
CLOSED_PROMPT = """下面是某产品官网抓取的正文(Markdown)。请抽取 JSON:
{name, description(一句话), commercial_note(收费/闭源/授权等关键信息)}。
只输出 JSON, 无代码围栏。"""


# 项目分析子 Agent (ANALYSIS_LLM, system prompt)
ANALYSIS_SYSTEM_PROMPT = """你是资深代码架构分析师。你可以调用工具遍历一个开源项目的本地副本,
自主决定读取哪些文件(README、依赖清单、入口、核心模块)。
探索完成后调用 submit_analysis 提交 Markdown 报告, 需包含:
一句话定位 / 技术栈 / 代码架构(目录+模块职责) / 核心执行流程 /
可借鉴点(具体到文件或模块) / 需注意的坑 / 与本调研意图「{task_summary}」的契合度。
约束: 只读该仓库目录内文件; 单文件读取会被截断; 步数上限 {max_steps}。
请先用 list_dir / find_files 了解结构, 再用 read_file 精读关键文件, 最后调用 submit_analysis。"""


# 总结 (SUMMARY_LLM)
SUMMARY_PROMPT = """根据调研意图、闭源项目清单、各开源项目分析摘要, 产出 SUMMARY.md:
1) 调研意图与知识领域概述
2) 闭源项目表: 名称 | 链接 | 简介
3) 开源项目: 逐个摘要并链接到对应 analysis_<repo>.md
4) 失败项目: 列出 clone/分析失败的开源项目及原因
5) 整合建议: 应优先参考哪些项目/设计。
只输出 Markdown 正文, 不要使用代码围栏包裹整篇。"""
