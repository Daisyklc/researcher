# -*- coding: utf-8 -*-
"""
project_analysis_agent.py
=========================
开源项目分析子 Agent (对应 DESIGN.md 第 3.5 节)。

设计成一个【带文件工具的 ReAct 循环】: 作用域锁死在单个 clone 下来的仓库目录内,
用强模型自主决定读哪些文件, 最后调用 submit_analysis 产出一份 Markdown 报告。

暴露给子 Agent 的工具 (全部限制在 repo_path 内, 防目录穿越):
    list_dir(path=".", depth=1)     列目录
    read_file(path, max_bytes=8000) 读文件(截断)
    find_files(glob="**/*.py")      按模式找文件
    submit_analysis(markdown)       结束并提交最终报告
"""

from __future__ import annotations

import os
import json
import fnmatch
import logging
from pathlib import Path
from typing import Any, Optional

from prompts import ANALYSIS_SYSTEM_PROMPT

logger = logging.getLogger("ReferenceMiner.Analysis")

# ---- 遍历时忽略的噪声目录 ----
_IGNORE_DIRS = {".git", "__pycache__", ".github", "node_modules", ".idea", ".venv", "dist", "build"}


# =============================================================================
# 子 Agent 的工具 schema (OpenAI function calling 格式)
# =============================================================================
_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出仓库内某个目录的文件与子目录(相对仓库根)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对仓库根的目录路径, 默认为 '.'"},
                    "depth": {"type": "integer", "description": "递归深度, 默认 1"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取仓库内某个文件的内容(会被截断)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对仓库根的文件路径"},
                    "max_bytes": {"type": "integer", "description": "最多读取字节数, 默认 8000"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "按 glob 模式在仓库内查找文件, 例如 '**/*.py'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "glob": {"type": "string", "description": "glob 模式, 如 '**/*.py'"},
                },
                "required": ["glob"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_analysis",
            "description": "提交最终的 Markdown 分析报告并结束分析。",
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown": {"type": "string", "description": "完整的 Markdown 分析报告正文"},
                },
                "required": ["markdown"],
            },
        },
    },
]


class ProjectAnalysisAgent:
    """针对单个已 clone 的开源项目, 自主遍历并产出分析报告。"""

    def __init__(
        self,
        repo_path: str,
        model: str,
        client,
        max_steps: int = 15,
    ):
        """
        参数
        ----
        repo_path : str    已 clone 到本地的仓库目录
        model     : str    分析用模型 (建议强推理模型)
        client             AsyncOpenAI 客户端 (openai 兼容)
        max_steps : int    ReAct 循环步数上限
        """
        self.repo_path = Path(repo_path).resolve()
        self.model = model
        self.client = client
        self.max_steps = max_steps

    # ------------------------------------------------------------------ #
    # 路径安全: 所有工具都必须经过 _safe_path, 防目录穿越
    # ------------------------------------------------------------------ #
    def _safe_path(self, rel: str) -> Path:
        """把相对路径解析为绝对路径, 并强制确认仍在 repo_path 之内。"""
        rel = (rel or ".").strip().lstrip("/\\")
        target = (self.repo_path / rel).resolve()
        if target != self.repo_path and self.repo_path not in target.parents:
            raise ValueError(f"路径越界(仓库目录之外), 已拒绝: {rel}")
        return target

    # ------------------------------------------------------------------ #
    # 工具实现
    # ------------------------------------------------------------------ #
    def _tool_list_dir(self, path: str = ".", depth: int = 1) -> str:
        base = self._safe_path(path)
        if not base.exists():
            return f"(路径不存在: {path})"
        if base.is_file():
            return f"(这是一个文件而非目录: {path})"
        depth = max(1, min(int(depth or 1), 3))
        lines: list[str] = []
        base_depth = len(base.parts)
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
            cur = Path(root)
            rel_depth = len(cur.parts) - base_depth
            if rel_depth >= depth:
                dirs[:] = []
            rel_root = os.path.relpath(root, self.repo_path)
            for d in sorted(dirs):
                lines.append(f"[dir]  {os.path.join(rel_root, d)}")
            for fn in sorted(files):
                lines.append(f"[file] {os.path.join(rel_root, fn)}")
            if len(lines) > 200:
                lines.append("... (条目过多, 已截断)")
                break
        return "\n".join(lines) if lines else "(空目录)"

    def _tool_read_file(self, path: str, max_bytes: int = 8000) -> str:
        target = self._safe_path(path)
        if not target.exists() or not target.is_file():
            return f"(文件不存在: {path})"
        max_bytes = max(200, min(int(max_bytes or 8000), 40000))
        try:
            data = target.read_bytes()[:max_bytes]
            text = data.decode("utf-8", errors="replace")
        except OSError as exc:
            return f"(读取失败: {exc})"
        truncated = target.stat().st_size > max_bytes
        suffix = "\n... (已截断)" if truncated else ""
        return text + suffix

    def _tool_find_files(self, glob: str) -> str:
        pattern = (glob or "").strip()
        if not pattern:
            return "(未提供 glob 模式)"
        matches: list[str] = []
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), self.repo_path)
                rel_posix = rel.replace(os.sep, "/")
                if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(fn, pattern):
                    matches.append(rel_posix)
                    if len(matches) >= 100:
                        matches.append("... (匹配过多, 已截断)")
                        return "\n".join(matches)
        return "\n".join(matches) if matches else "(无匹配文件)"

    def _dispatch_tool(self, name: str, args: dict) -> str:
        try:
            if name == "list_dir":
                return self._tool_list_dir(args.get("path", "."), args.get("depth", 1))
            if name == "read_file":
                return self._tool_read_file(args["path"], args.get("max_bytes", 8000))
            if name == "find_files":
                return self._tool_find_files(args["glob"])
        except Exception as exc:  # 工具异常回传给模型, 让它换个动作
            return f"(工具执行出错: {type(exc).__name__}: {exc})"
        return f"(未知工具: {name})"

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    async def analyze(self, intent: dict) -> str:
        """
        自主循环: 模型通过工具调用探索仓库, 直到调用 submit_analysis。
        返回 markdown 文本, 由外层写入 analysis_<repo>.md。
        """
        task_summary = intent.get("task_summary", "")
        system_prompt = ANALYSIS_SYSTEM_PROMPT.format(
            task_summary=task_summary, max_steps=self.max_steps
        )

        overview = self._tool_list_dir(".", depth=2)
        user_prompt = (
            f"仓库根目录: {self.repo_path.name}\n"
            f"本次调研意图: {task_summary}\n"
            f"知识领域: {intent.get('knowledge_domains', [])}\n"
            f"技术栈线索: {intent.get('tech_stack', [])}\n\n"
            f"仓库结构预览(前两层):\n{overview}\n\n"
            "请开始探索并最终提交分析报告。"
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for step in range(self.max_steps):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    stream=False,
                )
            except Exception as exc:
                # 某些模型(如 deepseek-reasoner)不支持 function calling → 退化为无工具单轮分析
                if step == 0 and self._looks_like_tool_unsupported(exc):
                    logger.warning(
                        "[分析] %s: 模型 %s 疑似不支持工具调用, 退化为无工具分析。",
                        self.repo_path.name, self.model,
                    )
                    return await self._analyze_without_tools(intent, overview)
                raise RuntimeError(f"分析子 Agent 模型调用失败(step {step}): {exc}") from exc

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                # 模型没有调用工具, 若已有正文则视为最终报告, 否则催促一次
                if msg.content and msg.content.strip():
                    logger.info("[分析] %s: 模型直接给出正文, 作为最终报告。", self.repo_path.name)
                    return msg.content.strip()
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({
                    "role": "user",
                    "content": "请继续: 用工具探索, 完成后调用 submit_analysis 提交报告。",
                })
                continue

            # 记录 assistant 的 tool_calls 消息
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                fname = tc.function.name
                try:
                    fargs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fargs = {}

                if fname == "submit_analysis":
                    markdown = (fargs.get("markdown") or "").strip()
                    if markdown:
                        logger.info("[分析] %s: 已提交分析报告(step %d)。", self.repo_path.name, step)
                        return markdown
                    result = "(submit_analysis 未提供 markdown, 请重新提交非空报告)"
                else:
                    result = self._dispatch_tool(fname, fargs)
                    logger.info("[分析] %s: 工具 %s -> %d 字符", self.repo_path.name, fname, len(result))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:12000],
                })

        # 步数用尽仍未提交: 强制让模型基于已看内容给出一份报告
        logger.warning("[分析] %s: 达到步数上限, 强制收尾。", self.repo_path.name)
        messages.append({
            "role": "user",
            "content": "已达步数上限。请立即基于已获取的信息, 直接输出完整的 Markdown 分析报告(不要再调用工具)。",
        })
        try:
            final = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                stream=False,
            )
            return (final.choices[0].message.content or "").strip() or "(分析子 Agent 未能产出报告)"
        except Exception as exc:
            raise RuntimeError(f"分析子 Agent 收尾调用失败: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 无工具降级路径 (适配不支持 function calling 的模型, 如 deepseek-reasoner)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _looks_like_tool_unsupported(exc: Exception) -> bool:
        s = str(exc).lower()
        return any(k in s for k in ("tool", "function call", "function_call", "does not support"))

    def _gather_key_files(self, max_files: int = 6, per_file_bytes: int = 6000) -> str:
        """自动挑选关键文件(README/依赖清单/入口), 拼成上下文供无工具分析使用。"""
        candidates: list[str] = []
        # README
        for item in sorted(self.repo_path.iterdir()):
            if item.is_file() and item.name.lower().startswith("readme"):
                candidates.append(item.name)
                break
        # 依赖/入口文件模式
        for pattern in (
            "requirements.txt", "pyproject.toml", "package.json", "setup.py",
            "main.py", "app.py", "index.js", "src/index.ts", "**/__main__.py",
        ):
            hit = self._tool_find_files(pattern)
            if hit and not hit.startswith("("):
                candidates.append(hit.splitlines()[0])
            if len(candidates) >= max_files:
                break

        seen: set[str] = set()
        blocks: list[str] = []
        for rel in candidates:
            if rel in seen:
                continue
            seen.add(rel)
            content = self._tool_read_file(rel, max_bytes=per_file_bytes)
            blocks.append(f"----- 文件: {rel} -----\n{content}")
        return "\n\n".join(blocks) if blocks else "(未能读取到关键文件)"

    async def _analyze_without_tools(self, intent: dict, overview: str) -> str:
        """无工具单轮分析: 预先读入结构与关键文件, 让模型一次性产出 Markdown 报告。"""
        task_summary = intent.get("task_summary", "")
        key_files = self._gather_key_files()
        system = (
            "你是资深代码架构分析师。基于给定的仓库结构与关键文件内容, 产出一份 Markdown 分析报告, 包含: "
            "一句话定位 / 技术栈 / 代码架构(目录+模块职责) / 核心执行流程 / "
            f"可借鉴点(具体到文件或模块) / 需注意的坑 / 与本调研意图「{task_summary}」的契合度。"
            "只输出 Markdown 正文, 不要代码围栏包裹整篇。"
        )
        user = (
            f"仓库: {self.repo_path.name}\n调研意图: {task_summary}\n\n"
            f"仓库结构(前两层):\n{overview}\n\n关键文件内容:\n{key_files}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip() or "(无工具分析未产出报告)"
