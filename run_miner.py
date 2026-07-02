# -*- coding: utf-8 -*-
"""
run_miner.py
============
参考项目挖掘 Agent 的命令行入口。

用法:
    # 需先设置 API Key
    #   Windows PowerShell:  $env:DEEPSEEK_API_KEY="sk-..."
    #   bash:                export DEEPSEEK_API_KEY="sk-..."
    python run_miner.py --input input.md --workspace . --max-open 3 --max-closed 5
"""

from __future__ import annotations

import json
import argparse

from model_config import ModelConfig
from reference_miner import ReferenceMiner


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="参考项目挖掘 Agent")
    p.add_argument("--input", "-i", default="input.md", help="输入 Markdown 路径(项目结构/调研需求)")
    p.add_argument("--workspace", "-w", default=".", help="工作区根目录(产物落此)")
    p.add_argument("--max-open", type=int, default=3, help="最多分析的开源项目数")
    p.add_argument("--max-closed", type=int, default=5, help="最多记录的闭源项目数")
    p.add_argument("--config", "-c", default=None, help="配置文件路径(默认同目录 config.json)")
    p.add_argument(
        "--discovery-mode", choices=["llm", "browser"], default="llm",
        help="发现模式: llm(默认,快,不开浏览器) / browser(真实浏览器检索,被拦截时回退 llm)",
    )
    p.add_argument("--provider", default=None, help="覆盖 active_provider(如 deepseek / qwen)")
    p.add_argument("--base-url", default=None, help="覆盖 LLM 服务 base_url")
    p.add_argument("--api-key-env", default=None, help="覆盖读取 API Key 的环境变量名")
    p.add_argument("--intent-model", default=None)
    p.add_argument("--discovery-model", default=None)
    p.add_argument("--closed-model", default=None)
    p.add_argument("--analysis-model", default=None)
    p.add_argument("--summary-model", default=None)
    return p


def build_config(args: argparse.Namespace) -> ModelConfig:
    # 以 config.json 为基线, 命令行参数按需覆盖
    cfg = ModelConfig.load(args.config, provider=args.provider)
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key_env:
        cfg.api_key_env = args.api_key_env
    for role in ("intent", "discovery", "closed", "analysis", "summary"):
        val = getattr(args, f"{role}_model")
        if val:
            setattr(cfg, f"{role}_model", val)
    return cfg


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = build_config(args)
    miner = ReferenceMiner(
        workspace_path=args.workspace,
        model_config=cfg,
        discovery_mode=args.discovery_mode,
    )

    result = miner.discover_and_analyze_sync(
        input_md_path=args.input,
        max_open_source=args.max_open,
        max_closed_source=args.max_closed,
    )

    print("\n===== 挖掘结果 =====")
    printable = {k: v for k, v in result.items() if k != "intent"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
