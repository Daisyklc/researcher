# -*- coding: utf-8 -*-
"""
model_config.py
===============
多模型分工配置 (对应 DESIGN.md 第 2 节)。

用一个配置对象声明每个角色用什么模型, 默认都走 DeepSeek,
但允许按角色替换。三处模型入口不同:
    - browser-use 的 ChatOpenAI (发现搜索)
    - crawl4ai 的 LiteLLM        (闭源页信息抽取)
    - openai SDK 客户端           (意图识别 / 子 Agent 分析 / 总结)
`ModelConfig` 统一 base_url 与 api_key, 分别适配。
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, fields

# 默认配置文件: 与本模块同目录下的 config.json (集中存放 provider / api_key / base_url / 各角色模型)
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")

# 管线中的五个模型角色
_ROLES = ("intent_model", "discovery_model", "closed_model", "analysis_model", "summary_model")

# 按【任务复杂度】给各角色分配的缺省模型档位:
#   简单任务(token 消耗小的模型) -> fast; 中等 -> standard; 复杂(性能更好的模型) -> reasoning
#     intent_model   意图识别(结构化 JSON)          简单  -> fast
#     closed_model   闭源页信息抽取                  简单  -> fast
#     summary_model  汇总总结                        中等  -> standard
#     discovery_model browser-use 多步搜索决策       中等偏复杂 -> standard
#     analysis_model 子 Agent 深度代码分析           复杂  -> reasoning
_ROLE_DEFAULT_TIER = {
    "intent_model": "fast",
    "discovery_model": "standard",
    "closed_model": "fast",
    "analysis_model": "reasoning",
    "summary_model": "standard",
}

# 档位缺失时的降级链: 某 provider 未定义该档位时, 依次回退到可用档位, 保证鲁棒。
_TIER_FALLBACKS = {
    "fast": ("standard", "chat", "strong", "reasoning"),
    "standard": ("chat", "fast", "strong", "reasoning"),
    "strong": ("standard", "chat", "reasoning", "fast"),
    "reasoning": ("strong", "standard", "chat", "fast"),
    "chat": ("standard", "fast", "strong"),
}


@dataclass
class ModelConfig:
    """各角色模型与 LLM 服务端点配置 (加载后为已解析的扁平配置)。"""

    # 意图识别: 要求稳定输出 JSON, 便宜即可
    intent_model: str = "deepseek-chat"
    # 发现搜索: 驱动 browser-use 的决策模型 (需 OpenAI 兼容)
    discovery_model: str = "deepseek-chat"
    # 闭源页信息抽取
    closed_model: str = "deepseek-chat"
    # 开源项目深度分析 (子 Agent, 建议用更强的推理模型)
    analysis_model: str = "deepseek-reasoner"
    # 最终总结
    summary_model: str = "deepseek-chat"

    # 统一的 OpenAI 兼容服务端点与鉴权
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"

    # 显式传入的 key (优先级高于环境变量); 缺省则运行时读环境变量
    api_key: str | None = None

    # 当前生效的 provider 名称 (如 deepseek / qwen), 便于日志与前缀判定
    provider: str | None = None
    # 当前 provider 的完整模型目录 (chat/reasoning/vision/video/omni 等),
    # 供后续按需取用视觉/视频等模型, 例如 cfg.models_catalog["video"]
    models_catalog: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # 从 JSON 配置文件加载 (api_key 等敏感/可扩展项集中存放, 便于后续扩展模型)
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | os.PathLike | None = None, provider: str | None = None) -> "ModelConfig":
        """
        载入配置: 优先读取 JSON 文件 (缺省为同目录 config.json), 文件不存在则用内置默认值。
        支持两种 schema:

        (A) 多 provider (推荐, 便于扩展 千问/DeepSeek 等):
            provider 的 models 按【复杂度档位】命名 (fast/standard/strong/reasoning...);
            roles 把每个角色映射到档位, 从而实现"简单任务用小模型、复杂任务用强模型"。
            {
              "active_provider": "deepseek",
              "providers": {
                "deepseek": {"base_url": ..., "api_key_env": ..., "api_key": "",
                             "models": {"fast": "deepseek-chat", "standard": "deepseek-chat",
                                        "reasoning": "deepseek-reasoner"}},
                "qwen":     {"base_url": ..., "api_key_env": ..., "api_key": "",
                             "models": {"fast": "qwen-turbo", "standard": "qwen-plus",
                                        "strong": "qwen-max", "reasoning": "qwq-plus",
                                        "vision": "qwen-vl-max", "video": "qwen-vl-max-latest",
                                        "omni": "qwen-omni-turbo"}}
              },
              "roles": {"intent_model": "fast", "closed_model": "fast",
                        "discovery_model": "standard", "summary_model": "standard",
                        "analysis_model": "reasoning"}
            }

        (B) 旧扁平 schema (兼容):
            {"base_url": ..., "api_key_env": ..., "api_key": "",
             "models": {"intent_model": "deepseek-chat", ...}}

        参数 provider 可临时覆盖 active_provider (仅对 schema A 生效)。
        """
        cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        if not cfg_path.is_file():
            return cls()
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"配置文件读取/解析失败: {cfg_path} ({exc})") from exc
        return cls.from_dict(data, provider=provider)

    @classmethod
    def from_dict(cls, data: dict, provider: str | None = None) -> "ModelConfig":
        """从字典构造 ModelConfig。自动识别多 provider(schema A) 或旧扁平(schema B)。"""
        if "providers" in data:
            return cls._from_providers(data, provider=provider)
        return cls._from_flat(data)

    @classmethod
    def _from_providers(cls, data: dict, provider: str | None = None) -> "ModelConfig":
        """解析多 provider schema: 选定 active provider, 把角色档位别名映射为真实模型名。"""
        providers = data.get("providers") or {}
        active = provider or data.get("active_provider")
        if not active:
            raise RuntimeError("配置缺少 active_provider, 且未显式指定 provider。")
        if active not in providers:
            raise RuntimeError(f"provider '{active}' 不在 providers 列表中: {list(providers)}")

        p = providers[active]
        catalog = p.get("models") or {}
        roles = data.get("roles") or {}

        def resolve(role: str) -> str:
            # roles 里可写档位别名(如 fast/standard/reasoning), 也可直接写真实模型名
            tier = roles.get(role, _ROLE_DEFAULT_TIER[role])
            if tier in catalog:
                return catalog[tier]
            # 档位缺失: 按降级链找一个该 provider 实际提供的档位
            for alt in _TIER_FALLBACKS.get(tier, ()):
                if alt in catalog:
                    return catalog[alt]
            # 仍找不到: 把 tier 当作真实模型名直接使用
            return tier

        return cls(
            base_url=p.get("base_url", cls.base_url),
            api_key_env=p.get("api_key_env", cls.api_key_env),
            api_key=(p.get("api_key") or None),
            provider=active,
            models_catalog=catalog,
            intent_model=resolve("intent_model"),
            discovery_model=resolve("discovery_model"),
            closed_model=resolve("closed_model"),
            analysis_model=resolve("analysis_model"),
            summary_model=resolve("summary_model"),
        )

    @classmethod
    def _from_flat(cls, data: dict) -> "ModelConfig":
        """解析旧扁平 schema, 只接受已知字段, 未知字段忽略。"""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known and k not in ("models", "models_catalog")}
        models = data.get("models") or {}
        for role in _ROLES:
            if models.get(role):
                kwargs[role] = models[role]
        return cls(**kwargs)

    def resolve_api_key(self) -> str:
        """解析实际使用的 API Key: 优先显式传入, 否则读环境变量。"""
        key = self.api_key or os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"缺少 API Key: 请设置环境变量 {self.api_key_env} 或在 ModelConfig 中传入 api_key。"
            )
        return key

    def litellm_model(self, model: str) -> str:
        """
        crawl4ai 的 LLM 抽取走 LiteLLM, 需要 provider 前缀。
        DeepSeek → 'deepseek/<model>'; 千问 DashScope → 'dashscope/<model>';
        其它自定义 OpenAI 兼容端点回退到 'openai/<model>'。
        """
        if "deepseek" in self.base_url:
            return f"deepseek/{model}"
        if "dashscope" in self.base_url or "aliyuncs" in self.base_url:
            return f"dashscope/{model}"
        return f"openai/{model}"
