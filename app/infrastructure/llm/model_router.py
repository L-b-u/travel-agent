# -*- coding: utf-8 -*-
"""LLM 路由器：基于 LangChain ChatOpenAI，失败时简单重试，LLM 不可用时由各节点 try/except 自动降级。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import APIError, RateLimitError


@dataclass
class ModelConfig:
    """单路模型配置。"""

    model_id: str
    api_key: str
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRouter:
    """LLM 路由器：基于 LangChain ChatOpenAI，瞬时失败按指数退避重试。

    单模型场景下不做多模型调度；调用方（各 Agent 节点）已有 try/except 兜底，
    重试耗尽后抛出的异常会触发规则降级路径。
    """

    def __init__(self, model_configs: list[ModelConfig], *, max_retries: int = 2) -> None:
        if not model_configs:
            raise ValueError("model_configs 不能为空")

        self._configs = list(model_configs)
        self._max_retries = max(1, max_retries)
        # 使用 LangChain ChatOpenAI 作为统一 LLM 客户端
        self._clients: dict[str, ChatOpenAI] = {}
        for cfg in self._configs:
            kwargs: dict[str, Any] = {
                "api_key": cfg.api_key,
                "model": cfg.model_id,
                "timeout": 120,       # 请求超时 120 秒（行程生成约需 60-90 秒）
                "max_retries": 1,     # ChatOpenAI 内部重试 1 次（避免长时间等待）
            }
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            self._clients[cfg.model_id] = ChatOpenAI(**kwargs)

    @property
    def primary_model_id(self) -> str:
        """主模型 ID（用于日志标注）。"""
        return self._configs[0].model_id

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        """
        异步调用 LLM；瞬时错误（API/限流/超时）按指数退避重试，全部失败则抛异常。

        Args:
            messages: LangChain 消息列表（BaseMessage 或 dict 形式均可）
            **kwargs: 透传给 ChatOpenAI 的参数（如 temperature、max_tokens）

        Returns:
            LangChain AIMessage（含 usage_metadata 与 response_metadata.model_id）
        """
        cfg = self._configs[0]
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._try_model(cfg.model_id, messages, **kwargs)
            except (APIError, RateLimitError, asyncio.TimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "模型 [{}] 第 {}/{} 次调用失败: {}",
                    cfg.model_id, attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    # 指数退避：1s, 2s, 4s ...（上限 4s）
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
            except Exception as exc:
                last_error = exc
                logger.exception("模型 [{}] 未预期错误，不重试: {}", cfg.model_id, exc)
                break

        raise RuntimeError(
            f"模型 [{cfg.model_id}] 重试 {self._max_retries} 次后仍不可用"
        ) from last_error

    async def _try_model(
        self,
        model_id: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> AIMessage:
        """调用指定模型一次（流式接收，避免长请求因服务端非流式超时被断开）。

        对外仍返回完整 AIMessage，对调用方透明。
        """
        client = self._clients[model_id]
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", None)

        # ChatOpenAI 支持 temperature / max_tokens 等运行时参数
        bind_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        bind_kwargs.update(kwargs)

        try:
            # stream_usage=True：让最后一个 chunk 携带累计 usage_metadata（Token 统计前提）
            bound = client.bind(**bind_kwargs, stream_usage=True)
            # 流式接收：逐 chunk 拼接，保持连接活跃，避免超时断连
            chunks: list[str] = []
            usage_meta: dict[str, Any] | None = None
            async for chunk in bound.astream(messages):
                if chunk.content:
                    chunks.append(chunk.content)
                # 最后一个 chunk 携带累计 usage
                um = getattr(chunk, "usage_metadata", None)
                if um:
                    usage_meta = um
            full_content = "".join(chunks)
            resp = AIMessage(content=full_content)
            # 补回 Token 统计与实际模型 ID（供日志可观测性）
            if usage_meta:
                resp.usage_metadata = usage_meta
            resp.response_metadata = {"model_id": model_id}
        except Exception:
            logger.exception("LangChain ChatOpenAI 调用失败 model_id={}", model_id)
            raise

        return resp
