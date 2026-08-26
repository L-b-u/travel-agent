"""LLM 路由器：基于 LangChain ChatOpenAI 的统一 LLM 接入层。

职责：
- 纯文本生成（ainvoke）：内部流式接收避免长请求被断连，瞬时失败按指数退避重试；
- 结构化输出（ainvoke_structured）：基于 with_structured_output，供偏好收集等节点使用；
- 工具绑定（chat_model）：暴露底层 ChatOpenAI，供 create_react_agent 做 Tool Calling；
- 可观测性：Token 用量日志；配置 LANGFUSE_* 环境变量后自动接入 Langfuse trace。

LLM 不可用时由各节点 try/except 自动降级到规则兜底。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import APIError, RateLimitError
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# 结构化输出的解析类错误：LLM 返回内容不符合 Schema（null/漏字段/多余文本），
# 具有随机性，应重试而非放弃
_STRUCTURED_PARSE_ERRORS = (OutputParserException, ValidationError)

# 流式进度回调：每收到一个 chunk 调用一次，参数为"累积全文快照"
# （传快照而非增量，重试导致的部分输出可被调用方直接覆盖）
TokenCallback = Callable[[str], None]


@dataclass
class ModelConfig:
    """单路模型配置。"""

    model_id: str
    api_key: str
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _is_bad_request(exc: Exception) -> bool:
    """判断是否为 400 类参数错误（重试无意义，应切换实现方式）。"""
    status = getattr(exc, "status_code", None)
    return status == 400 or "invalid_request_error" in str(exc).lower()


def _observability_callbacks() -> list[Any]:
    """构建可观测性回调列表。

    配置了 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 时返回 Langfuse
    CallbackHandler（自动上报每次 LLM 调用的输入/输出/Token/耗时），
    否则返回空列表。任何异常都不应阻断主流程。
    """
    try:
        from app.config import get_settings

        settings = get_settings()
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return []
        from langfuse.langchain import CallbackHandler

        logger.info("Langfuse 观测已启用: {}", settings.langfuse_host)
        return [CallbackHandler()]
    except Exception as e:  # pragma: no cover - 可选依赖缺失/未配置时静默
        logger.debug("可观测性回调未启用: {}", e)
        return []


class ModelRouter:
    """LLM 路由器：统一文本生成 / 结构化输出 / 工具绑定三种调用形态。

    单模型场景下不做多模型调度；调用方（各 Agent 节点）已有 try/except 兜底，
    重试耗尽后抛出的异常会触发规则降级路径。
    """

    def __init__(
        self,
        model_configs: list[ModelConfig],
        *,
        max_retries: int = 2,
        callbacks: list[Any] | None = None,
    ) -> None:
        if not model_configs:
            raise ValueError("model_configs 不能为空")

        self._configs = list(model_configs)
        self._max_retries = max(1, max_retries)
        # Langfuse 等 LangChain 兼容回调（可为空）
        self._callbacks = callbacks if callbacks is not None else _observability_callbacks()
        # 使用 LangChain ChatOpenAI 作为统一 LLM 客户端
        self._clients: dict[str, ChatOpenAI] = {}
        for cfg in self._configs:
            kwargs: dict[str, Any] = {
                "api_key": cfg.api_key,
                "model": cfg.model_id,
                "timeout": 120,       # 请求超时 120 秒（行程生成约需 60-90 秒）
                "max_retries": 1,     # ChatOpenAI 内部重试 1 次（外层还有指数退避重试）
            }
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            if cfg.extra:
                # 附加请求体参数（如 DeepSeek 系 {"thinking": {"type": "disabled"}} 关思考模式）
                kwargs["extra_body"] = cfg.extra
            self._clients[cfg.model_id] = ChatOpenAI(**kwargs)

    @property
    def primary_model_id(self) -> str:
        """主模型 ID（用于日志标注）。"""
        return self._configs[0].model_id

    @property
    def chat_model(self) -> ChatOpenAI:
        """底层 ChatOpenAI 实例。

        供需要原生 Runnable 能力的调用方使用：
        - create_react_agent(model, tools)：Agent 自行 bind_tools；
        - .with_structured_output(schema)、.bind_tools(tools) 等。
        """
        return self._clients[self._configs[0].model_id]

    def _run_config(self, **metadata: Any) -> dict[str, Any]:
        """构造 RunnableConfig（附带可观测性回调与标签）。"""
        cfg: dict[str, Any] = {"tags": ["travel-agent"]}
        if self._callbacks:
            cfg["callbacks"] = self._callbacks
        if metadata:
            cfg["metadata"] = metadata
        return cfg

    # ------------------------------------------------------------
    # 纯文本生成
    # ------------------------------------------------------------
    async def ainvoke(
        self,
        messages: list[Any],
        *,
        on_token: TokenCallback | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """
        异步调用 LLM 生成文本；瞬时错误按指数退避重试，全部失败则抛异常。

        Args:
            messages: LangChain 消息列表（BaseMessage 或 dict 形式均可）
            on_token: 可选流式回调，每个 chunk 调用一次，参数为累积全文快照
                （SSE 场景用；重试时会重新从零累积，调用方以最后一次为准）
            **kwargs: 透传给 ChatOpenAI 的运行时参数（temperature、max_tokens 等）

        Returns:
            LangChain AIMessage（含 usage_metadata 与 response_metadata.model_id）
        """
        cfg = self._configs[0]
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._try_model(cfg.model_id, messages, on_token=on_token, **kwargs)
            except (TimeoutError, APIError, RateLimitError) as exc:
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

    # ------------------------------------------------------------
    # 结构化输出
    # ------------------------------------------------------------
    async def ainvoke_structured(
        self,
        messages: list[Any],
        schema: type[T],
        **kwargs: Any,
    ) -> T:
        """
        结构化输出：让 LLM 按 Pydantic Schema 返回解析好的对象。

        实现方式降级链：function_calling → json_mode。
        不同 OpenAI 兼容网关支持度差异很大，实测两类典型拒绝：
        - "This response_format type is unavailable now"：不支持 json_schema/response_format；
        - "Thinking mode does not support this tool_choice"：思考型模型不允许强制指定工具。
        遇到 400 类参数错误立即切换下一种方式；超时/限流按指数退避重试当前方式。
        全部失败时抛出异常，由调用方降级（如规则兜底解析）。

        Args:
            messages: 消息列表
            schema: 输出 Pydantic 模型类
            **kwargs: 透传给 ChatOpenAI 的运行时参数

        Returns:
            schema 的实例
        """
        cfg = self._configs[0]
        last_error: Exception | None = None

        for method in ("function_calling", "json_mode"):
            for attempt in range(1, self._max_retries + 1):
                try:
                    if method == "json_mode":
                        # json_mode 不支持 strict 参数
                        structured = self.chat_model.with_structured_output(schema, method=method)
                    else:
                        structured = self.chat_model.with_structured_output(schema, strict=False, method=method)
                    t0 = asyncio.get_event_loop().time()
                    result = await structured.ainvoke(messages, config=self._run_config())
                    duration_ms = (asyncio.get_event_loop().time() - t0) * 1000
                    logger.info(
                        "结构化输出 [{}]({}): schema={}, 耗时 {:.0f}ms",
                        cfg.model_id, method, schema.__name__, duration_ms,
                    )
                    return result
                except Exception as exc:
                    last_error = exc

                    # 分类处置：换方式 / 同方式重试 / 直接放弃
                    if isinstance(exc, _STRUCTURED_PARSE_ERRORS):
                        # 解析/校验失败（LLM 输出 null、漏字段等）有随机性，同方式重试
                        delay = 0.5
                        detail = str(exc)[:150]
                    elif _is_bad_request(exc):
                        # 400 参数类错误重试无意义：立即切换实现方式
                        logger.warning(
                            "结构化输出 [{}] 不支持 {} 方式: {}",
                            cfg.model_id, method, str(exc)[:120],
                        )
                        break
                    elif isinstance(exc, (TimeoutError, APIError, RateLimitError)):
                        # 瞬时错误：指数退避后同方式重试
                        delay = min(2 ** (attempt - 1), 4)
                        detail = str(exc)[:120]
                    else:
                        logger.exception("结构化输出 [{}] 未预期错误，不重试: {}", cfg.model_id, exc)
                        return self._raise_structured(cfg.model_id, exc)

                    logger.warning(
                        "结构化输出 [{}]({}) 第 {}/{} 次失败({}): {}",
                        cfg.model_id, method, attempt, self._max_retries,
                        type(exc).__name__, detail,
                    )
                    await asyncio.sleep(delay)

        return self._raise_structured(cfg.model_id, last_error)

    def _raise_structured(self, model_id: str, last_error: Exception | None) -> T:
        raise RuntimeError(
            f"结构化输出 [{model_id}] 所有实现方式均不可用"
        ) from last_error

    # ------------------------------------------------------------
    # 内部：单次调用（流式接收）
    # ------------------------------------------------------------
    async def _try_model(
        self,
        model_id: str,
        messages: list[Any],
        *,
        on_token: TokenCallback | None = None,
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
            async for chunk in bound.astream(messages, config=self._run_config()):
                if chunk.content:
                    chunks.append(chunk.content)
                    if on_token is not None:
                        # 回调传累积快照而非增量：重试后从头重来时调用方可直接覆盖
                        on_token("".join(chunks))
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
