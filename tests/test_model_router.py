"""ModelRouter 结构化输出降级链测试（全部离线打桩）。"""

from __future__ import annotations

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from app.infrastructure.llm.model_router import ModelConfig, ModelRouter


class _Schema(BaseModel):
    name: str = "x"


def _router(monkeypatch, responses):
    """构造按顺序弹出响应的假 router（拦截 with_structured_output）。"""
    router = ModelRouter([ModelConfig(model_id="fake", api_key="k")], callbacks=[])
    calls = []

    class FakeStructured:
        def __init__(self, behavior):
            self.behavior = behavior

        async def ainvoke(self, messages, config=None):
            calls.append(1)
            b = self.behavior
            if isinstance(b, Exception):
                raise b
            return b

    def fake_with_structured_output(self, schema, strict=None, method="function_calling"):
        # 作为类属性补丁，实例会被绑定为第一个参数
        idx = len(calls)
        behavior = responses[min(idx, len(responses) - 1)]
        return FakeStructured(behavior)

    monkeypatch.setattr(router.chat_model.__class__, "with_structured_output", fake_with_structured_output)
    return router, calls


async def test_parse_failure_retries_same_method(monkeypatch):
    """解析失败应同方式重试而非直接放弃。"""
    ok = _Schema(name="ok")
    router, calls = _router(monkeypatch, [OutputParserException("bad json"), ok])
    result = await router.ainvoke_structured([{"role": "user", "content": "x"}], _Schema)
    assert result.name == "ok"
    assert len(calls) == 2


async def test_bad_request_switches_method(monkeypatch):
    """400 参数错误应立即切换实现方式，不浪费重试。"""

    class Fake400(Exception):
        status_code = 400

    ok = _Schema(name="ok")
    router, calls = _router(monkeypatch, [Fake400("tool_choice not supported"), ok])
    result = await router.ainvoke_structured([{"role": "user", "content": "x"}], _Schema)
    assert result.name == "ok"
    assert len(calls) == 2  # 第一种方式只调了一次就切换


async def test_all_methods_exhausted_raises(monkeypatch):
    err = OutputParserException("always bad")
    router, _calls = _router(monkeypatch, [err])
    with pytest.raises(RuntimeError):
        await router.ainvoke_structured([{"role": "user", "content": "x"}], _Schema)
