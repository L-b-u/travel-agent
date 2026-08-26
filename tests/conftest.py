"""
pytest 公共夹具。

核心原则：测试必须离线、快速、确定——所有外部依赖（高德/Open-Meteo/LLM）
在工具边界打桩，不真实发网络请求。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# 假数据
# ---------------------------------------------------------------------------

FAKE_POIS: list[dict[str, Any]] = [
    {"name": "西湖风景区", "lat": 30.2431, "lon": 120.1502, "category": "旅游景点",
     "rating": 4.8, "address": "龙井路1号", "source": "fallback"},
    {"name": "浙江省博物馆", "lat": 30.2585, "lon": 120.1400, "category": "博物馆",
     "rating": 4.7, "address": "孤山路25号", "source": "fallback"},
    {"name": "河坊街", "lat": 30.2420, "lon": 120.1670, "category": "美食",
     "rating": 4.3, "address": "上城区", "source": "fallback"},
    {"name": "灵隐寺", "lat": 30.2408, "lon": 120.0972, "category": "寺庙",
     "rating": 4.6, "address": "法云弄1号", "source": "fallback"},
    {"name": "中国茶叶博物馆", "lat": 30.2270, "lon": 120.1080, "category": "博物馆",
     "rating": 4.5, "address": "龙井路88号", "source": "fallback"},
    {"name": "京杭大运河", "lat": 30.3180, "lon": 120.1420, "category": "公园",
     "rating": 4.4, "address": "拱墅区", "source": "fallback"},
]

FAKE_ROUTES: list[dict[str, Any]] = [
    {"origin": "西湖风景区", "destination": "浙江省博物馆", "distance_km": 2.5,
     "duration_min": 8.0, "mode": "driving", "estimate_method": "haversine_fallback"},
]

FAKE_WEATHER: list[dict[str, Any]] = [
    {"date": "2026-08-27", "condition": "多云", "temp_max": 32.0, "temp_min": 25.0,
     "precipitation_prob": 20, "wind_speed": 12.0, "source": "open-meteo"},
    {"date": "2026-08-28", "condition": "小雨", "temp_max": 30.0, "temp_min": 24.0,
     "precipitation_prob": 70, "wind_speed": 15.0, "source": "open-meteo"},
]


def _fake_tool(async_fn=None, sync_fn=None):
    """构造带 ainvoke/invoke 接口的假工具（模拟 langchain Tool 对象）。"""

    async def ainvoke(inputs: dict[str, Any], **kwargs):
        return async_fn(**inputs)

    def invoke(inputs: dict[str, Any], **kwargs):
        return sync_fn(**inputs) if sync_fn else None

    return SimpleNamespace(ainvoke=ainvoke, invoke=invoke)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_pois() -> list[dict[str, Any]]:
    return [dict(p) for p in FAKE_POIS]


@pytest.fixture()
def stub_external_tools(monkeypatch):
    """
    打桩全部外部数据源（自动用于需要跑图的测试）：
    - POI 搜索 / 批量路线 / 天气 → 返回假数据；
    - 单段路线 → 走真实 haversine 降级路径（纯计算，无网络）。
    """
    monkeypatch.setattr(
        "app.core.travel.agents.research_agent.search_places",
        _fake_tool(async_fn=lambda **kw: [dict(p) for p in FAKE_POIS]),
    )
    monkeypatch.setattr(
        "app.core.travel.agents.research_agent.estimate_routes_batch",
        _fake_tool(async_fn=lambda **kw: [dict(r) for r in FAKE_ROUTES]),
    )
    monkeypatch.setattr(
        "app.core.travel.agents.weather_checker.get_weather_forecast",
        _fake_tool(async_fn=lambda **kw: [dict(w) for w in FAKE_WEATHER]),
    )


@pytest.fixture()
def offline_graph(stub_external_tools):
    """返回离线可跑的 run_travel_agent / resume_travel_agent（无 LLM）。"""
    from app.core.travel.graph import resume_travel_agent, run_travel_agent

    return run_travel_agent, resume_travel_agent


class FakeLLM:
    """测试用假 ModelRouter：结构化输出返回预置对象，文本生成返回预置文本。"""

    def __init__(self, structured_result: Any = None, text_result: str = "") -> None:
        self._structured_result = structured_result
        self._text_result = text_result
        self.structured_calls: list[Any] = []
        self.text_calls: list[Any] = []

    @property
    def chat_model(self):  # pragma: no cover - ReAct 主路径单测不覆盖
        raise RuntimeError("FakeLLM 不支持 chat_model")

    async def ainvoke(self, messages, **kwargs):
        self.text_calls.append(messages)
        return SimpleNamespace(content=self._text_result, usage_metadata={})

    async def ainvoke_structured(self, messages, schema, **kwargs):
        self.structured_calls.append((messages, schema))
        if isinstance(self._structured_result, Exception):
            raise self._structured_result
        return self._structured_result
