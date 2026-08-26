"""研究 Agent 工具层测试：幂等缓存。"""

from __future__ import annotations

from types import SimpleNamespace

import app.core.travel.agents.research_agent as ra


async def test_search_pois_dedupes_identical_calls(monkeypatch):
    """相同参数的重复搜索应命中缓存，不打两次 API（线上观察到 LLM 重复发同参调用）。"""
    call_count = {"n": 0}

    async def counting_ainvoke(inputs, **kwargs):
        call_count["n"] += 1
        return [{"name": f"poi_{call_count['n']}", "lat": 30.0, "lon": 120.0,
                 "category": "景点", "rating": 4.5}]

    monkeypatch.setattr(ra, "search_places", SimpleNamespace(ainvoke=counting_ainvoke))

    capture: dict = {}
    trace: list = []
    tools = ra._build_react_tools(amap_key="", capture=capture, trace=trace)
    search_tool = tools[0]

    r1 = await search_tool.ainvoke({"destination": "成都", "interests": ["美食"]})
    r2 = await search_tool.ainvoke({"destination": "成都", "interests": ["美食"]})  # 相同参数
    await search_tool.ainvoke({"destination": "成都", "interests": ["博物馆"]})  # 不同参数

    assert call_count["n"] == 2      # 相同调用只打一次 API
    assert r1 == r2                  # 命中缓存返回同一结果
    assert len(capture["pois"]) == 2 # 两次真实调用各产出 1 个不同 POI
    assert any(t.get("cached") for t in trace)  # 缓存命中在轨迹里可见
