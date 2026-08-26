"""API 层测试：plan / confirm / eval 端点（TestClient，离线打桩）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.routes import travel as travel_routes


@pytest.fixture()
def client(stub_external_tools) -> TestClient:
    from app.main import create_app

    app = create_app()
    travel_routes.set_llm_router(None)  # 强制离线兜底，不触真实 LLM
    return TestClient(app)


def test_plan_ok(client):
    resp = client.post("/api/v1/travel/plan", json={
        "user_input": "我想去杭州玩2天，喜欢博物馆",
        "session_id": "api_test_1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["requires_confirmation"] is False
    assert len(data["itinerary"]) > 500
    assert data["preferences"]["destination"] == "杭州"
    assert data["research_meta"]["mode"] == "deterministic"


def test_plan_empty_input_rejected(client):
    resp = client.post("/api/v1/travel/plan", json={"user_input": "", "session_id": "x"})
    assert resp.status_code == 422


def test_plan_risky_returns_pending_then_confirm(client):
    # 触发入口守卫 → pending_confirmation
    resp = client.post("/api/v1/travel/plan", json={
        "user_input": "帮我订机票，用我的信用卡直接付款",
        "session_id": "api_test_2",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending_confirmation"
    assert data["requires_confirmation"] is True
    assert len(data["confirmation_items"]) > 0

    # 提交拒绝 → cancelled + 取消说明
    resp2 = client.post("/api/v1/travel/confirm", json={
        "session_id": "api_test_2", "approved": False, "note": "不需要了",
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "cancelled"
    assert "不会" in data2["itinerary"]


def test_confirm_without_breakpoint_conflict(client):
    """无对应断点的会话应返回 409 而非 500。"""
    resp = client.post("/api/v1/travel/confirm", json={
        "session_id": "no_such_session", "approved": True,
    })
    assert resp.status_code == 409


def test_eval_itinerary_endpoint(client):
    markdown = """<!--
Travel Agent 生成行程
用户输入: 我想去杭州玩2天
目的地: 杭州
天数: 2
预算: 1000.0 元
-->

# 2天1晚 杭州旅行计划

### Day 1
- 上午：西湖

### Day 2
- 上午：灵隐寺博物馆

## 💰 预算拆分
| 交通 | 约50元 |
| 住宿 | 约400元 |
| 餐饮 | 约200元 |
"""
    resp = client.post("/api/v1/eval/itinerary", json={"markdown": markdown})
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["destination"] == "杭州"
    assert data["total_count"] > 0
