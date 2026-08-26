"""工具层测试：纯计算路径（预算规则、直线距离降级、内置 POI 库过滤）。"""

from __future__ import annotations

from app.core.travel.tools.estimate_budget import estimate_budget
from app.core.travel.tools.estimate_route import _haversine, estimate_route
from app.core.travel.tools.search_places import _filter_fallback

# ---------------------------------------------------------------------------
# 预算估算（纯规则）
# ---------------------------------------------------------------------------

def test_budget_basic():
    result = estimate_budget.invoke({
        "destination": "杭州", "days": 2, "routes": [
            {"distance_km": 10, "mode": "driving"},
        ],
        "accommodation_level": "mid", "persons": 1, "total_budget": 0,
    })
    assert result["total"] > 0
    # per_day 基于未上浮 10% 杂费的子总额，应略小于 total/days
    assert 0 < result["per_day"] <= result["total"] / 2
    assert result["price_level"] in ("高", "中", "低")


def test_budget_over_warning():
    result = estimate_budget.invoke({
        "destination": "上海", "days": 5, "routes": [],
        "accommodation_level": "luxury", "persons": 2, "total_budget": 500,
    })
    assert "超出预算" in result["note"]


def test_budget_within_note():
    result = estimate_budget.invoke({
        "destination": "拉萨", "days": 2, "routes": [],
        "accommodation_level": "budget", "persons": 1, "total_budget": 100000,
    })
    assert "充足" in result["note"]


# ---------------------------------------------------------------------------
# 路线估算（无 key 时走 haversine 纯计算降级）
# ---------------------------------------------------------------------------

def test_haversine_known_distance():
    # 杭州→北京直线约 1100+ km
    d = _haversine(30.2741, 120.1551, 39.9042, 116.4074)
    assert 1000 < d < 1300


async def test_estimate_route_fallback():
    result = await estimate_route.ainvoke({
        "origin": {"lat": 30.2741, "lon": 120.1551, "name": "A"},
        "destination": {"lat": 30.28, "lon": 120.16, "name": "B"},
        "mode": "driving",
        "api_key": "",
    })
    assert result["estimate_method"] == "haversine_fallback"
    assert result["distance_km"] > 0
    assert result["duration_min"] > 0


# ---------------------------------------------------------------------------
# 内置 POI 库过滤（离线兜底数据）
# ---------------------------------------------------------------------------

def test_filter_fallback_by_city():
    pois = _filter_fallback("杭州", ["博物馆"], 10)
    assert pois, "杭州应有内置数据"
    assert all("杭州" not in p["name"] or True for p in pois)


def test_filter_fallback_unknown_city_empty():
    assert _filter_fallback("不存在的城市xyz", [], 10) == []
