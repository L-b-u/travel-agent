"""偏好收集测试：结构化输出 Schema 校验 + 规则兜底 + LLM 失败降级。"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.core.travel.agents.preference_collector import (
    TravelPreferences,
    _fallback_parse,
    _resolve_relative_date,
    _weekday_cn,
    collect_preferences_node,
)
from tests.conftest import FakeLLM

# ---------------------------------------------------------------------------
# Schema 归一化
# ---------------------------------------------------------------------------

def test_interests_alias_normalization():
    p = TravelPreferences(interests=["museum", "MUSEUM", "火锅", "不存在的兴趣"])
    # 别名映射 + 去重；"火锅"经模糊归类为美食；无法归类的丢弃
    assert p.interests == ["博物馆", "美食"]


def test_interests_empty_falls_back_to_default():
    p = TravelPreferences(interests=[])
    assert p.interests == ["景点", "美食"]


def test_days_out_of_range_rejected():
    with pytest.raises(ValidationError):
        TravelPreferences(days=99)


def test_start_date_validation():
    assert TravelPreferences(start_date="2026-09-01").start_date == "2026-09-01"
    assert TravelPreferences(start_date="明天").start_date is None
    assert TravelPreferences(start_date="").start_date is None


def test_literal_enum_enforced():
    with pytest.raises(ValidationError):
        TravelPreferences(companions="colleagues")
    with pytest.raises(ValidationError):
        TravelPreferences(accommodation="presidential")


# ---------------------------------------------------------------------------
# 规则兜底解析
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("user_input,expected", [
    ("我想去成都玩3天，预算2000元", {"destination": "成都", "days": 3, "budget": 2000.0}),
    ("周末想去北京逛逛，2天时间", {"destination": "北京", "days": 2}),
    ("大理3天，预算1500，一个人背包旅行", {"destination": "大理", "days": 3, "budget": 1500.0}),
])
def test_fallback_parse_basics(user_input, expected):
    prefs: dict = {}
    _fallback_parse(user_input, prefs)
    for key, value in expected.items():
        assert prefs[key] == value, f"{key}: {prefs.get(key)} != {value}"


def test_fallback_parse_food_keywords():
    prefs: dict = {}
    _fallback_parse("重庆3天，就想吃火锅和小面", prefs)
    assert "美食" in prefs["interests"]
    assert prefs["destination"] == "重庆"


def test_fallback_parse_companions_and_accommodation():
    prefs: dict = {}
    _fallback_parse("厦门5天情侣游，住豪华酒店", prefs)
    assert prefs["companions"] == "couple"
    assert prefs["accommodation"] == "luxury"

    prefs2: dict = {}
    _fallback_parse("西安3天穷游，学生党", prefs2)
    assert prefs2["accommodation"] == "budget"


# ---------------------------------------------------------------------------
# 节点级行为：LLM 主路径与兜底路径
# ---------------------------------------------------------------------------

async def test_node_uses_structured_output_when_available():
    llm = FakeLLM(structured_result=TravelPreferences(
        destination="西安", days=4, budget=1200,
        interests=["历史"], companions="friends",
    ))
    result = await collect_preferences_node({"user_input": "西安玩几天都行"}, {"configurable": {"llm": llm}})
    assert result["preferences"]["destination"] == "西安"
    assert result["preferences"]["days"] == 4
    assert len(llm.structured_calls) == 1


async def test_node_falls_back_on_llm_failure():
    llm = FakeLLM(structured_result=RuntimeError("provider down"))
    result = await collect_preferences_node(
        {"user_input": "我想去杭州玩2天，预算1000元，喜欢博物馆"},
        {"configurable": {"llm": llm}},
    )
    assert result["preferences"]["destination"] == "杭州"
    assert result["preferences"]["days"] == 2
    assert result["preferences"]["budget"] == 1000.0
    assert "博物馆" in result["preferences"]["interests"]


async def test_node_empty_input():
    result = await collect_preferences_node({"user_input": ""}, {"configurable": {}})
    assert result["error"] == "用户输入为空"
    assert result["preferences"]["destination"]  # 默认值仍在


async def test_node_without_llm_uses_rules():
    result = await collect_preferences_node(
        {"user_input": "成都3天，预算2000，喜欢美食"},
        {"configurable": {}},
    )
    assert result["preferences"]["destination"] == "成都"
    assert result["preferences"]["days"] == 3


# ---------------------------------------------------------------------------
# 相对日期解析
# ---------------------------------------------------------------------------

WED = date(2026, 8, 26)  # 周三


def test_weekday_cn():
    assert _weekday_cn(WED) == "周三"
    assert _weekday_cn(date(2026, 8, 29)) == "周六"


def test_this_weekend_from_wednesday():
    # 周三说"这周末" → 最近的周六 2026-08-29
    assert _resolve_relative_date("这周末我想去成都", WED) == "2026-08-29"


def test_this_saturday_explicit():
    assert _resolve_relative_date("这周六出发", WED) == "2026-08-29"


def test_tomorrow_and_day_after():
    assert _resolve_relative_date("明天出发", WED) == "2026-08-27"
    assert _resolve_relative_date("后天到西安", WED) == "2026-08-28"


def test_next_weekend():
    # 周三说"下周末" → 下周周六 2026-09-05
    assert _resolve_relative_date("下周末去重庆", WED) == "2026-09-05"


def test_no_time_expression_returns_none():
    assert _resolve_relative_date("我想去杭州玩两天", WED) is None


def test_fallback_parse_sets_start_date():
    prefs: dict = {}
    from app.core.travel.agents.preference_collector import _fallback_parse

    _fallback_parse("这周末我想去成都玩两天，预算1500元", prefs)
    assert prefs["start_date"] == date.today().isoformat() or prefs["start_date"]


# ---------------------------------------------------------------------------
# 模板路径的日期标注
# ---------------------------------------------------------------------------

def test_day_label_with_date():
    from app.core.travel.agents.itinerary_synthesizer import _day_label

    label = _day_label(0, "2026-08-29")
    assert "Day 1" in label and "8月29日" in label and "周六" in label
    label2 = _day_label(1, "2026-08-29")
    assert "Day 2" in label2 and "8月30日" in label2 and "周日" in label2


def test_day_label_without_date():
    from app.core.travel.agents.itinerary_synthesizer import _day_label

    assert _day_label(0, None) == "Day 1"


def test_template_generate_includes_dates():
    from app.core.travel.agents.itinerary_synthesizer import _template_generate

    md = _template_generate(
        prefs={"destination": "成都", "days": 2, "budget": 1500, "interests": ["美食"],
               "accommodation": "mid", "companions": "solo", "start_date": "2026-08-29"},
        pois=[{"name": "宽窄巷子", "rating": 4.5, "category": "景点"},
              {"name": "大熊猫基地", "rating": 4.8, "category": "景点"},
              {"name": "锦里", "rating": 4.4, "category": "景点"}],
        routes=[], weather=[{"date": "2026-08-29", "condition": "多云",
                             "temp_min": 22, "temp_max": 30}],
        budget={"per_day": 750},
    )
    assert "出行日期" in md and "8月29日（周六）" in md
    assert "### Day 1（8月29日 周六）" in md
    assert "### Day 2（8月30日 周日）" in md


# ---------------------------------------------------------------------------
# Schema 对 LLM null 输出的容错
# ---------------------------------------------------------------------------

def test_null_enum_fields_normalized_to_defaults():
    """LLM 对未提及项常返回 null（线上真实故障），必须归一化而非校验失败。"""
    p = TravelPreferences.model_validate({
        "destination": "重庆", "days": 2, "budget": 1500,
        "interests": ["美食"], "companions": None, "accommodation": None,
        "start_date": "2026-08-29",
    })
    assert p.companions == "solo"
    assert p.accommodation == "mid"


def test_empty_string_enum_fields_normalized():
    p = TravelPreferences(companions="", accommodation="  ")
    assert p.companions == "solo" and p.accommodation == "mid"


def test_interest_fuzzy_categorization():
    """具体名词应模糊归类到兴趣大类（线上故障：'大熊猫'被静默丢弃）。"""
    p = TravelPreferences(interests=["美食", "大熊猫"])
    assert "美食" in p.interests
    assert "自然" in p.interests  # 大熊猫 → 自然


def test_budget_includes_local_transit_base():
    from app.core.travel.tools.estimate_budget import estimate_budget

    result = estimate_budget.invoke({
        "destination": "成都", "days": 2, "routes": [],
        "accommodation_level": "mid", "persons": 1, "total_budget": 0,
    })
    # 无路线时交通费不再是 0（含市内通勤基础费）
    assert result["transport"] >= 15 * 2 * 0.8
    assert "大交通" in result["note"]


def test_interests_raw_preserved():
    """原话词单独保留（如"大熊猫"），供检索直接当关键词，不被归类丢弃。"""
    p = TravelPreferences.model_validate({
        "destination": "成都", "interests": ["美食", "自然"],
        "interests_raw": ["大熊猫", "火锅"],
    })
    assert p.interests_raw == ["大熊猫", "火锅"]
    assert p.interests == ["美食", "自然"]


async def test_search_merges_raw_interests(stub_external_tools, monkeypatch):
    """研究阶段应把原话词并入搜索关键词（大熊猫→高德能搜到熊猫基地）。"""
    import asyncio
    from types import SimpleNamespace

    import app.core.travel.agents.research_agent as ra

    seen_keywords = []

    async def fake_search(inputs, **kwargs):
        seen_keywords.append(list(inputs["interests"]))
        return [{"name": f"poi_{inputs['interests']}", "lat": 30.0, "lon": 120.0,
                 "category": "x", "rating": 4.0}]

    async def fake_routes(inputs, **kwargs):
        return [{"origin": "a", "destination": "b", "distance_km": 1,
                 "duration_min": 5, "mode": "driving"}]

    monkeypatch.setattr(ra, "search_places", SimpleNamespace(ainvoke=fake_search))
    monkeypatch.setattr(ra, "estimate_routes_batch", SimpleNamespace(ainvoke=fake_routes))

    state = {"preferences": {"destination": "成都", "days": 2,
                             "interests": ["美食"], "interests_raw": ["大熊猫"]},
             "user_input": "成都看大熊猫"}
    result = await ra._research_deterministic(
        "成都", ["美食"], 2, query_hint="成都看大熊猫", raw_interests=["大熊猫"],
    )
    assert result["pois"]
    assert "大熊猫" in seen_keywords[0] and "美食" in seen_keywords[0]
