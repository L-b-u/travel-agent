"""偏好收集测试：结构化输出 Schema 校验 + 规则兜底 + LLM 失败降级。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.travel.agents.preference_collector import (
    TravelPreferences,
    _fallback_parse,
    collect_preferences_node,
)
from tests.conftest import FakeLLM

# ---------------------------------------------------------------------------
# Schema 归一化
# ---------------------------------------------------------------------------

def test_interests_alias_normalization():
    p = TravelPreferences(interests=["museum", "MUSEUM", "火锅", "不存在的兴趣"])
    assert p.interests == ["博物馆"]  # 别名映射 + 去重 + 无效过滤 → 非空补默认


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
