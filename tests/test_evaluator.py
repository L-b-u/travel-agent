"""评估器测试：中文数字变体、元信息解析、结构检查、端到端规则评估。"""

from __future__ import annotations

import pytest

from app.core.travel.eval.evaluator import (
    _check_structure,
    _chinese_number_variants,
    _int_to_chinese,
    _strip_itinerary_header,
    evaluate_itinerary,
    parse_itinerary_metadata,
)

# ---------------------------------------------------------------------------
# 中文数字
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (2, "二"), (3, "三"), (10, "十"), (12, "十二"),
    (1000, "一千"), (1500, "一千五百"), (554, "五百五十四"),
])
def test_int_to_chinese(n, expected):
    assert _int_to_chinese(n) == expected


def test_chinese_variants_includes_liang():
    variants = _chinese_number_variants(2)
    assert "二" in variants and "两" in variants


# ---------------------------------------------------------------------------
# 元信息解析
# ---------------------------------------------------------------------------

SAMPLE_HEADER = """<!--
Travel Agent 生成行程
生成时间: 2026-08-08 11:54:54
会话 ID: eval_case_001
用户输入: 我想去杭州玩2天，预算1000元，喜欢博物馆和美食
目的地: 杭州
天数: 2
预算: 1000.0 元
-->

# 杭州旅行计划
"""

def test_parse_itinerary_metadata():
    meta = parse_itinerary_metadata(SAMPLE_HEADER)
    assert meta["destination"] == "杭州"
    assert meta["days"] == 2
    assert meta["budget"] == 1000.0
    assert meta["session_id"] == "eval_case_001"


def test_strip_itinerary_header():
    body = _strip_itinerary_header(SAMPLE_HEADER)
    assert "<!--" not in body and "目的地" not in body
    assert "杭州旅行计划" in body


# ---------------------------------------------------------------------------
# 结构检查
# ---------------------------------------------------------------------------

GOOD_ITINERARY = """
# 2天1晚 杭州旅行计划

## 📋 约束摘要
- 目的地：杭州
- 天数：2天
- 兴趣：博物馆、美食

## 🗺️ 行程总览
| 时间 | 地点 | 交通 | 预算 | 备注 |
|------|------|------|------|------|
| Day1 上午 | 西湖风景区 | 步行 | 约0元 | 免费开放 |

## 📅 每日详细计划

### Day 1
- 上午：西湖风景区，沿白堤步行
- 中午：河坊街品尝杭州小吃与特色美食
- 下午：浙江省博物馆，了解江南历史文物

### Day 2
- 上午：灵隐寺
- 下午：中国茶叶博物馆

## 💰 预算拆分
| 项目 | 费用 |
|------|------|
| 交通 | 约50元 |
| 住宿 | 约400元 |
| 餐饮 | 约200元 |
| 门票 | 约80元 |
| 其他 | 约50元 |
| **总计** | **约780元** |

## ⚠️ 注意事项
1. 以上价格为估算，实际消费以当地为准
2. 建议提前查看景点开放时间，部分景点需预约
3. 出行前请关注当地天气预报，做好防晒/防雨准备
"""

def test_check_structure_pass():
    checks, details = _check_structure(GOOD_ITINERARY)
    assert checks["行程完整"]
    assert checks["有标题结构"]
    assert checks["有日程安排"]
    assert checks["有预算拆分"]
    assert checks["无越界操作"]


def test_check_structure_flags_violation():
    bad = GOOD_ITINERARY + "\n放心，我已为你完成预订。"
    checks, details = _check_structure(bad)
    assert not checks["无越界操作"]
    assert "violation_phrases" in details


def test_check_structure_rejects_tiny_text():
    checks, _ = _check_structure("# 短")
    assert not checks["行程完整"]


# ---------------------------------------------------------------------------
# 端到端规则评估（含元信息约束检查）
# ---------------------------------------------------------------------------

def test_evaluate_itinerary_end_to_end():
    # 头部元信息与正文需自洽（预算 1000 → 正文含"约1000元"）
    doc = (
        SAMPLE_HEADER.replace("预算: 1000.0 元", "预算: 780.0 元")
        + GOOD_ITINERARY
        + "\n喜欢博物馆的游客可去浙江省博物馆。"
    )
    result = evaluate_itinerary(doc)
    # 元信息里的目的地/天数/预算都应在正文中体现
    for key in ("目的地在行程中", "天数在行程中", "预算在行程中"):
        assert result["checks"].get(key), f"{key} 未通过: {result['checks']}"
    assert result["passed_count"] == result["total_count"]
