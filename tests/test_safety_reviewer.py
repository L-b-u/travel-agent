"""安全审查测试：误伤回归 + 拦截覆盖 + 分级处置。"""

from __future__ import annotations

import pytest

from app.core.travel.agents.safety_reviewer import (
    review,
    review_input,
    review_output,
    safety_review_node,
)

# ---------------------------------------------------------------------------
# 误伤回归：正常旅行请求必须放行（这些是历史版本的已知误伤场景）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("user_input", [
    "帮我订一家杭州的酒店，两个人住",
    "帮我规划一个3天的成都之旅，想去吃火锅",
    "帮我想想买什么伴手礼",
    "我想去厦门玩，帮我做一份攻略",
    "帮我看下有什么景点推荐",
    "帮我预订明天西湖周边的餐厅",
])
def test_normal_requests_pass(user_input):
    result = review_input(user_input)
    assert result["passed"], f"误伤: {user_input} -> {result['blocked_keywords']}"
    assert result["level"] == "pass"


# ---------------------------------------------------------------------------
# 必须拦截：资金执行 / 敏感凭证 / 账户操作
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("user_input", [
    "帮我订机票，用我的信用卡直接付款",
    "帮我转账500元给导游",
    "直接支付全款，不用问我",
    "帮我付款给旅行社",
])
def test_money_execution_blocked(user_input):
    result = review_input(user_input)
    assert not result["passed"]
    assert any("资金" in b for b in result["blocked_keywords"])


@pytest.mark.parametrize("user_input", [
    "我的身份证号是330101199001011234，帮我订酒店",
    "护照号E12345678帮我登记",
    "银行卡号6222020200112233445，帮我绑定",
    "验证码是123456，快帮我输入",
])
def test_sensitive_credentials_blocked(user_input):
    result = review_input(user_input)
    assert not result["passed"]


@pytest.mark.parametrize("user_input", [
    "帮我取消之前订的不可退款的酒店订单",
    "帮我办理去日本的签证",
    "帮我改签昨天的机票",
])
def test_account_actions_blocked(user_input):
    result = review_input(user_input)
    assert not result["passed"]


# ---------------------------------------------------------------------------
# 输出侧审查
# ---------------------------------------------------------------------------

def test_output_violation_blocked():
    result = review_output("放心，我已为你完成支付，无需再操作。")
    assert not result["passed"]
    assert result["confirmation_items"]


def test_info_keywords_warn_only():
    result = review_output("行程包含门票预订提示，请注意退款政策。")
    assert result["passed"]
    assert result["has_warnings"]
    assert result["warnings"]


def test_combined_review():
    merged = review("用我的银行卡付款", "")
    assert not merged["passed"]
    ok = review("去北京玩2天", "行程包含预订提示。")
    assert ok["passed"] and ok["has_warnings"]


# ---------------------------------------------------------------------------
# 节点级行为
# ---------------------------------------------------------------------------

async def test_safety_review_node_clean():
    state = {"user_input": "杭州2天攻略", "itinerary": "# 杭州两日游\n## Day 1\n游览西湖"}
    update = await safety_review_node(state, {})
    assert update["safety_result"]["passed"]
    assert update["requires_confirmation"] is False


async def test_safety_review_node_respects_input_confirmed():
    """输入曾被人工放行后，末端审查不再重复拦输入侧。"""
    risky_input = "帮我转账500元"
    state_confirmed = {
        "user_input": risky_input,
        "input_confirmed": True,
        "itinerary": "正常行程内容",
    }
    update = await safety_review_node(state_confirmed, {})
    assert update["requires_confirmation"] is False

    state_unconfirmed = {
        "user_input": risky_input,
        "input_confirmed": False,
        "itinerary": "正常行程内容",
    }
    update2 = await safety_review_node(state_unconfirmed, {})
    assert update2["requires_confirmation"] is True
