# -*- coding: utf-8 -*-
"""
安全审查 Agent：检查用户输入和行程中的高风险操作，标记需人工确认且拒绝自动执行。

审查两个层面：
1. 用户输入（user_input）：用户是否要求 Agent 代执行敏感操作（付款/转账/取消订单/办签证）
   或提供了敏感凭证（身份证号/护照号/银行卡号）→ 拦截，标记 requires_confirmation
2. 行程输出（itinerary）：Agent 是否越界声称"已为你付款""请提供银行卡号"等 → 拦截

检测到风险时设置 requires_confirmation=True 并记录 confirmation_items，由上层决定
如何提示用户（当前未接 LangGraph interrupt_before 中断恢复，属预留扩展）。

正常的旅行建议（"建议预订""注意退款政策""需自行办理签证"）不会被拦截。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.travel.state import TravelState

# ============================================================
# 敏感凭证关键词（用户输入或行程中出现即拦截）
# ============================================================
SENSITIVE_INFO_KEYWORDS = [
    "身份证号", "身份证号码", "护照号", "护照号码",
    "银行卡号", "信用卡号", "卡号",
    "支付密码", "银行密码", "密码",
    "验证码", "CVV", "cvv",
    "短信验证码", "手机验证码",
]

# 敏感凭证正则（匹配号码模式）
SENSITIVE_PATTERNS = [
    (r"\d{17}[\dXx]", "身份证号"),
    (r"[A-Z]\d{8}", "护照号"),
    (r"\d{16,19}", "银行卡号"),
    (r"\b\d{3}\s?\d{3,4}\s?\d{3,4}\b", "手机号"),
]

# ============================================================
# 用户输入中的代操作请求（出现即拦截）
# ============================================================
# 用户要求 Agent 代执行的资金操作
USER_PAYMENT_REQUESTS = [
    "付款", "支付", "转账", "汇款", "扣款", "代付", "代扣",
    "直接付", "帮我付", "帮我订", "帮我买", "帮我预订",
    "用我的信用卡", "用我的银行卡", "用我的卡",
]

# 用户要求 Agent 代执行的非资金操作
USER_ACTION_REQUESTS = [
    "帮我取消", "帮我退", "帮我改签", "帮我办理",
    "帮我订", "帮我下单", "帮我预约",
]

# ============================================================
# 行程输出中的越界行为（Agent 声称已执行）
# ============================================================
EXECUTION_PHRASES = [
    "已为你", "已帮你", "我帮你", "我为你",
    "代你", "代为", "代替你",
    "自动完成", "正在执行", "正在预订", "正在支付",
    "已完成预订", "已完成支付", "已完成下单", "已经预订",
    "请提供", "请输入", "请填写",
]

PAYMENT_ACTIONS = [
    "付款", "支付", "转账", "汇款", "扣款",
    "自动续费", "代付", "下单",
]

# 旅行信息性词汇（行程中出现仅提示，不拦截）
INFO_KEYWORDS = [
    "预订", "预约", "订票", "订酒店", "booking",
    "取消", "退订", "退款", "不可退款",
    "签证", "护照", "合同", "协议",
]


async def safety_review_node(state: TravelState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点 6：安全审查。

    同时检查用户输入和行程输出：
    - 用户输入：是否要求代执行敏感操作或提供敏感凭证
    - 行程输出：Agent 是否越界声称已执行敏感操作
    """
    user_input = state.get("user_input", "")
    itinerary = state.get("itinerary", "")
    text_lower = itinerary.lower()
    input_lower = user_input.lower()

    blocked: List[str] = []
    warnings: List[str] = []
    confirmation_items: List[str] = []

    # ============================================================
    # 1. 检查用户输入中的敏感凭证 → 拦截
    # ============================================================
    for keyword in SENSITIVE_INFO_KEYWORDS:
        if keyword.lower() in input_lower:
            blocked.append(f"用户输入包含敏感凭证「{keyword}」")
            confirmation_items.append(f"用户在请求中提供了「{keyword}」，需人工确认是否安全处理")

    # 1.2 正则匹配敏感号码
    for pattern, name in SENSITIVE_PATTERNS:
        if re.search(pattern, user_input):
            # 排除短数字（如"2天""1000元"）
            match = re.search(pattern, user_input)
            if match and len(match.group()) >= 8:
                blocked.append(f"用户输入包含敏感信息「{name}」")
                confirmation_items.append(f"检测到疑似「{name}」，需人工确认")

    # ============================================================
    # 2. 检查用户输入中的代操作请求 → 拦截
    # ============================================================
    # 2.1 代执行资金操作
    for phrase in USER_PAYMENT_REQUESTS:
        if phrase in user_input:
            blocked.append(f"用户请求代执行资金操作「{phrase}」")
            confirmation_items.append(f"用户要求「{phrase}」，Agent 不应代用户执行资金操作")

    # 2.2 代执行非资金操作（取消/办理/预订等）
    for phrase in USER_ACTION_REQUESTS:
        if phrase in user_input:
            blocked.append(f"用户请求代执行操作「{phrase}」")
            confirmation_items.append(f"用户要求「{phrase}」，需人工确认")

    # ============================================================
    # 3. 检查行程输出中的越界行为 → 拦截
    # ============================================================
    # 3.1 索要敏感凭证
    for keyword in SENSITIVE_INFO_KEYWORDS:
        if keyword.lower() in text_lower:
            blocked.append(f"行程中索要敏感凭证「{keyword}」")
            confirmation_items.append(f"行程中要求用户提供「{keyword}」，需人工确认")

    # 3.2 执行性短语 + 资金动作
    has_execution = any(phrase in itinerary for phrase in EXECUTION_PHRASES)
    if has_execution:
        for action in PAYMENT_ACTIONS:
            if action in itinerary:
                blocked.append(f"行程中代用户执行资金操作「{action}」")
                confirmation_items.append(f"行程中包含代用户「{action}」的表述，需人工确认")

    # ============================================================
    # 4. 旅行信息性词汇 → 仅提示（不拦截）
    # ============================================================
    seen_warnings: set = set()
    for keyword in INFO_KEYWORDS:
        if keyword.lower() in text_lower and keyword not in seen_warnings:
            seen_warnings.add(keyword)
            warnings.append(f"行程中提到「{keyword}」相关事项，建议用户自行确认")

    # 去重
    blocked = list(dict.fromkeys(blocked))
    confirmation_items = list(dict.fromkeys(confirmation_items))

    has_blocked = len(blocked) > 0
    has_warnings = len(warnings) > 0

    safety_result = {
        "passed": not has_blocked,
        "has_warnings": has_warnings,
        "blocked_keywords": blocked,
        "warnings": warnings,
        "confirmation_items": confirmation_items,
        "summary": _generate_summary(has_blocked, has_warnings, blocked, warnings),
    }

    logger.info(
        "安全审查完成: passed={}, blocked={}, warnings={}",
        safety_result["passed"],
        len(blocked),
        len(warnings),
    )

    return {
        "safety_result": safety_result,
        "requires_confirmation": has_blocked,
        "confirmation_items": confirmation_items,
    }


def _generate_summary(
    has_blocked: bool,
    has_warnings: bool,
    blocked: List[str],
    warnings: List[str],
) -> str:
    """生成审查摘要。"""
    if not has_blocked and not has_warnings:
        return "✅ 行程安全审查通过，未发现高风险操作"

    parts = []
    if has_blocked:
        parts.append(f"⚠️ 发现 {len(blocked)} 项高风险操作：")
        for b in blocked:
            parts.append(f"  - {b}")
        parts.append("以上操作需用户手动确认，系统不会自动执行")
    if has_warnings:
        parts.append(f"💡 {len(warnings)} 项提醒：")
        for w in warnings:
            parts.append(f"  - {w}")

    return "\n".join(parts)
