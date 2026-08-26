# -*- coding: utf-8 -*-
"""
安全审查 Agent：检查用户输入和行程输出中的高风险操作。

设计原则：
1. 分级处置而非一刀切——
   - requires_confirmation：资金执行请求（"帮我付款"）、敏感凭证（身份证号/银行卡号）、
     Agent 越界表述（"已为你支付"）→ 中断流程等待人工确认（HITL）；
   - warning：预订/签证等事务性提示 → 仅提醒，不打断；
   - pass：直接放行。
2. 不误伤正常旅行请求——"帮我订酒店""帮我买票"是旅行助手的标准语境，
   只有涉及**动钱**（付款/转账/扣款）或**索要凭证**时才需要确认。
3. 纯函数 review() 承载全部检测逻辑，节点仅做状态读写，便于单元测试。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.travel.state import TravelState

# ============================================================
# 敏感凭证关键词（用户输入或行程中出现即需确认）
# ============================================================
SENSITIVE_INFO_KEYWORDS = [
    "身份证号", "身份证号码", "护照号", "护照号码",
    "银行卡号", "信用卡号",
    "支付密码", "银行密码", "密码",
    "验证码", "CVV", "cvv",
]

# 敏感号码正则（匹配号码模式）
SENSITIVE_PATTERNS = [
    (r"\d{17}[\dXx]", "身份证号"),
    (r"\b[A-Z]\d{8}\b", "护照号"),
    (r"\b\d{16,19}\b", "银行卡号"),
    (r"\b1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}\b", "手机号"),
]

# ============================================================
# 用户输入：资金执行请求（Agent 绝不代执行，出现即需确认）
# 注意：只收录明确"动钱"的表述；"帮我订/帮我买"属正常旅行语境，不触发
# ============================================================
MONEY_EXECUTION_PHRASES = [
    "帮我付款", "帮我支付", "帮我转账", "帮我汇款", "帮我交钱", "帮我充值",
    "替我付", "代我付", "代我转账", "代扣",
    "直接付款", "直接支付", "直接扣款", "自动扣费", "自动续费",
    "用我的信用卡付", "用我的银行卡付", "刷我的卡", "用我的余额",
    "用我的信用卡支付", "用我的银行卡支付", "用我的卡付款", "用我的卡支付",
]

# ============================================================
# 用户输入：敏感账户操作请求（本系统无此类能力，需转人工确认）
# ============================================================
ACCOUNT_ACTION_PHRASES = [
    "帮我取消", "帮我退订", "帮我改签", "帮我退款",
    "帮我办理签证", "帮我办签证", "代办签证",
    "帮我下单",
]

# ============================================================
# 行程输出：越界行为（Agent 声称已执行 / 索要凭证）
# ============================================================
EXECUTION_PHRASES = [
    "已为你", "已帮你", "我帮你", "我为你",
    "代你", "代为", "代替你",
    "自动完成", "正在执行", "正在预订", "正在支付",
    "已完成预订", "已完成支付", "已完成下单", "已经预订",
    "请提供你的", "请输入你的", "请填写你的",
]

PAYMENT_ACTIONS = [
    "付款", "支付", "转账", "汇款", "扣款",
    "自动续费", "代付", "下单",
]

# 旅行信息性词汇（行程中出现仅提示，不拦截）
INFO_KEYWORDS = [
    "预订", "预约", "订票", "订酒店",
    "取消", "退订", "退款", "不可退款",
    "签证", "合同", "协议",
]


def review(user_input: str, itinerary: str) -> Dict[str, Any]:
    """
    安全审查纯函数：检测用户输入与行程输出中的高风险内容。

    Returns:
        {
            "level": "pass" | "confirmation_required",
            "passed": bool,                  # True = 无需确认可直接交付
            "has_warnings": bool,
            "blocked_keywords": [...],       # 触发确认的原因列表
            "warnings": [...],               # 事务性提醒
            "confirmation_items": [...],     # 需人工确认的具体事项
        }
    """
    text_lower = itinerary.lower()
    input_lower = user_input.lower()

    blocked: List[str] = []
    warnings: List[str] = []
    confirmation_items: List[str] = []

    # ============================================================
    # 1. 用户输入中的敏感凭证 → 需确认
    # ============================================================
    for keyword in SENSITIVE_INFO_KEYWORDS:
        if keyword.lower() in input_lower:
            blocked.append(f"用户输入包含敏感凭证「{keyword}」")
            confirmation_items.append(f"检测到「{keyword}」，为保护隐私请勿向助手提供，已转入人工确认")

    # 1.2 正则匹配敏感号码（短数字如"2天""1000元"天然不满足位数要求）
    for pattern, name in SENSITIVE_PATTERNS:
        m = re.search(pattern, user_input)
        if m:
            blocked.append(f"用户输入疑似包含{name}")
            confirmation_items.append(f"检测到疑似「{name}」，已转入人工确认")

    # ============================================================
    # 2. 用户输入中的资金执行请求 → 需确认
    # ============================================================
    for phrase in MONEY_EXECUTION_PHRASES:
        if phrase in user_input:
            blocked.append(f"用户请求代执行资金操作「{phrase}」")
            confirmation_items.append(f"「{phrase}」属于资金操作，助手不会代为执行")

    # 2.2 敏感账户操作（取消订单/办签证等，本系统无此能力）
    for phrase in ACCOUNT_ACTION_PHRASES:
        if phrase in user_input:
            blocked.append(f"用户请求代执行账户操作「{phrase}」")
            confirmation_items.append(f"「{phrase}」需用户自行前往对应平台办理，已标记待确认")

    # ============================================================
    # 3. 行程输出中的越界行为 → 需确认
    # ============================================================
    # 3.1 索要敏感凭证
    for keyword in SENSITIVE_INFO_KEYWORDS:
        if keyword.lower() in text_lower:
            blocked.append(f"行程中索要敏感凭证「{keyword}」")
            confirmation_items.append(f"行程中要求用户提供「{keyword}」，已拦截待确认")

    # 3.2 执行性短语 + 资金动作（如"已为你完成支付"）
    has_execution = any(phrase in itinerary for phrase in EXECUTION_PHRASES)
    if has_execution:
        for action in PAYMENT_ACTIONS:
            if action in itinerary:
                blocked.append(f"行程中疑似代用户执行资金操作「{action}」")
                confirmation_items.append(f"行程中包含代用户「{action}」的表述，需人工确认")

    # ============================================================
    # 4. 旅行信息性词汇 → 仅提示（不拦截）
    # ============================================================
    seen_warnings: set = set()
    for keyword in INFO_KEYWORDS:
        if keyword.lower() in text_lower and keyword not in seen_warnings:
            seen_warnings.add(keyword)
            warnings.append(f"行程中提到「{keyword}」相关事项，建议自行确认")

    # 去重保序
    blocked = list(dict.fromkeys(blocked))
    confirmation_items = list(dict.fromkeys(confirmation_items))
    warnings = list(dict.fromkeys(warnings))

    return {
        "level": "confirmation_required" if blocked else "pass",
        "passed": not blocked,
        "has_warnings": len(warnings) > 0,
        "blocked_keywords": blocked,
        "warnings": warnings,
        "confirmation_items": confirmation_items,
    }


async def safety_review_node(state: TravelState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点：安全审查。

    同时检查用户输入和行程输出，结果写入 safety_result；
    有需确认项时置 requires_confirmation=True，由 human_gate 节点中断等待人工决定。
    """
    result = review(
        user_input=state.get("user_input", ""),
        itinerary=state.get("itinerary", ""),
    )

    safety_result = {
        **result,
        "summary": _generate_summary(result),
    }

    logger.info(
        "安全审查完成: level={}, blocked={}, warnings={}",
        safety_result["level"],
        len(result["blocked_keywords"]),
        len(result["warnings"]),
    )

    return {
        "safety_result": safety_result,
        "requires_confirmation": not result["passed"],
        "confirmation_items": result["confirmation_items"],
    }


def _generate_summary(result: Dict[str, Any]) -> str:
    """生成审查摘要。"""
    blocked = result["blocked_keywords"]
    warnings = result["warnings"]

    if not blocked and not warnings:
        return "✅ 行程安全审查通过，未发现高风险操作"

    parts = []
    if blocked:
        parts.append(f"⚠️ 发现 {len(blocked)} 项需人工确认的操作：")
        for b in blocked:
            parts.append(f"  - {b}")
        parts.append("以上操作不会自动执行，等待用户确认")
    if warnings:
        parts.append(f"💡 {len(warnings)} 项提醒：")
        for w in warnings:
            parts.append(f"  - {w}")

    return "\n".join(parts)
