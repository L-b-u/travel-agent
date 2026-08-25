﻿﻿﻿# -*- coding: utf-8 -*-
"""
偏好收集 Agent：将自然语言需求转为结构化偏好。

使用 LLM 从用户输入中提取：目的地、天数、预算、兴趣、同行人、住宿偏好等。
"""

from __future__ import annotations

import json
import time
from datetime import date
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.travel.state import TravelState

PREFERENCE_SYSTEM_PROMPT = """你是一个旅行偏好分析助手。请从用户输入中提取结构化旅行偏好。

## 输出格式（仅输出 JSON，不要其他文字）
{
  "destination": "目的地城市名",
  "days": 旅行天数(整数),
  "budget": 总预算(元, 0表示无限制),
  "interests": ["兴趣1", "兴趣2"],
  "companions": "solo/couple/family/friends",
  "accommodation": "budget/mid/luxury",
  "start_date": "YYYY-MM-DD 或 null",
  "notes": "补充说明"
}

## 规则
- destination 必须是中国城市名
- days 默认为 2，范围 1-30
- budget 为 0 表示无限制
- interests 从以下选：景点、美食、博物馆、自然、历史、购物、宗教、建筑、公园、咖啡
- 只提取用户明确提到的兴趣，不要根据目的地推断用户未提及的兴趣
- companions 默认为 solo
- accommodation 默认为 mid
- 缺失字段使用合理默认值
- start_date: 用户说"今天"则用当前日期，"明天"用当前日期+1，未提及则 null"""

PREFERENCE_QUESTIONS = [
    "你想去哪个城市？",
    "计划玩几天？",
    "预算大概多少？（可选）",
    "有什么特别的兴趣偏好吗？（如博物馆、美食、自然风光等）",
]


async def collect_preferences_node(state: TravelState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点 1：偏好收集。

    使用 LLM 将用户自然语言输入转为结构化偏好 JSON。
    如果输入信息不足，生成追问。
    """
    user_input = state.get("user_input", "")

    preferences: Dict[str, Any] = {
        "destination": "杭州",
        "days": 2,
        "budget": 0,
        "interests": ["景点", "美食"],
        "companions": "solo",
        "accommodation": "mid",
        "start_date": None,
        "notes": "",
    }

    if not user_input.strip():
        return {"preferences": preferences, "error": "用户输入为空"}

    # 尝试使用 LLM 提取结构化偏好（LLM 通过 RunnableConfig 注入，不进 state）
    llm = config.get("configurable", {}).get("llm")
    llm_succeeded = False
    if llm:
        t0 = time.perf_counter()
        try:
            messages = [
                {"role": "system", "content": PREFERENCE_SYSTEM_PROMPT},
                {"role": "user", "content": f"当前日期：{date.today().isoformat()}\n用户需求：\n{user_input}"},
            ]
            # 使用 LangChain ainvoke 接口（兼容 BaseMessage / dict）
            resp = await llm.ainvoke(messages, temperature=0.2)
            duration_ms = (time.perf_counter() - t0) * 1000
            raw = resp.content if hasattr(resp, "content") else str(resp)
            # 提取 JSON
            json_match = _extract_json(raw)
            if json_match:
                parsed = json_match
                preferences["destination"] = str(parsed.get("destination", preferences["destination"]))
                preferences["days"] = max(1, min(30, int(parsed.get("days", preferences["days"]))))
                preferences["budget"] = max(0, float(parsed.get("budget", preferences["budget"])))
                if parsed.get("interests"):
                    preferences["interests"] = list(parsed["interests"])
                preferences["companions"] = str(parsed.get("companions", preferences["companions"]))
                preferences["accommodation"] = str(parsed.get("accommodation", preferences["accommodation"]))
                preferences["start_date"] = parsed.get("start_date")
                preferences["notes"] = str(parsed.get("notes", ""))
                llm_succeeded = True
            else:
                logger.warning("LLM 返回内容无法解析为 JSON，降级到规则兜底")
            usage = getattr(resp, "usage_metadata", None) or {}
            logger.info(
                "偏好收集 LLM: 输入{}/输出{} token, 耗时 {:.0f}ms",
                usage.get("input_tokens", 0), usage.get("output_tokens", 0), duration_ms,
            )
        except Exception as e:
            logger.warning("LLM 偏好提取失败，使用规则兜底: {} ({:.0f}ms)", e, (time.perf_counter() - t0) * 1000)

    # 规则兜底：仅在 LLM 未成功时执行
    if not llm_succeeded:
        _fallback_parse(user_input, preferences)

    logger.info(
        "偏好收集完成: 目的地={}, 天数={}, 预算={}, 兴趣={}",
        preferences["destination"],
        preferences["days"],
        preferences["budget"],
        preferences["interests"],
    )

    return {"preferences": preferences}


def _fallback_parse(user_input: str, prefs: Dict[str, Any]) -> None:
    """基于规则的偏好提取兜底。"""
    import re

    # 提取目的地（从已知城市列表匹配，避免误用默认值"杭州"）
    from app.core.travel.tools._poi_data import CITY_COORDS
    for city in CITY_COORDS:
        if city in user_input:
            prefs["destination"] = city
            break

    # 提取天数
    days_patterns = [
        r"(\d+)\s*天", r"(\d+)\s*日", r"玩\s*(\d+)", r"(\d+)\s*晚",
    ]
    for pat in days_patterns:
        m = re.search(pat, user_input)
        if m:
            prefs["days"] = max(1, min(30, int(m.group(1))))
            break

    # 提取预算
    budget_patterns = [
        r"预算\s*(\d+)", r"(\d+)\s*元", r"(\d+)\s*块", r"不超过\s*(\d+)",
    ]
    for pat in budget_patterns:
        m = re.search(pat, user_input)
        if m:
            prefs["budget"] = float(m.group(1))
            break

    # 提取兴趣
    interest_keywords = {
        "博物馆": "博物馆", "museum": "博物馆",
        "美食": "美食", "吃": "美食", "小吃": "美食", "餐厅": "美食",
        "自然": "自然", "风景": "自然", "美景": "自然", "山水": "自然", "公园": "自然", "海": "自然",
        "购物": "购物", "买": "购物", "逛街": "购物",
        "历史": "历史", "古迹": "历史", "古城": "历史", "古镇": "历史",
        "宗教": "宗教", "寺庙": "宗教", "教堂": "宗教", "佛": "宗教",
        "咖啡": "咖啡", "cafe": "咖啡",
        "建筑": "建筑",
    }
    detected = []
    for kw, cat in interest_keywords.items():
        if kw.lower() in user_input.lower() and cat not in detected:
            detected.append(cat)
    if detected:
        prefs["interests"] = detected

    # 同行人
    if any(w in user_input for w in ["情侣", "女朋友", "男朋友", "对象", "couple"]):
        prefs["companions"] = "couple"
    elif any(w in user_input for w in ["家庭", "孩子", "小孩", "亲子", "family"]):
        prefs["companions"] = "family"
    elif any(w in user_input for w in ["朋友", "同学", "朋友", "friends", "闺蜜", "兄弟"]):
        prefs["companions"] = "friends"

    # 住宿偏好
    if any(w in user_input for w in ["豪华", "五星", "高端", "luxury"]):
        prefs["accommodation"] = "luxury"
    elif any(w in user_input for w in ["经济", "便宜", "青旅", "穷游", "budget"]):
        prefs["accommodation"] = "budget"


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取 JSON 对象。"""
    import re

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None