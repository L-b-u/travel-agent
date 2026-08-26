"""
预算估算 Agent：估算旅行总费用。
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.travel.state import TravelState
from app.core.travel.tools.estimate_budget import estimate_budget


async def estimate_budget_node(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """
    节点 4b：预算估算（与天气并行）。

    根据目的地、天数、路线、住宿等级估算总费用。
    """
    preferences = state.get("preferences", {})
    destination = preferences.get("destination", "杭州")
    days = preferences.get("days", 2)
    accommodation = preferences.get("accommodation", "mid")
    total_budget = preferences.get("budget", 0)
    routes = state.get("routes", [])

    companions = preferences.get("companions", "solo")
    persons = 2 if companions == "couple" else (3 if companions == "family" else 1)

    try:
        budget = estimate_budget.invoke({
            "destination": destination,
            "days": days,
            "routes": routes,
            "accommodation_level": accommodation,
            "persons": persons,
            "total_budget": total_budget,
        })
    except Exception as e:
        logger.warning("预算估算失败: {}", e)
        budget = {
            "transport": 0, "accommodation": 0, "food": 0,
            "tickets": 0, "other": 0, "total": 0,
            "per_day": 0, "price_level": "未知", "note": f"估算失败: {e}",
        }

    logger.info("预算估算完成: 总计 {:.0f} 元", budget.get("total", 0))
    return {"budget": budget}