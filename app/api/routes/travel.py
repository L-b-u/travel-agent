# -*- coding: utf-8 -*-
"""Travel Agent API 路由。"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.travel.graph import run_travel_agent
from app.core.travel.output import save_itinerary
from app.infrastructure.llm.model_router import ModelRouter
from app.models.travel_schemas import TravelPlanResponse, TravelRequest

router = APIRouter(prefix="/travel", tags=["travel"])

# 全局 LLM 引用（由 main.py 注入）
_llm_router: Optional[ModelRouter] = None


def set_llm_router(router: ModelRouter) -> None:
    """注入 LLM 路由器。"""
    global _llm_router
    _llm_router = router


def get_llm_router() -> Optional[ModelRouter]:
    """获取注入的 LLM 路由器（未配置时为 None，节点走规则兜底）。"""
    return _llm_router


@router.post("/plan", response_model=TravelPlanResponse)
async def plan_travel(request: TravelRequest):
    """
    旅行规划接口（同步模式）。

    输入自然语言需求，返回完整旅行计划。
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=422, detail="输入不能为空")

    logger.info("收到旅行规划请求: session={}, input={}", request.session_id, request.user_input[:100])

    result = await run_travel_agent(
        user_input=request.user_input,
        session_id=request.session_id,
        llm=_llm_router,
    )

    if result.get("error"):
        logger.error("旅行规划失败: {}", result["error"])

    # 保存行程为 Markdown 文件
    itinerary = result.get("itinerary", "")
    if itinerary:
        filepath = save_itinerary(
            itinerary=itinerary,
            session_id=request.session_id,
            user_input=request.user_input,
            preferences=result.get("preferences", {}),
        )
        logger.info("行程已保存到: {}", filepath)

    return TravelPlanResponse(
        session_id=request.session_id,
        preferences=result.get("preferences", {}),
        itinerary=result.get("itinerary", ""),
        safety_result=result.get("safety_result", {}),
        requires_confirmation=result.get("requires_confirmation", False),
        confirmation_items=result.get("confirmation_items", []),
    )