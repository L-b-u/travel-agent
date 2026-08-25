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


class _LLMWrapper:
    """将 ModelRouter 适配为 LangChain ainvoke 接口。"""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    async def ainvoke(self, messages: list, **kwargs):
        """调用 LLM，返回 LangChain AIMessage。"""
        return await self._router.ainvoke(messages, **kwargs)


def _get_llm() -> Optional[Any]:
    """获取 LLM 实例（具有 LangChain ainvoke 接口）。"""
    if _llm_router:
        return _LLMWrapper(_llm_router)
    return None


@router.post("/plan", response_model=TravelPlanResponse)
async def plan_travel(request: TravelRequest):
    """
    旅行规划接口（同步模式）。

    输入自然语言需求，返回完整旅行计划。
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=422, detail="输入不能为空")

    logger.info("收到旅行规划请求: session={}, input={}", request.session_id, request.user_input[:100])

    llm = _get_llm()
    result = await run_travel_agent(
        user_input=request.user_input,
        session_id=request.session_id,
        llm=llm,
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