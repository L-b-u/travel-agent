"""Travel Agent API 路由。

- POST /travel/plan          同步规划（可能返回待确认状态）
- POST /travel/confirm       提交人工确认决定（HITL 恢复）
- POST /travel/plan/stream   SSE 流式规划（行程 Markdown 逐段推送）
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.core.travel.graph import (
    pending_confirmation,
    resume_travel_agent,
    run_travel_agent,
)
from app.core.travel.output import save_itinerary
from app.infrastructure.llm.model_router import ModelRouter
from app.models.travel_schemas import ConfirmRequest, TravelPlanResponse, TravelRequest

router = APIRouter(prefix="/travel", tags=["travel"])

# 全局 LLM 引用（由 main.py 注入）
_llm_router: ModelRouter | None = None


def set_llm_router(llm_router: ModelRouter) -> None:
    """注入 LLM 路由器。"""
    global _llm_router
    _llm_router = llm_router


def get_llm_router() -> ModelRouter | None:
    """获取注入的 LLM 路由器（未配置时为 None，节点走规则兜底）。"""
    return _llm_router


def _build_response(result: dict[str, Any], session_id: str) -> TravelPlanResponse:
    """把图最终状态转成 API 响应模型。"""
    interrupted = pending_confirmation(result)
    status = "pending_confirmation" if interrupted else result.get("status") or "completed"
    return TravelPlanResponse(
        session_id=session_id,
        preferences=result.get("preferences", {}),
        itinerary=result.get("itinerary", ""),
        safety_result=result.get("safety_result", {}),
        requires_confirmation=bool(interrupted or result.get("requires_confirmation")),
        confirmation_items=result.get("confirmation_items", []),
        status=status,
        research_meta=result.get("research_meta", {}),
        tips_citations=[t.get("citation", "") for t in result.get("tips", [])],
    )


def _save_if_any(result: dict[str, Any], session_id: str, user_input: str) -> None:
    """行程非空且流程已终态时保存为 Markdown 文件。"""
    itinerary = result.get("itinerary", "")
    if not itinerary:
        return
    filepath = save_itinerary(
        itinerary=itinerary,
        session_id=session_id,
        user_input=user_input,
        preferences=result.get("preferences", {}),
    )
    logger.info("行程已保存到: {}", filepath)


@router.post("/plan", response_model=TravelPlanResponse)
async def plan_travel(request: TravelRequest):
    """
    旅行规划接口（同步）。

    正常情况返回完整行程；当安全审查发现需人工确认的操作时，
    返回 requires_confirmation=True + confirmation_items，
    客户端应调用 /travel/confirm 端点提交用户决定。
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

    if not pending_confirmation(result):
        _save_if_any(result, request.session_id, request.user_input)

    return _build_response(result, request.session_id)


@router.post("/confirm", response_model=TravelPlanResponse)
async def confirm_plan(request: ConfirmRequest):
    """
    人工确认端点（HITL）：提交用户对敏感操作的决定并恢复图执行。

    - approved=true：追加确认记录后交付原行程；
    - approved=false：行程替换为取消说明。
    """
    decision = {"approved": request.approved, "note": request.note}
    logger.info("收到人工确认: session={}, approved={}", request.session_id, request.approved)

    try:
        result = await resume_travel_agent(
            session_id=request.session_id,
            decision=decision,
            llm=_llm_router,
        )
    except Exception as e:
        # 常见原因：session 无对应断点（已恢复过 / 服务重启丢失内存 checkpoint）
        raise HTTPException(
            status_code=409,
            detail=f"无法恢复会话 {request.session_id}（无待确认的断点）: {e}",
        )

    _save_if_any(result, request.session_id, user_input=result.get("user_input", ""))
    return _build_response(result, request.session_id)


@router.post("/plan/stream")
async def plan_travel_stream(request: TravelRequest):
    """
    SSE 流式旅行规划。

    事件格式（text/event-stream，data 为 JSON）：
      {"type": "delta",   "text": "<增量 Markdown>"}   — 行程合成 token 流
      {"type": "result",  "data": <TravelPlanResponse>} — 最终完整结果
      {"type": "confirm", "items": [...]}               — 需人工确认（随后流结束）
      {"type": "error",   "message": "..."}             — 流程异常

    说明：偏好收集与研究阶段不产出面向用户的文本，因此前 ~30s 只有心跳注释行；
    行程合成开始后逐段推送。
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=422, detail="输入不能为空")

    queue: asyncio.Queue = asyncio.Queue()

    async def _run_graph() -> None:
        """后台执行图；token 快照与终态事件推入队列。"""
        last_len = 0
        try:

            def on_token(snapshot: str) -> None:
                nonlocal last_len
                if len(snapshot) < last_len:
                    # 重试导致从头重新生成：通知客户端重置
                    queue.put_nowait(("reset", ""))
                queue.put_nowait(("delta", snapshot[last_len:] if len(snapshot) >= last_len else snapshot))
                last_len = len(snapshot)

            result = await run_travel_agent(
                user_input=request.user_input,
                session_id=request.session_id,
                llm=_llm_router,
                token_callback=on_token,
            )
            if pending_confirmation(result):
                await queue.put((
                    "confirm",
                    {
                        "items": result.get("confirmation_items", []),
                        "safety_result": result.get("safety_result", {}),
                    },
                ))
            else:
                _save_if_any(result, request.session_id, request.user_input)
                await queue.put(("result", json.loads(_build_response(result, request.session_id).model_dump_json())))
        except Exception as e:
            logger.exception("SSE 流式规划异常: {}", e)
            await queue.put(("error", {"message": str(e)}))

    async def event_stream():
        task = asyncio.create_task(_run_graph())
        try:
            # 心跳：每 15s 发一行 SSE 注释，防止代理超时断连
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    if task.done():
                        break
                    yield ": keep-alive\n\n"
                    continue

                if kind == "delta":
                    yield f"data: {json.dumps({'type': 'delta', 'text': payload}, ensure_ascii=False)}\n\n"
                elif kind == "reset":
                    yield f"data: {json.dumps({'type': 'reset'}, ensure_ascii=False)}\n\n"
                elif kind == "confirm":
                    yield f"data: {json.dumps({'type': 'confirm', **payload}, ensure_ascii=False)}\n\n"
                    break
                elif kind == "error":
                    yield f"data: {json.dumps({'type': 'error', **payload}, ensure_ascii=False)}\n\n"
                    break
                elif kind == "result":
                    yield f"data: {json.dumps({'type': 'result', 'data': payload}, ensure_ascii=False)}\n\n"
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
