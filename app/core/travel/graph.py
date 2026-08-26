"""
LangGraph 状态图：编排 Travel Agent 流水线。

流程：
    input_guard ─┬─(风险)──────────────→ human_gate ─┬─(批准·继续)→ collect_preferences
                 └─(通过)→ collect_preferences         ├─(批准·行程已成)→ END
                          → research（ReAct 工具调用）  └─(拒绝)→ END（取消说明）
                          → check_weather ──┐ (Fan-out 并行)
                          → estimate_budget ┘
                          → synthesize → safety_review ─┬─(风险)→ human_gate
                                                        └─(通过)→ END

设计要点：
- fail-fast：入口守卫在消耗任何 LLM/API 资源前拦截输入侧风险；
- 研究阶段由 LLM 通过 Tool Calling 自主探索；天气与预算是关键数据，
  走确定性并行节点保证覆盖；
- 双层审查 + HITL：输入侧在入口、输出侧在合成后；命中即 interrupt() 中断，
  人工批准后输入侧风险可继续规划（input_confirmed 防止重复拦截）。
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from loguru import logger

from app.core.travel.agents import (
    check_weather_node,
    collect_preferences_node,
    estimate_budget_node,
    research_node,
    safety_review_node,
    synthesize_node,
)
from app.core.travel.agents.safety_reviewer import input_guard_node
from app.core.travel.state import TravelState


def build_travel_graph() -> StateGraph:
    """构建 Travel Agent 的 LangGraph 状态图（MemorySaver 按 thread_id 隔离会话）。"""
    workflow = StateGraph(TravelState)

    # ---- 添加节点 ----
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("collect_preferences", collect_preferences_node)
    workflow.add_node("research", research_node)
    workflow.add_node("check_weather", check_weather_node)
    workflow.add_node("estimate_budget", estimate_budget_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("safety_review", safety_review_node)
    workflow.add_node("human_gate", human_gate_node)

    # ---- 设置边 ----
    workflow.set_entry_point("input_guard")

    # 入口守卫：命中风险直接进人工门（fail-fast），否则开始规划
    workflow.add_conditional_edges(
        "input_guard",
        _route_after_input_guard,
        {"gate": "human_gate", "plan": "collect_preferences"},
    )

    # 规划主链路：偏好 → 研究（Tool Calling）→ 天气/预算并行 → 合成 → 输出侧审查
    workflow.add_edge("collect_preferences", "research")
    workflow.add_edge("research", "check_weather")
    workflow.add_edge("research", "estimate_budget")
    workflow.add_edge("check_weather", "synthesize")
    workflow.add_edge("estimate_budget", "synthesize")
    workflow.add_edge("synthesize", "safety_review")
    workflow.add_conditional_edges(
        "safety_review",
        _route_after_safety,
        {"gate": "human_gate", "end": END},
    )

    # 人工门：拒绝→结束；批准且行程已生成→结束；批准但输入侧拦截（无行程）→继续规划
    workflow.add_conditional_edges(
        "human_gate",
        _route_after_gate,
        {"continue_plan": "collect_preferences", "end": END},
    )

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    logger.info("Travel Agent LangGraph 编译完成")
    return graph


def _route_after_input_guard(state: TravelState) -> str:
    """入口守卫路由：有需确认项则进人工门。"""
    return "gate" if state.get("requires_confirmation") else "plan"


def _route_after_safety(state: TravelState) -> str:
    """输出侧审查后的路由：有需确认项则进人工门。"""
    return "gate" if state.get("requires_confirmation") else "end"


def _route_after_gate(state: TravelState) -> str:
    """人工门之后的路由：
    - 批准且已有行程 → 交付；
    - 批准但行程为空（输入侧拦截后放行）→ 继续规划主流程；
    - 拒绝 → 结束（取消说明已写入行程字段）。
    """
    decision = state.get("confirmation_decision", {})
    if decision.get("approved") and not state.get("itinerary"):
        return "continue_plan"
    return "end"


async def human_gate_node(state: TravelState) -> dict[str, Any]:
    """
    人工确认门（HITL）。

    审查（入口守卫或输出侧审查）发现需确认项时，通过 interrupt() 中断图执行，
    将待确认事项抛给调用方；调用方以 Command(resume={"approved": bool, "note": str})
    恢复。resume 后本节点从头重跑，第二次执行时 interrupt() 返回用户决定。

    - 批准 + 行程已生成：追加"已人工确认"记录后交付；
    - 批准 + 行程为空（入口守卫拦截）：标记 input_confirmed 后回到规划主流程；
    - 拒绝：写入取消说明（明确未执行任何敏感操作），status="cancelled"。
    """
    if not state.get("requires_confirmation"):
        return {"status": "completed"}

    decision = interrupt({
        "type": "safety_confirmation",
        "question": "请求包含需要你确认的操作，确认后才会继续",
        "items": state.get("confirmation_items", []),
    })

    # 兼容 bool / dict 两种恢复值
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        note = str(decision.get("note", ""))
    else:
        approved = bool(decision)
        note = ""

    has_itinerary = bool(state.get("itinerary"))
    logger.info("人工确认结果: approved={}, has_itinerary={}, note={}", approved, has_itinerary, note)

    if approved and has_itinerary:
        stamp = (
            "\n\n---\n\n> ✅ **人工确认记录**：以上请求中的敏感操作已由用户确认。\n"
            "> 注意：系统不会代为执行任何资金/账户操作，相关事项请自行办理。\n"
            + (f"> 用户备注：{note}\n" if note else "")
        )
        return {
            "confirmation_decision": {"approved": True, "note": note},
            "itinerary": state.get("itinerary", "") + stamp,
            "status": "completed",
        }

    if approved:
        # 输入侧拦截被放行：继续完整规划；末端审查不再重复拦输入侧
        return {
            "confirmation_decision": {"approved": True, "note": note},
            "input_confirmed": True,
            "requires_confirmation": False,
            "confirmation_items": [],
        }

    items = state.get("confirmation_items", []) or ["未明确列出的事项"]
    item_lines = "\n".join(f"- {i}" for i in items)
    return {
        "confirmation_decision": {"approved": False, "note": note},
        "itinerary": (
            "## ❌ 已按你的要求终止本次规划\n\n"
            "以下操作涉及资金或账户安全，**系统不会也不会被授权代为执行**：\n\n"
            f"{item_lines}\n\n"
            "如仍需旅行规划建议，请去掉上述敏感操作后重新描述你的需求。\n"
            + (f"\n> 用户备注：{note}\n" if note else "")
        ),
        "status": "cancelled",
    }


# 全局单例
_travel_graph: StateGraph | None = None


def get_travel_graph() -> StateGraph:
    """获取 Travel Agent 图单例。"""
    global _travel_graph
    if _travel_graph is None:
        _travel_graph = build_travel_graph()
    return _travel_graph


def reset_travel_graph() -> None:
    """重置图单例（测试用：更换节点实现后重建图）。"""
    global _travel_graph
    _travel_graph = None


async def run_travel_agent(
    user_input: str,
    session_id: str = "default",
    llm: Any = None,
    token_callback: Any = None,
) -> dict[str, Any]:
    """
    运行 Travel Agent 全流程。

    Args:
        user_input: 用户自然语言输入
        session_id: 会话 ID（同时作为 checkpointer 的 thread_id）
        llm: ModelRouter 实例（含 ainvoke / ainvoke_structured / chat_model），可选
        token_callback: 可选流式回调，行程合成节点的 LLM 每 output 一个 chunk 调用一次
            （参数为累积全文快照），供 SSE 推送；不影响非流式调用方

    Returns:
        最终状态字典；若触发人工确认，包含 requires_confirmation 与 confirmation_items，
        可通过 resume_travel_agent 传入用户决定继续。
    """
    graph = get_travel_graph()

    config = {
        "configurable": {
            "thread_id": session_id,
            "llm": llm,
            "token_callback": token_callback,
        }
    }

    initial_state: TravelState = {
        "user_input": user_input,
        "session_id": session_id,
        "preferences": {},
        "pois": [],
        "routes": [],
        "research_summary": "",
        "research_trace": [],
        "research_meta": {},
        "tips": [],
        "weather": [],
        "budget": {},
        "itinerary": "",
        "safety_result": {},
        "requires_confirmation": False,
        "confirmation_items": [],
        "confirmation_decision": {},
        "input_confirmed": False,
        "status": None,
        "error": None,
    }

    logger.info("开始 Travel Agent 流程: session={}", session_id)

    t0 = time.perf_counter()
    try:
        final_state = await graph.ainvoke(initial_state, config)
        if pending_confirmation(final_state):
            logger.info("流程中断等待人工确认: session={}", session_id)
        else:
            logger.info(
                "Travel Agent 流程完成: session={}, 模式={}, 总耗时 {:.2f}s",
                session_id,
                final_state.get("research_meta", {}).get("mode", "-"),
                time.perf_counter() - t0,
            )
        return final_state
    except Exception as e:
        logger.exception("Travel Agent 流程异常: {}", e)
        return {
            **initial_state,
            "error": str(e),
            "itinerary": f"## 抱歉，旅行规划过程中出现错误\n\n错误信息：{e}\n\n请稍后重试。",
        }


async def resume_travel_agent(
    session_id: str,
    decision: dict[str, Any],
    llm: Any = None,
) -> dict[str, Any]:
    """
    恢复被人工确认中断的图执行。

    Args:
        session_id: 与中断时相同的会话 ID（checkpointer 按 thread_id 定位断点）
        decision: {"approved": bool, "note": str}
        llm: ModelRouter 实例（恢复执行通常不再需要，但保持接口一致）

    Returns:
        最终状态字典
    """
    graph = get_travel_graph()
    config = {"configurable": {"thread_id": session_id, "llm": llm}}

    # 防御：无断点时 Command(resume=...) 会以空状态重跑全图（表现为"凭空规划行程"），
    # 必须显式拒绝
    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise ValueError(f"会话 {session_id} 没有可恢复的断点")

    logger.info("恢复 Travel Agent 流程: session={}, decision={}", session_id, decision)
    final_state = await graph.ainvoke(Command(resume=decision), config)
    logger.info("Travel Agent 流程恢复完成: session={}", session_id)
    return final_state


def pending_confirmation(state: dict[str, Any]) -> bool:
    """判断图是否正中断等待人工确认（ainvoke 返回值含 __interrupt__ 即在等待）。"""
    return bool(state.get("__interrupt__"))

