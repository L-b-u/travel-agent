# -*- coding: utf-8 -*-
"""
LangGraph 状态图：编排 7 个 Agent 节点的流水线。

流程：
    collect_preferences → search_pois → plan_routes
                                    → check_weather ─┐
                                    → estimate_budget ┘
                                    → synthesize → safety_review → END

天气和预算节点并行执行（Fan-out / Fan-in）。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from app.core.travel.agents import (
    check_weather_node,
    collect_preferences_node,
    estimate_budget_node,
    plan_routes_node,
    safety_review_node,
    search_pois_node,
    synthesize_node,
)
from app.core.travel.state import TravelState


def build_travel_graph() -> StateGraph:
    """
    构建 Travel Agent 的 LangGraph 状态图。

    返回编译后的 graph 实例。使用 MemorySaver 按 thread_id 隔离会话状态
    （安全审查节点标记 requires_confirmation，但当前未接 interrupt_before。
    """
    workflow = StateGraph(TravelState)

    # ---- 添加节点 ----
    workflow.add_node("collect_preferences", collect_preferences_node)
    workflow.add_node("search_pois", search_pois_node)
    workflow.add_node("plan_routes", plan_routes_node)
    workflow.add_node("check_weather", check_weather_node)
    workflow.add_node("estimate_budget", estimate_budget_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("safety_review", safety_review_node)

    # ---- 设置边 ----
    workflow.set_entry_point("collect_preferences")

    # 顺序链路
    workflow.add_edge("collect_preferences", "search_pois")
    workflow.add_edge("search_pois", "plan_routes")

    # Fan-out: 并行分发到天气和预算
    workflow.add_edge("plan_routes", "check_weather")
    workflow.add_edge("plan_routes", "estimate_budget")

    # Fan-in: 汇聚到合成节点
    workflow.add_edge("check_weather", "synthesize")
    workflow.add_edge("estimate_budget", "synthesize")

    # 合成 → 安全审查 → 结束
    workflow.add_edge("synthesize", "safety_review")
    workflow.add_edge("safety_review", END)

    # MemorySaver 按 thread_id 隔离会话状态（interrupt/resume HITL 为预留扩展，未接入）
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    logger.info("Travel Agent LangGraph 编译完成")
    return graph


# 全局单例
_travel_graph: Optional[StateGraph] = None


def get_travel_graph() -> StateGraph:
    """获取 Travel Agent 图单例。"""
    global _travel_graph
    if _travel_graph is None:
        _travel_graph = build_travel_graph()
    return _travel_graph


async def run_travel_agent(
    user_input: str,
    session_id: str = "default",
    llm: Any = None,
) -> Dict[str, Any]:
    """
    运行 Travel Agent 全流程。

    Args:
        user_input: 用户自然语言输入
        session_id: 会话 ID
        llm: LLM 实例（需实现 acomplete 方法），可选

    Returns:
        最终状态字典，包含 itinerary / safety_result / preferences 等
    """
    graph = get_travel_graph()

    config = {"configurable": {"thread_id": session_id, "llm": llm}}

    initial_state: TravelState = {
        "user_input": user_input,
        "session_id": session_id,
        "preferences": {},
        "pois": [],
        "routes": [],
        "weather": [],
        "budget": {},
        "itinerary": "",
        "safety_result": {},
        "requires_confirmation": False,
        "confirmation_items": [],
        "error": None,
    }

    logger.info("开始 Travel Agent 流程: session={}", session_id)

    t0 = time.perf_counter()
    try:
        final_state = await graph.ainvoke(initial_state, config)
        logger.info("Travel Agent 流程完成: session={}, 总耗时 {:.2f}s", session_id, time.perf_counter() - t0)
        return final_state
    except Exception as e:
        logger.exception("Travel Agent 流程异常: {}", e)
        return {
            **initial_state,
            "error": str(e),
            "itinerary": f"## 抱歉，旅行规划过程中出现错误\n\n错误信息：{e}\n\n请稍后重试。",
        }