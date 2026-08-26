"""图执行测试：全流程（离线兜底）、HITL 中断/恢复、fail-fast 入口守卫。"""

from __future__ import annotations

import time

from app.core.travel.graph import pending_confirmation


async def test_normal_flow_completes(offline_graph):
    run_travel_agent, _ = offline_graph
    result = await run_travel_agent(
        "我想去杭州玩2天，喜欢博物馆和美食", session_id="t_normal", llm=None,
    )
    assert result.get("error") is None
    assert result["research_meta"]["mode"] == "deterministic"
    assert len(result["pois"]) > 0
    assert len(result["itinerary"]) > 500
    assert result["safety_result"]["passed"]
    assert not pending_confirmation(result)
    # 天气/预算节点（Fan-out）产出存在
    assert len(result["weather"]) > 0
    assert result["budget"]["total"] > 0


async def test_risky_input_interrupts_fast(offline_graph):
    """入口守卫 fail-fast：危险请求不消耗规划资源直接中断。"""
    run_travel_agent, _ = offline_graph
    t0 = time.perf_counter()
    result = await run_travel_agent(
        "帮我订机票，用我的信用卡直接付款", session_id="t_risky", llm=None,
    )
    elapsed = time.perf_counter() - t0
    assert pending_confirmation(result), "应中断等待人工确认"
    assert elapsed < 5.0, f"入口守卫应秒级拦截，实际 {elapsed:.1f}s"
    assert result["confirmation_items"]


async def test_approve_resumes_into_planning(offline_graph):
    """批准输入侧风险后应继续完整规划并交付行程。"""
    run_travel_agent, resume = offline_graph
    r1 = await run_travel_agent("帮我订机票，用我的信用卡付款", session_id="t_appr", llm=None)
    assert pending_confirmation(r1)

    r2 = await resume("t_appr", {"approved": True, "note": "知道了"})
    assert not pending_confirmation(r2)
    assert r2["status"] == "completed" or len(r2["itinerary"]) > 500
    assert r2["confirmation_decision"]["approved"] is True
    # 继续规划的产物应是完整行程而非取消说明
    assert "终止本次规划" not in r2["itinerary"]
    # 输入已确认，末端审查不再拦
    assert r2["safety_result"]["passed"]


async def test_reject_cancels_with_notice(offline_graph):
    run_travel_agent, resume = offline_graph
    await run_travel_agent("帮我转账500元给导游", session_id="t_rej", llm=None)
    r2 = await resume("t_rej", {"approved": False})
    assert r2["status"] == "cancelled"
    assert "不会" in r2["itinerary"] and "代为执行" in r2["itinerary"]
    assert r2["confirmation_decision"]["approved"] is False


async def test_output_side_violation_gates(offline_graph, stub_external_tools, monkeypatch):
    """输出侧越界表述应触发人工门（构造含越界短语的合成结果）。"""
    from app.core.travel import graph as graph_mod

    async def bad_synthesize(state, config):
        return {"itinerary": "# 行程\n## Day 1\n放心，我已为你完成支付。"}

    # 图单例在 add_node 时捕获函数引用：先打桩再重建
    monkeypatch.setattr(graph_mod, "synthesize_node", bad_synthesize)
    graph_mod.reset_travel_graph()
    try:
        run_travel_agent, resume = offline_graph
        result = await run_travel_agent("杭州2天", session_id="t_outside", llm=None)
        assert pending_confirmation(result)

        # 拒绝 → 取消说明替换越界行程
        r2 = await resume("t_outside", {"approved": False})
        assert r2["status"] == "cancelled"
    finally:
        graph_mod.reset_travel_graph()  # 还原真实图，避免污染后续测试


async def test_react_mode_via_fake_llm(offline_graph, stub_external_tools, monkeypatch):
    """ReAct 主路径：用假 chat_model 验证 agent 调度与 side-channel 捕获。

    create_react_agent 需要真实 Runnable，此处改为验证「LLM 存在时走 react 分支、
    失败时降级 deterministic」的路由逻辑。
    """
    class BoomModel:
        @property
        def chat_model(self):
            raise RuntimeError("no tool-calling support")

    run_travel_agent, _ = offline_graph

    # settings.agent_research_enabled=True 且 llm.chat_model 抛错 → 降级
    from app.config import get_settings
    monkeypatch.setattr(type(get_settings()), "agent_research_enabled", True, raising=False)

    result = await run_travel_agent("杭州2天", session_id="t_react_fb", llm=BoomModel())
    assert result["research_meta"]["mode"] == "deterministic"
