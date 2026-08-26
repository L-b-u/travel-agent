# -*- coding: utf-8 -*-
"""
研究 Agent：旅行信息的探索式收集（真正的 LLM Tool Calling 节点）。

主路径：create_react_agent 构建的 ReAct 循环。LLM 通过 bind_tools 拿到
search_pois / estimate_route 两个工具，自主决定调用顺序与次数
（如先搜博物馆再搜美食、对相邻景点估算交通耗时），工具结果经
side-channel 捕获器直接写入结构化通道，不依赖 LLM 复述，避免信息失真。

兜底路径：LLM 不可用 / Agent 异常 / 配置关闭时，退回确定性流水线
（按兴趣搜索 + 贪心按天分组 + 批量路线估算），保证功能可用。

为什么研究阶段用 Agent、天气/预算阶段用确定性节点：
    探索性任务（搜什么、按什么顺序看）收益来自 LLM 的动态决策；
    而天气/预算是必须保证覆盖的关键数据，走确定性并行节点更可控可测。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from loguru import logger

from app.config import get_settings
from app.core.travel.state import TravelState
from app.core.travel.tools.estimate_route import estimate_route, estimate_routes_batch
from app.core.travel.tools.search_places import search_places

# 单次研究的调用上限（防止 Agent 循环失控）
MAX_TOOL_CALLS = 12


RESEARCH_SYSTEM_PROMPT = """你是旅行研究助理，负责为行程规划收集目的地信息。

可用工具：
1. search_pois(destination, interests, limit): 搜索景点/美食/博物馆等兴趣点，
   返回名称、评分、坐标、地址。坐标是后续估算路线的输入。
2. estimate_route(o_lat, o_lon, d_lat, d_lon, mode): 估算两点间距离与耗时。

工作要求：
- 目的地：{destination}；用户兴趣：{interests}；行程天数：{days} 天
- 按兴趣逐一搜索（每个兴趣一次调用即可），总 POI 数量控制在 {poi_target} 个左右
- 若搜索结果覆盖了多个相距较远的区域，可用 estimate_route 验证主要景点间的通勤时间
- 收集完成后，用两三句话总结该目的地的游览布局特点（如集中片区、跨度），不要罗列全部数据"""


def _build_react_tools(
    amap_key: str,
    capture: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> list:
    """
    构建给 ReAct Agent 用的 LLM 工具集。

    与底层 API 工具的差异：
    - 隐藏 api_key 等实现细节（注入闭包，不进 LLM 的参数空间）；
    - 结果通过 side-channel 写入 capture（pois 去重累积、routes 追加），
      图状态取数不经过 LLM 复述；
    - 每次调用记录 trace（工具/参数摘要/耗时/返回量），供可观测性与测试断言。
    """

    @tool
    async def search_pois(
        destination: str,
        interests: List[str],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """搜索目的地的兴趣点（景点/美食/博物馆等）。返回名称、评分、坐标、地址、类别。

        Args:
            destination: 目的地城市名，如 "杭州"
            interests: 兴趣列表，如 ["博物馆", "美食"]
            limit: 每类兴趣最多返回数量
        """
        t0 = time.perf_counter()
        result = await search_places.ainvoke({
            "destination": destination,
            "interests": interests,
            "radius": 8000,
            "limit": min(max(limit, 3), 15),
            "api_key": amap_key,
        })
        # side-channel：按名称去重累积
        pool: Dict[str, Dict[str, Any]] = capture.setdefault("pois", {})
        for poi in result:
            name = poi.get("name", "")
            if name and name not in pool:
                pool[name] = poi
        trace.append({
            "tool": "search_pois",
            "args": {"destination": destination, "interests": interests},
            "returned": len(result),
            "duration_ms": round((time.perf_counter() - t0) * 1000),
        })
        return result

    @tool
    async def estimate_route_by_coords(
        o_lat: float,
        o_lon: float,
        d_lat: float,
        d_lon: float,
        o_name: str = "",
        d_name: str = "",
        mode: str = "driving",
    ) -> Dict[str, Any]:
        """估算两个地点之间的距离与通行时间。

        Args:
            o_lat: 起点纬度
            o_lon: 起点经度
            d_lat: 终点纬度
            d_lon: 终点经度
            o_name: 起点名称（可选，便于阅读）
            d_name: 终点名称（可选）
            mode: 交通方式 driving / walking
        """
        t0 = time.perf_counter()
        result = await estimate_route.ainvoke({
            "origin": {"lat": o_lat, "lon": o_lon, "name": o_name},
            "destination": {"lat": d_lat, "lon": d_lon, "name": d_name},
            "mode": mode,
            "api_key": amap_key,
        })
        capture.setdefault("routes", []).append(result)
        trace.append({
            "tool": "estimate_route_by_coords",
            "args": {"o": o_name or f"({o_lat},{o_lon})", "d": d_name or f"({d_lat},{d_lon})"},
            "returned": 1,
            "duration_ms": round((time.perf_counter() - t0) * 1000),
        })
        return result

    return [search_pois, estimate_route_by_coords]


async def research_node(state: TravelState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点 2：研究（替代原 POI 搜索 + 路线规划两个节点）。

    主路径：ReAct Agent 工具调用；兜底路径：确定性流水线。
    """
    preferences = state.get("preferences", {})
    destination = preferences.get("destination", "杭州")
    interests = preferences.get("interests") or ["景点", "美食"]
    days = preferences.get("days", 2)

    llm = config.get("configurable", {}).get("llm")
    use_agent = llm is not None and get_settings().agent_research_enabled

    if use_agent:
        try:
            return await _research_with_agent(state, config, llm)
        except Exception as e:
            logger.warning("ReAct 研究失败，降级到确定性流水线: {}", e)

    return await _research_deterministic(destination, interests, days)


async def _research_with_agent(
    state: TravelState,
    config: RunnableConfig,
    llm: Any,
) -> Dict[str, Any]:
    """ReAct 主路径：LLM 决策工具调用，结果 side-channel 捕获。"""
    from langgraph.prebuilt import create_react_agent

    preferences = state.get("preferences", {})
    destination = preferences.get("destination", "杭州")
    interests = preferences.get("interests") or ["景点", "美食"]
    days = preferences.get("days", 2)
    amap_key = get_settings().amap_api_key

    capture: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    tools = _build_react_tools(amap_key, capture, trace)

    agent = create_react_agent(
        llm.chat_model,
        tools,
        prompt=RESEARCH_SYSTEM_PROMPT.format(
            destination=destination,
            interests="、".join(interests),
            days=days,
            poi_target=min(days * 5, 15),
        ),
    )

    t0 = time.perf_counter()
    # recursion_limit 兜底：系统消息+工具轮次不会超过 2*MAX_TOOL_CALLS+4
    agent_config = {
        **{k: v for k, v in (config or {}).items() if k != "configurable"},
        "configurable": {**(config or {}).get("configurable", {})},
        "recursion_limit": MAX_TOOL_CALLS * 2 + 6,
    }
    result = await agent.ainvoke(
        {"messages": [("user", f"请开始收集「{destination}」的旅行信息。")]},
        config=agent_config,
    )
    duration_ms = round((time.perf_counter() - t0) * 1000)

    pois = list(capture.get("pois", {}).values())
    routes = capture.get("routes", [])
    summary = ""
    if result.get("messages"):
        last = result["messages"][-1]
        summary = getattr(last, "content", "") or ""

    logger.info(
        "ReAct 研究完成: {} 个 POI, {} 条路线, {} 次工具调用, 耗时 {:.0f}ms",
        len(pois), len(routes), len(trace), duration_ms,
    )
    return {
        "pois": pois,
        "routes": routes,
        "research_summary": summary,
        "research_trace": trace,
        "research_meta": {
            "mode": "react",
            "tool_calls": len(trace),
            "duration_ms": duration_ms,
        },
    }


async def _research_deterministic(
    destination: str,
    interests: List[str],
    days: int,
) -> Dict[str, Any]:
    """
    确定性兜底路径：原 POI 搜索 + 贪心路线规划逻辑收编于此。

    不依赖 LLM：搜索按兴趣直查、按天贪心分组后批量估路线。
    """
    amap_key = get_settings().amap_api_key
    trace: List[Dict[str, Any]] = []

    # ---- 1. POI 搜索 ----
    t0 = time.perf_counter()
    try:
        pois = await search_places.ainvoke({
            "destination": destination,
            "interests": interests,
            "radius": 8000,
            "limit": 15,
            "api_key": amap_key,
        })
    except Exception as e:
        logger.exception("POI 搜索失败: {}", e)
        return {
            "pois": [],
            "routes": [],
            "research_summary": "",
            "research_trace": [],
            "research_meta": {"mode": "deterministic", "error": str(e)},
        }
    trace.append({
        "tool": "search_pois(direct)",
        "args": {"destination": destination, "interests": interests},
        "returned": len(pois),
        "duration_ms": round((time.perf_counter() - t0) * 1000),
    })

    if not pois:
        logger.warning("无 POI 数据，跳过路线规划")
        return {
            "pois": [],
            "routes": [],
            "research_summary": "",
            "research_trace": trace,
            "research_meta": {"mode": "deterministic"},
        }

    # ---- 2. 贪心按天分组（每天 2-4 个）----
    daily_pois: List[List[Dict[str, Any]]] = []
    pois_per_day = min(4, max(2, len(pois) // days))
    remaining = list(pois)
    for _day in range(days):
        day_pois = remaining[:pois_per_day]
        remaining = remaining[pois_per_day:]
        if day_pois:
            daily_pois.append(day_pois)
        if not remaining:
            break
    day_idx = 0
    while remaining and daily_pois:
        daily_pois[day_idx % len(daily_pois)].append(remaining.pop(0))
        day_idx += 1

    # ---- 3. 批量估算每日路线 ----
    all_routes: List[Dict[str, Any]] = []
    for day, day_group in enumerate(daily_pois):
        if len(day_group) < 2:
            continue
        t0 = time.perf_counter()
        try:
            routes = await estimate_routes_batch.ainvoke({
                "waypoints": day_group,
                "mode": "driving",
                "api_key": amap_key,
            })
            for route in routes:
                route["day"] = day + 1
            all_routes.extend(routes)
            trace.append({
                "tool": "estimate_routes_batch",
                "args": {"day": day + 1, "n_waypoints": len(day_group)},
                "returned": len(routes),
                "duration_ms": round((time.perf_counter() - t0) * 1000),
            })
        except Exception as e:
            logger.warning("第 {} 天路线计算失败: {}", day + 1, e)

    logger.info("确定性研究完成: {} 个 POI, {} 条路线", len(pois), len(all_routes))
    return {
        "pois": pois,
        "routes": all_routes,
        "research_summary": "",
        "research_trace": trace,
        "research_meta": {"mode": "deterministic"},
    }
