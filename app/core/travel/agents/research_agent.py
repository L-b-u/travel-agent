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
from typing import Any

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
3. search_travel_tips(city, query): 检索本地攻略知识库（交通/预约/避坑/高反等实用信息）。

工作要求：
- 目的地：{destination}；用户兴趣：{interests}；行程天数：{days} 天
- 先用 search_travel_tips 查一次目的地的关键注意事项（预约、避坑）
- 按兴趣逐一调用 search_pois（每类兴趣一次即可），总 POI 数量控制在 {poi_target} 个左右
- 若景点分布较散，可用 estimate_route 验证主要景点间的通勤时间
- 收集完成后，用两三句话总结该目的地的游览布局特点（如集中片区、跨度），不要罗列全部数据"""


def _build_react_tools(
    amap_key: str,
    capture: dict[str, Any],
    trace: list[dict[str, Any]],
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
        interests: list[str],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """搜索目的地的兴趣点（景点/美食/博物馆等）。返回名称、评分、坐标、地址、类别。
        相同参数的重复调用会直接返回缓存结果。

        Args:
            destination: 目的地城市名，如 "杭州"
            interests: 兴趣列表，如 ["博物馆", "美食"]
            limit: 每类兴趣最多返回数量
        """
        # 幂等缓存：LLM 偶尔会重复发完全相同的搜索，命中则不打 API
        cache_key = (destination.strip(), tuple(sorted(i.strip() for i in interests)))
        poi_cache: dict = capture.setdefault("_poi_search_cache", {})
        if cache_key in poi_cache:
            cached = poi_cache[cache_key]
            trace.append({
                "tool": "search_pois",
                "args": {"destination": destination, "interests": interests},
                "returned": len(cached),
                "cached": True,
                "duration_ms": 0,
            })
            return cached

        t0 = time.perf_counter()
        result = await search_places.ainvoke({
            "destination": destination,
            "interests": interests,
            "radius": 8000,
            "limit": min(max(limit, 3), 15),
            "api_key": amap_key,
        })
        poi_cache[cache_key] = result
        # side-channel：按名称去重累积
        pool: dict[str, dict[str, Any]] = capture.setdefault("pois", {})
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
    ) -> dict[str, Any]:
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

    @tool
    async def search_travel_tips(city: str, query: str) -> list[dict[str, Any]]:
        """检索本地旅行攻略知识库，返回实用提示（交通、门票预约、避坑、健康提醒等）。

        Args:
            city: 城市名，如 "丽江"
            query: 想了解的主题，如 "雪山预约"、"海鲜避坑"
        """
        t0 = time.perf_counter()
        from app.core.travel.rag import get_retriever

        hits = get_retriever().retrieve(query, city=city, k=3)
        # side-channel：引用来源单独收集（合成行程时标注信息来源）
        capture.setdefault("tips", []).extend(hits)
        trace.append({
            "tool": "search_travel_tips",
            "args": {"city": city, "query": query},
            "returned": len(hits),
            "duration_ms": round((time.perf_counter() - t0) * 1000),
        })
        return [{"citation": h["citation"], "text": h["text"]} for h in hits]

    return [search_pois, estimate_route_by_coords, search_travel_tips]


async def research_node(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
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

    return await _research_deterministic(destination, interests, days, query_hint=state.get("user_input", ""))


async def _research_with_agent(
    state: TravelState,
    config: RunnableConfig,
    llm: Any,
) -> dict[str, Any]:
    """ReAct 主路径：LLM 决策工具调用，结果 side-channel 捕获。"""
    from langgraph.prebuilt import create_react_agent

    preferences = state.get("preferences", {})
    destination = preferences.get("destination", "杭州")
    interests = preferences.get("interests") or ["景点", "美食"]
    days = preferences.get("days", 2)
    amap_key = get_settings().amap_api_key

    capture: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
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
    tips = _dedup_tips(capture.get("tips", []))
    summary = ""
    if result.get("messages"):
        last = result["messages"][-1]
        summary = getattr(last, "content", "") or ""

    logger.info(
        "ReAct 研究完成: {} 个 POI, {} 条路线, {} 条攻略, {} 次工具调用, 耗时 {:.0f}ms",
        len(pois), len(routes), len(tips), len(trace), duration_ms,
    )
    return {
        "pois": pois,
        "routes": routes,
        "tips": tips,
        "research_summary": summary,
        "research_trace": trace,
        "research_meta": {
            "mode": "react",
            "tool_calls": len(trace),
            "duration_ms": duration_ms,
        },
    }


def _dedup_tips(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按引用来源去重（Agent 可能多次查询命中同一章节）。"""
    seen: set = set()
    result: list[dict[str, Any]] = []
    for h in hits:
        key = h.get("citation", "")
        if key and key not in seen:
            seen.add(key)
            result.append({"citation": h["citation"], "text": h["text"]})
    return result


def _fetch_tips_deterministic(
    destination: str,
    interests: list[str],
    query_hint: str = "",
) -> list[dict[str, Any]]:
    """确定性兜底路径的攻略检索：用户原话 + 目的地直查知识库。"""
    try:
        from app.core.travel.rag import get_retriever

        retriever = get_retriever()
        query = f"{query_hint or destination} {' '.join(interests)}"
        hits = retriever.retrieve(query, city=destination, k=3)
        return [{"citation": h["citation"], "text": h["text"]} for h in hits]
    except Exception as e:
        logger.debug("攻略检索不可用: {}", e)
        return []


async def _research_deterministic(
    destination: str,
    interests: list[str],
    days: int,
    query_hint: str = "",
) -> dict[str, Any]:
    """
    确定性兜底路径：原 POI 搜索 + 贪心路线规划逻辑收编于此。

    不依赖 LLM：搜索按兴趣直查、按天贪心分组后批量估路线。
    query_hint（用户原话）作为攻略检索查询，比兴趣词更能命中具体关切（如高反、预约）。
    """
    amap_key = get_settings().amap_api_key
    trace: list[dict[str, Any]] = []

    # ---- 1. 攻略检索（不依赖网络 API，离线可用）----
    tips = _fetch_tips_deterministic(destination, interests, query_hint)

    # ---- 2. POI 搜索 ----
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
            "tips": tips,
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
            "tips": tips,
            "research_summary": "",
            "research_trace": trace,
            "research_meta": {"mode": "deterministic"},
        }

    # ---- 2. 贪心按天分组（每天 2-4 个）----
    daily_pois: list[list[dict[str, Any]]] = []
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
    all_routes: list[dict[str, Any]] = []
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

    logger.info("确定性研究完成: {} 个 POI, {} 条路线, {} 条攻略", len(pois), len(all_routes), len(tips))
    return {
        "pois": pois,
        "routes": all_routes,
        "tips": tips,
        "research_summary": "",
        "research_trace": trace,
        "research_meta": {"mode": "deterministic"},
    }
