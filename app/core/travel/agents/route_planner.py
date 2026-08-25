﻿﻿﻿# -*- coding: utf-8 -*-
"""
路线规划 Agent：根据 POI 列表规划游览路线。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.config import get_settings
from app.core.travel.state import TravelState
from app.core.travel.tools.estimate_route import estimate_routes_batch


async def plan_routes_node(state: TravelState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点 3：路线规划。

    按天分配 POI，计算每天各景点间的路线。
    采用贪心策略：每天不超过 4 个景点，按距离排序。
    """
    preferences = state.get("preferences", {})
    days = preferences.get("days", 2)
    pois = state.get("pois", [])

    if not pois:
        logger.warning("无 POI 数据，跳过路线规划")
        return {"routes": []}

    # 按天分配 POI（每天 3-4 个）
    daily_pois: List[List[Dict[str, Any]]] = []
    pois_per_day = min(4, max(2, len(pois) // days))
    remaining = list(pois)

    for day in range(days):
        day_pois = remaining[:pois_per_day]
        remaining = remaining[pois_per_day:]
        if day_pois:
            daily_pois.append(day_pois)
        if not remaining:
            break

    # 如果还有剩余，分配到各天
    day_idx = 0
    while remaining and daily_pois:
        daily_pois[day_idx % len(daily_pois)].append(remaining.pop(0))
        day_idx += 1

    # 计算每天路线
    amap_key = get_settings().amap_api_key
    all_routes: List[Dict[str, Any]] = []
    for day, day_pois in enumerate(daily_pois):
        if len(day_pois) < 2:
            continue
        t0 = time.perf_counter()
        try:
            routes = await estimate_routes_batch.ainvoke({
                "waypoints": day_pois,
                "mode": "driving",
                "api_key": amap_key,
            })
            for route in routes:
                route["day"] = day + 1
            all_routes.extend(routes)
            amap_hit = any(r.get("estimate_method") == "amap" for r in routes)
            logger.info(
                "第{}天路线: {} 条 (来源={}, 耗时 {:.0f}ms)",
                day + 1, len(routes), "amap" if amap_hit else "haversine",
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.warning("第 {} 天路线计算失败: {}", day + 1, e)

    logger.info("路线规划完成: {} 条路线", len(all_routes))
    return {"routes": all_routes}