﻿﻿﻿# -*- coding: utf-8 -*-
"""
POI 搜索 Agent：根据偏好搜索景点、餐厅、博物馆等。
"""

from __future__ import annotations

import time
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.config import get_settings
from app.core.travel.state import TravelState
from app.core.travel.tools.search_places import search_places


def _get_amap_api_key() -> str:
    """从配置读取高德地图 API Key（未配置时返回空字符串，触发降级）。"""
    return get_settings().amap_api_key


async def search_pois_node(state: TravelState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点 2：POI 搜索。

    根据目的地和兴趣搜索景点，每类兴趣返回 5-10 个结果。
    """
    preferences = state.get("preferences", {})
    destination = preferences.get("destination", "杭州")
    interests = preferences.get("interests", ["景点", "美食"])

    if not isinstance(interests, list):
        interests = ["景点", "美食"]

    logger.info("搜索 POI: 目的地={}, 兴趣={}", destination, interests)

    t0 = time.perf_counter()
    try:
        pois = await search_places.ainvoke({
            "destination": destination,
            "interests": interests,
            "radius": 8000,
            "limit": 15,
            "api_key": _get_amap_api_key(),
        })
    except Exception as e:
        logger.exception("POI 搜索失败: {}", e)
        return {"pois": [], "error": f"POI 搜索失败: {e}"}

    amap_hit = any(p.get("source") == "amap" for p in pois)
    logger.info(
        "POI 搜索: {} 个 (来源={}, 耗时 {:.0f}ms)",
        len(pois), "amap" if amap_hit else "fallback", (time.perf_counter() - t0) * 1000,
    )
    return {"pois": pois}