﻿﻿﻿# -*- coding: utf-8 -*-
"""
天气查询 Agent：查询目的地天气预报。
"""

from __future__ import annotations

import time
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.travel.state import TravelState
from app.core.travel.tools._poi_data import find_city_coords
from app.core.travel.tools.get_weather import get_weather_forecast


async def check_weather_node(state: TravelState, config: RunnableConfig) -> Dict[str, Any]:
    """
    节点 4a：天气查询（与预算并行）。

    查询目的地未来几天的天气，用于行程调整建议。
    """
    preferences = state.get("preferences", {})
    destination = preferences.get("destination", "杭州")
    days = preferences.get("days", 2)
    start_date = preferences.get("start_date")

    coords = find_city_coords(destination)
    if not coords:
        # 城市不在内置坐标表，从已搜到的 POI 取近似坐标
        pois = state.get("pois", [])
        if pois:
            first = pois[0]
            coords = {"lat": first.get("lat", 30.2741), "lon": first.get("lon", 120.1551)}
            logger.info("城市 [{}] 未收录坐标，使用首个 POI [{}] 的坐标", destination, first.get("name", ""))
        else:
            logger.warning("未找到城市 [{}] 坐标且无 POI 数据，使用默认坐标", destination)
            coords = {"lat": 30.2741, "lon": 120.1551}  # 默认杭州

    t0 = time.perf_counter()
    try:
        weather = await get_weather_forecast.ainvoke({
            "lat": coords["lat"],
            "lon": coords["lon"],
            "start_date": start_date,
            "days": days,
        })
    except Exception as e:
        logger.warning("天气查询失败: {}", e)
        return {"weather": []}

    ometeo_hit = any(w.get("source") == "open-meteo" for w in weather)
    logger.info(
        "天气查询: {} 条 (来源={}, 耗时 {:.0f}ms)",
        len(weather), "open-meteo" if ometeo_hit else "fallback", (time.perf_counter() - t0) * 1000,
    )
    return {"weather": weather}