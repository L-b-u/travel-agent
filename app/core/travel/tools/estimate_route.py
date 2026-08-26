"""
路线估算工具：高德地图路线规划 API + 直线距离降级。

使用 LangChain @tool 装饰器标注。
高德路线规划 API 文档：https://lbs.amap.com/api/webservice/guide/api/direction
"""

from __future__ import annotations

import math
from typing import Any

import httpx
from langchain_core.tools import tool
from loguru import logger


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的大圆距离（km）。"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@tool
async def estimate_route(
    origin: dict[str, Any],
    destination: dict[str, Any],
    mode: str = "driving",
    api_key: str = "",
) -> dict[str, Any]:
    """
    估算两点间路线（距离、耗时）。优先调用高德路线规划 API，失败时降级为直线距离。

    Args:
        origin: 起点，{"lat": float, "lon": float, "name": str}
        destination: 终点，{"lat": float, "lon": float, "name": str}
        mode: 交通方式 driving / walking
        api_key: 高德地图 API Key（可选，未提供时走直线距离估算）

    Returns:
        {"origin": str, "destination": str, "distance_km": float, "duration_min": float, "mode": str}
    """
    o_lat = origin.get("lat", 0)
    o_lon = origin.get("lon", 0)
    d_lat = destination.get("lat", 0)
    d_lon = destination.get("lon", 0)

    # 尝试高德路线规划 API
    if api_key and o_lat and o_lon and d_lat and d_lon:
        try:
            result = await _amap_route(o_lat, o_lon, d_lat, d_lon, mode, api_key)
            result["origin"] = origin.get("name", "")
            result["destination"] = destination.get("name", "")
            return result
        except Exception as e:
            logger.warning("高德路线规划失败，降级为直线距离: {}", e)

    # 降级：直线距离估算
    distance = _haversine(o_lat, o_lon, d_lat, d_lon)
    if distance < 0.1:
        distance = 0.5  # 最小 500m

    # 按交通方式估算速度
    speeds = {"driving": 40, "walking": 5, "cycling": 15}
    speed = speeds.get(mode, 40)
    duration = (distance / speed) * 60  # 分钟

    return {
        "origin": origin.get("name", ""),
        "destination": destination.get("name", ""),
        "distance_km": round(distance, 1),
        "duration_min": round(duration, 0),
        "mode": mode,
        "estimate_method": "haversine_fallback",
    }


async def _amap_route(
    o_lat: float, o_lon: float,
    d_lat: float, d_lon: float,
    mode: str,
    api_key: str,
) -> dict[str, Any]:
    """调用高德路线规划 API。

    高德坐标格式：经度,纬度（lon,lat）
    驾车：/direction/driving
    步行：/direction/walking
    """
    # 高德坐标格式：lon,lat
    origin_str = f"{o_lon},{o_lat}"
    dest_str = f"{d_lon},{d_lat}"

    # 步行用 walking 接口，其他用 driving
    if mode == "walking":
        url = "https://restapi.amap.com/v3/direction/walking"
    else:
        url = "https://restapi.amap.com/v3/direction/driving"

    params = {
        "origin": origin_str,
        "destination": dest_str,
        "key": api_key,
        "extensions": "base",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") != "1":
        raise RuntimeError(f"高德 API 返回错误: {data.get('info', 'unknown')}")

    route = data.get("route", {})
    paths = route.get("paths", [])
    if not paths:
        raise RuntimeError("高德 API 返回无路线")

    path = paths[0]
    distance_m = float(path.get("distance", 0))
    duration_s = float(path.get("duration", 0))

    return {
        "distance_km": round(distance_m / 1000, 1),
        "duration_min": round(duration_s / 60, 0),
        "mode": mode,
        "estimate_method": "amap",
    }


@tool
async def estimate_routes_batch(
    waypoints: list[dict[str, Any]],
    mode: str = "driving",
    api_key: str = "",
) -> list[dict[str, Any]]:
    """
    批量估算连续路线（A→B→C→D）。

    Args:
        waypoints: 地点列表，每项含 lat/lon/name
        mode: 交通方式
        api_key: 高德地图 API Key

    Returns:
        路线段列表
    """
    routes = []
    for i in range(len(waypoints) - 1):
        route = await estimate_route.ainvoke({
            "origin": waypoints[i],
            "destination": waypoints[i + 1],
            "mode": mode,
            "api_key": api_key,
        })
        routes.append(route)
    return routes
