# -*- coding: utf-8 -*-
"""
POI 搜索工具：高德地图 API + 内置降级数据库。

使用 LangChain @tool 装饰器标注，便于 LLM 工具调用与 LangGraph 节点复用。
高德地图 API 文档：https://lbs.amap.com/api/webservice/guide/api/search
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
from langchain_core.tools import tool
from loguru import logger

from app.core.travel.tools._poi_data import FALLBACK_POIS


# 高德搜索关键词映射（将用户兴趣映射为高德搜索关键词）
AMAP_KEYWORD_MAP: Dict[str, str] = {
    "博物馆": "博物馆",
    "museums": "博物馆",
    "museum": "博物馆",
    "美食": "美食",
    "foods": "餐厅",
    "餐厅": "餐厅",
    "restaurant": "餐厅",
    "景点": "旅游景点",
    "interesting_places": "旅游景点",
    "attraction": "景点",
    "自然": "公园",
    "natural": "公园",
    "公园": "公园",
    "park": "公园",
    "购物": "购物",
    "shops": "商场",
    "shopping": "商场",
    "历史": "古迹",
    "historic": "古迹",
    "宗教": "寺庙",
    "religion": "寺庙",
    "temple": "寺庙",
    "咖啡": "咖啡",
    "cafe": "咖啡",
    "建筑": "地标建筑",
    "architecture": "地标建筑",
}


@tool
async def search_places(
    destination: str,
    interests: List[str],
    radius: int = 5000,
    limit: int = 15,
    api_key: str = "",
) -> List[Dict[str, Any]]:
    """
    搜索目的地景点和兴趣点（POI）。优先调用高德地图 API，失败时降级到内置数据库。

    Args:
        destination: 目的地城市名，如 "杭州"
        interests: 兴趣列表，如 ["博物馆", "美食", "自然"]
        radius: 搜索半径（米）
        limit: 返回结果数上限
        api_key: 高德地图 API Key（可选，未提供时走内置数据）

    Returns:
        POI 列表，每项含 name/lat/lon/category/rating/address
    """
    # 未配置 API Key 时，直接使用内置数据（不发 HTTP 请求）
    if not api_key:
        logger.info("未配置 AMAP_API_KEY，使用内置景点数据: {}", destination)
        return _filter_fallback(destination, interests, limit)

    try:
        all_pois = await _search_via_amap(destination, interests, limit, api_key)
        if all_pois:
            all_pois.sort(key=lambda x: float(x.get("rating", 0) or 0), reverse=True)
            return all_pois[:limit]
    except Exception as e:
        logger.warning("高德地图 API 调用失败，降级到内置数据: {}", e)

    return _filter_fallback(destination, interests, limit)


async def _search_via_amap(
    destination: str,
    interests: List[str],
    limit: int,
    api_key: str,
) -> List[Dict[str, Any]]:
    """通过高德地图 POI 搜索 API 查询。

    高德 API 文档：https://lbs.amap.com/api/webservice/guide/api/search
    返回 location 格式为 "经度,纬度"（lon,lat）。
    """
    all_pois: List[Dict[str, Any]] = []
    seen: set = set()

    # 将兴趣映射为高德搜索关键词
    keywords_list = []
    for interest in interests:
        keyword = AMAP_KEYWORD_MAP.get(interest, interest)
        if keyword not in keywords_list:
            keywords_list.append(keyword)

    if not keywords_list:
        keywords_list = ["旅游景点"]

    per_keyword_limit = max(limit // len(keywords_list), 5)

    url = "https://restapi.amap.com/v3/place/text"
    headers = {"User-Agent": "TravelAgent/1.0 (educational-project)"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        for keyword in keywords_list:
            params = {
                "keywords": keyword,
                "city": destination,
                "citylimit": "true",
                "offset": per_keyword_limit,
                "page": 1,
                "key": api_key,
                "extensions": "all",
            }

            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            raw = resp.json()

            if raw.get("status") != "1":
                logger.warning(
                    "高德 API 返回错误: keyword={}, info={}",
                    keyword,
                    raw.get("info", "unknown"),
                )
                continue

            for poi in raw.get("pois", []):
                name = poi.get("name", "")
                if not name or name in seen:
                    continue
                seen.add(name)

                # 高德 location 格式: "经度,纬度"（lon,lat）
                location_str = poi.get("location", "")
                lat, lon = 0.0, 0.0
                if "," in location_str:
                    parts = location_str.split(",")
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                    except (ValueError, IndexError):
                        pass

                # 评分（高德 biz_ext.rating 可能不存在）
                biz_ext = poi.get("biz_ext", {})
                rating_str = biz_ext.get("rating", "") if biz_ext else ""
                try:
                    rating = float(rating_str) if rating_str else 0.0
                except ValueError:
                    rating = 0.0

                all_pois.append({
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "rating": rating,
                    "address": poi.get("address", ""),
                    "category": keyword,
                    "source": "amap",
                })

    logger.info(
        "高德 API 搜索完成: 城市={}, 关键词={}, 结果数={}",
        destination,
        keywords_list,
        len(all_pois),
    )
    return all_pois


def _filter_fallback(destination: str, interests: List[str], limit: int) -> List[Dict[str, Any]]:
    """从内置数据库筛选 POI。"""
    city_pois = FALLBACK_POIS.get(destination)

    if not city_pois:
        # 模糊匹配
        for city, pois in FALLBACK_POIS.items():
            if city in destination or destination in city:
                city_pois = pois
                break

    if not city_pois:
        # 未收录该城市，返回空列表而非任意城市数据
        logger.warning("城市 [{}] 暂未收录在内置 POI 数据库中", destination)
        return []

    # 按兴趣过滤
    if interests:
        matched = [p for p in city_pois if any(
            i.lower() in p.get("category", "").lower()
            for i in interests
        )]
        if matched:
            city_pois = matched

    # 按评分排序
    city_pois.sort(key=lambda x: float(x.get("rating", 0) or 0), reverse=True)
    return city_pois[:limit]
