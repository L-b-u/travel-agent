"""
天气查询工具：Open-Meteo API。

使用 LangChain @tool 装饰器标注。
Open-Meteo 免费天气 API：https://open-meteo.com/en/docs
无需 API Key，每日 10000 次免费调用。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import httpx
from langchain_core.tools import tool
from loguru import logger

# 天气代码映射
WMO_CODES: dict[int, str] = {
    0: "晴",
    1: "大部晴",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "沉积雾凇",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


@tool
async def get_weather_forecast(
    lat: float,
    lon: float,
    start_date: str | None = None,
    days: int = 3,
) -> list[dict[str, Any]]:
    """
    获取目的地天气预报。

    Args:
        lat: 纬度
        lon: 经度
        start_date: 开始日期 YYYY-MM-DD，默认今天
        days: 预报天数

    Returns:
        [{date, condition, temp_max, temp_min, precipitation_prob, wind_speed}, ...]
    """
    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            # Open-Meteo 只支持未来 16 天预报，过去日期会导致 400 错误
            if start < date.today():
                logger.info("起始日期 {} 已过期，使用今天", start)
                start = date.today()
        except ValueError:
            start = date.today()
    else:
        start = date.today()

    end = start + timedelta(days=days - 1)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "wind_speed_10m_max",
        ],
        "timezone": "Asia/Shanghai",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("Open-Meteo 天气查询失败: {}", e)
        return _generate_fallback_weather(start, days)

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return _generate_fallback_weather(start, days)

    def _daily_series(key: str, default: float) -> list:
        """取每日序列，缺失时回退为等长默认值序列。"""
        series = daily.get(key) or []
        return series if len(series) == len(dates) else [default] * len(dates)

    temp_max = _daily_series("temperature_2m_max", 0)
    temp_min = _daily_series("temperature_2m_min", 0)
    precip = _daily_series("precipitation_probability_max", 0)
    wind = _daily_series("wind_speed_10m_max", 0)
    codes = _daily_series("weather_code", 0)

    forecasts = []
    for i, d in enumerate(dates):
        code = int(codes[i])
        forecasts.append({
            "date": d,
            "condition": WMO_CODES.get(code, "未知"),
            "temp_max": temp_max[i],
            "temp_min": temp_min[i],
            "precipitation_prob": precip[i],
            "wind_speed": wind[i],
            "source": "open-meteo",
        })

    return forecasts


def _generate_fallback_weather(start_date: date, days: int) -> list[dict[str, Any]]:
    """API 不可用时生成占位天气数据。"""
    forecasts = []
    for i in range(days):
        d = start_date + timedelta(days=i)
        forecasts.append({
            "date": d.isoformat(),
            "condition": "未知（API 不可用，请参考当地天气预报）",
            "temp_max": 25,
            "temp_min": 15,
            "precipitation_prob": 30,
            "wind_speed": 10,
            "source": "fallback",
        })
    return forecasts
