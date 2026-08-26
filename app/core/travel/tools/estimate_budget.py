"""
预算估算工具：基于规则 + 城市物价水平。

使用 LangChain @tool 装饰器标注。
数据来源：公开的旅游消费参考数据。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

# 城市消费水平指数（以杭州为基准 1.0）
CITY_PRICE_INDEX: dict[str, float] = {
    "杭州": 1.0, "北京": 1.3, "上海": 1.4, "广州": 1.1, "深圳": 1.3,
    "成都": 0.85, "西安": 0.8, "南京": 0.95, "重庆": 0.8, "武汉": 0.8,
    "苏州": 0.95, "厦门": 1.0, "长沙": 0.75, "青岛": 0.9, "大理": 0.7,
    "丽江": 0.75, "三亚": 1.3, "桂林": 0.7, "拉萨": 0.85, "哈尔滨": 0.75,
    "昆明": 0.7, "贵阳": 0.65, "天津": 0.9, "郑州": 0.7, "合肥": 0.7,
    "南昌": 0.65, "福州": 0.8, "济南": 0.75, "太原": 0.65, "石家庄": 0.65,
    "沈阳": 0.7, "长春": 0.65, "兰州": 0.6, "呼和浩特": 0.65, "海口": 0.85,
    "南宁": 0.65, "珠海": 1.0, "大连": 0.85, "宁波": 0.9, "无锡": 0.9,
    "温州": 0.85, "绍兴": 0.8, "黄山": 0.7, "张家界": 0.7, "洛阳": 0.6,
    "敦煌": 0.7, "延边": 0.6, "长白山": 0.7, "九寨沟": 0.75, "香格里拉": 0.7,
    "腾冲": 0.65, "漠河": 0.6, "喀纳斯": 0.7, "呼伦贝尔": 0.65, "林芝": 0.7,
    "景德镇": 0.65, "婺源": 0.6, "凤凰": 0.6, "平遥": 0.55, "乌镇": 0.8,
    "千岛湖": 0.8, "恩施": 0.6, "稻城亚丁": 0.7, "西双版纳": 0.7, "北海": 0.6,
    "开封": 0.55, "武当山": 0.6, "庐山": 0.65, "华山": 0.6, "泰山": 0.6,
    "峨眉山": 0.65, "普陀山": 0.8, "武夷山": 0.7, "青海湖": 0.65, "龙虎山": 0.6,
    "三清山": 0.65, "雁荡山": 0.7, "衡山": 0.6, "神农架": 0.65, "丹霞山": 0.6,
    "井冈山": 0.55, "崂山": 0.8, "鼓浪屿": 0.9, "涠洲岛": 0.7, "宏村": 0.6,
    "周庄": 0.75, "阳朔": 0.6, "嘉峪关": 0.6, "银川": 0.6, "西宁": 0.6,
    "乌鲁木齐": 0.7, "喀什": 0.55, "伊犁": 0.6, "满洲里": 0.6, "阿尔山": 0.6,
    "额济纳": 0.55, "禾木": 0.65, "赛里木湖": 0.6, "日喀则": 0.55, "阿里": 0.6,
    "霞浦": 0.6, "元阳": 0.55, "东极岛": 0.7, "荔波": 0.6, "色达": 0.55,
    "纳木错": 0.6, "三沙": 0.8, "抚远": 0.5,
}

# 住宿基准价格（元/晚/人）
ACCOMMODATION_BASE = {
    "budget": 80,    # 青旅/经济型
    "mid": 200,      # 舒适型
    "luxury": 500,   # 豪华型
}

# 交通方式基准价格（元/公里/人）
TRANSPORT_BASE = {
    "driving": 0.5,    # 自驾/打车
    "transit": 0.3,    # 公共交通
    "walking": 0,      # 步行
    "cycling": 0.05,   # 骑行
}


@tool
def estimate_budget(
    destination: str,
    days: int,
    routes: list[dict[str, Any]],
    accommodation_level: str = "mid",
    persons: int = 1,
    total_budget: float = 0,
) -> dict[str, Any]:
    """
    估算旅行预算。

    Args:
        destination: 目的地城市
        days: 旅行天数
        routes: 路线列表
        accommodation_level: 住宿等级 budget/mid/luxury
        persons: 人数
        total_budget: 用户总预算（0 表示无限制）

    Returns:
        {"transport": float, "accommodation": float, "food": float, "tickets": float, "total": float, ...}
    """
    price_index = CITY_PRICE_INDEX.get(destination, 0.8)

    # 交通费 = 市内通勤基础费 + 景点间路线费
    # （基础费兜底：路线估算只覆盖 POI 之间的移动，不含地铁/公交日常出行）
    transport = 15.0 * days * persons * price_index
    for route in routes:
        dist = route.get("distance_km", 0)
        mode = route.get("mode", "transit")
        rate = TRANSPORT_BASE.get(mode, 0.3)
        transport += dist * rate * price_index

    # 住宿费
    acc_rate = ACCOMMODATION_BASE.get(accommodation_level, 200)
    accommodation = acc_rate * max(1, days - 1) * min(persons, 2) * price_index

    # 餐饮费
    food_per_day = 80 * price_index
    food = food_per_day * days * persons

    # 门票/活动
    tickets_per_day = 60 * price_index
    tickets = tickets_per_day * days * persons

    transport = round(transport, 0)
    accommodation = round(accommodation, 0)
    food = round(food, 0)
    tickets = round(tickets, 0)
    total = transport + accommodation + food + tickets

    result: dict[str, Any] = {
        "transport": transport,
        "accommodation": accommodation,
        "food": food,
        "tickets": tickets,
        "other": round(total * 0.1, 0),  # 预留 10% 杂费
        "total": round(total * 1.1, 0),
        "per_day": round(total / days, 0) if days > 0 else 0,
        "price_level": "高" if price_index > 1.1 else ("中" if price_index > 0.8 else "低"),
        "note": "",
    }

    # 预算提醒（始终注明未含往返大交通）
    disclaimer = "未含往返大交通（机票/高铁）"
    if total_budget > 0:
        if result["total"] > total_budget:
            over = result["total"] - total_budget
            pct = over / total_budget * 100
            result["note"] = (
                f"⚠️ 超出预算约 {over:.0f} 元（{pct:.0f}%），"
                f"建议调整住宿等级或减少天数；{disclaimer}"
            )
        elif result["total"] > total_budget * 0.8:
            result["note"] = f"⚡ 预算使用率 {result['total']/total_budget*100:.0f}%，请合理控制支出；{disclaimer}"
        else:
            result["note"] = f"✅ 预算充足，剩余 {total_budget - result['total']:.0f} 元；{disclaimer}"
    else:
        result["note"] = f"ℹ️ {disclaimer}"

    return result