# -*- coding: utf-8 -*-
"""
Travel Agent 共享状态定义。

LangGraph 各节点通过此 TypedDict 读写流水线状态。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict


class POI(TypedDict, total=False):
    """景点/兴趣点。"""

    name: str
    lat: float
    lon: float
    category: str  # attraction / restaurant / museum / hotel / ...
    rating: Optional[float]
    address: Optional[str]
    opening_hours: Optional[str]
    description: Optional[str]


class RouteSegment(TypedDict, total=False):
    """一段路线。"""

    origin: str
    destination: str
    distance_km: float
    duration_min: float
    mode: str  # walk / drive / transit


class WeatherInfo(TypedDict, total=False):
    """单日天气信息。"""

    date: str
    condition: str  # sunny / cloudy / rainy / ...
    temp_max: float
    temp_min: float
    precipitation_prob: int
    wind_speed: float


class BudgetBreakdown(TypedDict, total=False):
    """预算拆分。"""

    transport: float
    accommodation: float
    food: float
    tickets: float
    other: float
    total: float


class TravelState(TypedDict, total=False):
    """Travel Agent 全流水线共享状态。"""

    # ---- 用户输入 ----
    user_input: str
    session_id: str

    # ---- 结构化偏好 ----
    preferences: Dict[str, Any]

    # ---- POI 搜索结果 ----
    pois: List[Dict[str, Any]]

    # ---- 路线规划 ----
    routes: List[Dict[str, Any]]

    # ---- 研究阶段（ReAct Agent / 确定性兜底共用）----
    research_summary: str          # Agent 对目的地布局的自然语言总结
    research_trace: List[Dict[str, Any]]  # 工具调用轨迹（工具/参数/耗时）
    research_meta: Dict[str, Any]  # {"mode": "react"|"deterministic", ...}
    tips: List[Dict[str, Any]]     # RAG 攻略知识库检索结果（含引用来源）

    # ---- 天气信息 ----
    weather: List[Dict[str, Any]]

    # ---- 预算估算 ----
    budget: Dict[str, Any]

    # ---- 最终行程 ----
    itinerary: str

    # ---- 安全审查 ----
    safety_result: Dict[str, Any]

    # ---- HITL 人工确认（safety_review → human_gate 中断/resume）----
    requires_confirmation: bool
    confirmation_items: List[str]
    confirmation_decision: Dict[str, Any]  # {"approved": bool, "note": str}

    # ---- 流程终态 ----
    status: Optional[str]  # "completed" / "cancelled" / None(进行中)

    # ---- 错误信息 ----
    error: Optional[str]