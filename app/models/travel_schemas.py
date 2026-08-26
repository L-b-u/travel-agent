"""Travel Agent API 请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TravelRequest(BaseModel):
    """旅行规划请求。"""

    user_input: str = Field(
        ...,
        description="用户自然语言偏好，如：'我想去杭州玩2天，预算1000元，喜欢博物馆和美食'",
        min_length=5,
        max_length=2000,
    )
    session_id: str = Field(default="default", description="会话 ID，用于多轮对话")


class ConfirmRequest(BaseModel):
    """人工确认请求（HITL 恢复）。"""

    session_id: str = Field(..., description="与触发中断时相同的会话 ID")
    approved: bool = Field(..., description="是否批准继续")
    note: str = Field(default="", description="备注（如拒绝原因）")


class PreferenceResponse(BaseModel):
    """结构化偏好。"""

    destination: str = Field(description="目的地城市")
    days: int = Field(default=1, ge=1, le=30, description="旅行天数")
    budget: float = Field(default=0, ge=0, description="总预算（元）")
    interests: list[str] = Field(default_factory=list, description="兴趣爱好")
    companions: str = Field(default="solo", description="同行人：solo/couple/family/friends")
    accommodation: str = Field(default="mid", description="住宿偏好：budget/mid/luxury")
    start_date: str | None = Field(default=None, description="出发日期 YYYY-MM-DD")
    notes: str = Field(default="", description="补充说明")


class TravelPlanResponse(BaseModel):
    """旅行计划响应。

    requires_confirmation=True 时表示流程已中断等待人工确认，
    应引导用户调用 /travel/confirm 端点提交决定。
    """

    session_id: str
    preferences: dict[str, Any]
    itinerary: str
    safety_result: dict[str, Any]
    requires_confirmation: bool
    confirmation_items: list[str]
    status: str | None = Field(
        default=None,
        description="流程状态：pending_confirmation / completed / cancelled",
    )
    research_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="研究阶段元信息（react/deterministic、工具调用次数等）",
    )
    tips_citations: list[str] = Field(
        default_factory=list,
        description="RAG 攻略引用来源",
    )


class EvalItineraryRequest(BaseModel):
    """行程评估请求（JSON 模式）：传入 Markdown 行程文本。"""

    markdown: str = Field(
        ...,
        description="Markdown 行程文本（含头部元信息注释时可做约束满足检查）",
        min_length=10,
    )


class EvalItineraryResponse(BaseModel):
    """行程评估响应：返回逐项检查结果与通过率。"""

    metadata: dict[str, Any] = Field(default_factory=dict, description="从行程头部解析的元信息")
    checks: dict[str, bool] = Field(default_factory=dict, description="逐项检查结果")
    passed_count: int = Field(description="通过项数")
    total_count: int = Field(description="总检查项数")
    pass_rate: str = Field(description="通过率，如 87.5%")
    failed_checks: list[str] = Field(default_factory=list, description="未通过的检查项")
    itinerary_length: int = Field(description="行程文本长度")
    details: dict[str, Any] = Field(default_factory=dict, description="检查明细")
    summary: str = Field(description="评估摘要")