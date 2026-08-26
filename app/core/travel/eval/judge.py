"""
LLM-as-Judge：用 LLM 按评分量表（rubric）对生成的行程打分。

与规则评估的分工：
- 规则检查（evaluator.py）：毫秒级、确定性，负责回归拦截（结构/约束/安全合规）；
- LLM 评审（本模块）：捕捉规则覆盖不到的质量维度——行程是否真的合理可执行、
  备选方案是否有诚意、表达是否清晰。两者互补，规则先跑，judge 补深度。

评审输出结构化分数（Pydantic Schema），避免自由文本解析。
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

JUDGE_SYSTEM_PROMPT = """你是严格的旅行行程质量评审员。根据用户需求与生成的 Markdown 行程，
按以下五个维度各打 0-5 分（整数），并给出简要理由。

评分维度：
1. constraint_satisfaction 约束满足：目的地/天数/预算/兴趣等明确要求是否全部满足
2. feasibility 可执行性：每日安排是否地理顺路、时间是否充裕、交通衔接是否交代清楚
3. actionability 信息完备：是否包含门票预约方式、开放时间、注意事项等落地信息
4. contingency 应变能力：天气不佳/景点关门等情况是否给出有诚意的备选方案（而非空话）
5. clarity 表达质量：结构是否清晰、预算拆分是否可信、有无编造事实的迹象

打分纪律：
- 有实质内容才给分，套话空话不给分；发现编造（如不存在的景点信息、虚构价格）该维度 ≤2 分
- 整体评价 verdict 用一两句话指出最大的一处不足"""

DIMENSIONS = [
    "constraint_satisfaction",
    "feasibility",
    "actionability",
    "contingency",
    "clarity",
]


class JudgeScores(BaseModel):
    """LLM 评审的结构化输出。"""

    constraint_satisfaction: int = Field(ge=0, le=5, description="约束满足 0-5")
    feasibility: int = Field(ge=0, le=5, description="可执行性 0-5")
    actionability: int = Field(ge=0, le=5, description="信息完备 0-5")
    contingency: int = Field(ge=0, le=5, description="应变能力 0-5")
    clarity: int = Field(ge=0, le=5, description="表达质量 0-5")
    reasons: dict[str, str] = Field(default_factory=dict, description="各维度打分理由")
    verdict: str = Field(default="", description="整体评价与最大不足")

    @property
    def total(self) -> int:
        return sum(getattr(self, d) for d in DIMENSIONS)

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["total"] = self.total
        return data


async def judge_itinerary(
    user_input: str,
    preferences: dict[str, Any],
    itinerary: str,
    llm: Any,
) -> dict[str, Any] | None:
    """
    对单份行程做 LLM 评审。

    Args:
        user_input: 用户原始需求
        preferences: 提取后的结构化偏好
        itinerary: 行程 Markdown 正文
        llm: ModelRouter 实例

    Returns:
        评分字典；LLM 不可用时返回 None（调用方跳过，不影响规则评估）
    """
    if llm is None:
        return None

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## 用户需求\n{user_input}\n\n"
                f"## 结构化偏好\n{preferences}\n\n"
                f"## 待评审行程\n{itinerary[:6000]}"
            ),
        },
    ]
    try:
        scores: JudgeScores = await llm.ainvoke_structured(messages, JudgeScores, temperature=0.0)
        result = scores.to_dict()
        logger.info("LLM 评审完成: {}/25 分", result["total"])
        return result
    except Exception as e:
        logger.warning("LLM 评审失败（跳过，不影响规则评估）: {}", e)
        return None


def aggregate_judge_scores(details_list: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总多条用例的评审分：均值 + 各维度均值。"""
    totals = []
    dim_sums: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    for d in details_list:
        j = d.get("judge")
        if not j:
            continue
        totals.append(j["total"])
        for dim in DIMENSIONS:
            dim_sums[dim].append(j[dim])

    n = len(totals)
    if n == 0:
        return {"count": 0}

    return {
        "count": n,
        "avg_total": round(sum(totals) / n, 2),
        "max_total": max(totals),
        "min_total": min(totals),
        "avg_by_dimension": {dim: round(sum(v) / len(v), 2) for dim, v in dim_sums.items() if v},
    }
