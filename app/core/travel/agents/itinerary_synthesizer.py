"""
行程合成 Agent：将所有信息汇总为结构化 Markdown 旅行计划。

使用 LLM 生成可读的旅行行程。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.travel.state import TravelState

ITINERARY_SYSTEM_PROMPT = """你是专业旅行规划师。请根据提供的旅行信息，生成一份详细的 Markdown 格式旅行计划。

## 输出格式
```markdown
# {天数}天{天数-1}晚 {目的地}旅行计划

## 📋 约束摘要
- 目的地：{目的地}
- 天数：{天数}天
- 预算：{预算}元
- 兴趣：{兴趣列表}
- 同行人：{同行人}
- 住宿偏好：{住宿等级}

## 🗺️ 行程总览

| 时间 | 地点 | 交通 | 预算 | 备注 |
|------|------|------|------|------|
| ... | ... | ... | ... | ... |

## 📅 每日详细计划

### Day 1（M月D日 周X）
- **上午**：...
- **中午**：...
- **下午**：...
- **晚上**：...

### Day 2（M月D日 周X）
...

## 🌤️ 天气提醒
（根据天气情况给出建议；天气数据对应的就是上述出发日期起的预报）

## 💰 预算拆分
| 项目 | 费用 |
|------|------|
| 交通 | ...元 |
| 住宿 | ...元 |
| 餐饮 | ...元 |
| 门票 | ...元 |
| 其他 | ...元 |
| **总计** | **...元** |

## ⚠️ 注意事项
1. ...
2. ...
```

## 规则
- 每天安排 3-4 个景点，劳逸结合
- 若用户偏好含出发日期（start_date），每日标题必须标注对应的真实日期与星期，
  格式如 "Day 1（8月29日 周六）"，后续日期依次顺延；无出发日期才可只写 "Day N"
- 行程中的天气提醒必须与各天实际日期对应，不要张冠李戴
- 上午安排体力消耗大的景点，下午安排轻松的
- 标注各景点间的交通方式和时间
- 如果天气不佳，给出备选方案
- 结合"本地攻略参考"中的门票预约、避坑提示完善每日安排和注意事项
- 预算金额用阿拉伯数字表示（如"约450元"），并保留"约"字表示估算；不要用中文大写数字（壹贰叁）
- 所有价格标注"约"字，表示估算"""


async def synthesize_node(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """
    节点：行程合成。

    汇总偏好、POI、路线、天气、预算与 RAG 攻略摘录，使用 LLM 生成 Markdown 旅行计划。
    """
    preferences = state.get("preferences", {})
    pois = state.get("pois", [])
    routes = state.get("routes", [])
    weather = state.get("weather", [])
    budget = state.get("budget", {})
    tips = state.get("tips", [])

    # 构建 LLM 上下文（LLM 通过 RunnableConfig 注入，不进 state）
    llm = config.get("configurable", {}).get("llm")
    if llm:
        t0 = time.perf_counter()
        try:
            context = _build_context(preferences, pois, routes, weather, budget, tips)
            messages = [
                {"role": "system", "content": ITINERARY_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ]
            # 使用 LangChain ainvoke 接口
            token_cb = config.get("configurable", {}).get("token_callback")
            resp = await llm.ainvoke(messages, temperature=0.4, on_token=token_cb)
            duration_ms = (time.perf_counter() - t0) * 1000
            itinerary = resp.content if hasattr(resp, "content") else str(resp)
            # 清理可能的 markdown 代码块包裹
            itinerary = _clean_markdown(itinerary)
            usage = getattr(resp, "usage_metadata", None) or {}
            logger.info(
                "行程合成 LLM: {} 字符, 输入{}/输出{} token, 耗时 {:.0f}ms",
                len(itinerary), usage.get("input_tokens", 0), usage.get("output_tokens", 0), duration_ms,
            )
            return {"itinerary": _append_tips_footer(itinerary, tips)}
        except Exception as e:
            logger.warning("LLM 行程合成失败，使用模板生成: {} ({:.0f}ms)", e, (time.perf_counter() - t0) * 1000)

    # 降级：模板生成
    itinerary = _template_generate(preferences, pois, routes, weather, budget)
    return {"itinerary": _append_tips_footer(itinerary, tips)}


def _build_context(
    prefs: dict[str, Any],
    pois: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    weather: list[dict[str, Any]],
    budget: dict[str, Any],
    tips: list[dict[str, Any]] | None = None,
) -> str:
    """构建 LLM 输入上下文。"""
    parts = []

    parts.append(f"## 用户偏好\n{json.dumps(prefs, ensure_ascii=False, indent=2)}")

    if pois:
        poi_lines = []
        for i, poi in enumerate(pois[:20], 1):
            poi_lines.append(
                f"{i}. **{poi.get('name', '')}** "
                f"({poi.get('category', '')}) "
                f"评分: {poi.get('rating', 'N/A')} "
                f"坐标: ({poi.get('lat', 0):.4f}, {poi.get('lon', 0):.4f})"
            )
        parts.append("## 景点列表\n" + "\n".join(poi_lines))

    if routes:
        route_lines = []
        for r in routes:
            route_lines.append(
                f"- {r.get('origin', '')} → {r.get('destination', '')}: "
                f"{r.get('distance_km', 0)}km, "
                f"{r.get('duration_min', 0)}分钟 ({r.get('mode', '')})"
            )
        parts.append("## 路线信息\n" + "\n".join(route_lines))

    if weather:
        weather_lines = []
        for w in weather:
            weather_lines.append(
                f"- {w.get('date', '')}: {w.get('condition', '')}, "
                f"{w.get('temp_min', 0)}°C ~ {w.get('temp_max', 0)}°C, "
                f"降水概率 {w.get('precipitation_prob', 0)}%"
            )
        parts.append("## 天气预报\n" + "\n".join(weather_lines))

    if budget:
        parts.append(f"## 预算估算\n{json.dumps(budget, ensure_ascii=False, indent=2)}")

    if tips:
        tip_lines = [f"- [{t['citation']}] {t['text'][:200]}" for t in tips[:5]]
        parts.append("## 本地攻略参考（知识库摘录，写注意事项时请结合这些信息）\n" + "\n".join(tip_lines))

    return "\n\n".join(parts)


def _append_tips_footer(itinerary: str, tips: list[dict[str, Any]]) -> str:
    """行程尾部追加攻略引用来源（source grounding：标注信息出处）。"""
    if not tips:
        return itinerary
    citations = "、".join(dict.fromkeys(t["citation"] for t in tips[:5]))
    return itinerary + f"\n\n---\n\n> 📚 攻略参考来源：{citations}\n"


def _day_label(day_offset: int, start_date_str: str | None) -> str:
    """
    构造每日标题标签："Day N（8月29日 周六）"；无出发日期时退化为 "Day N"。

    Args:
        day_offset: 第几天（0 起）
        start_date_str: 出发日期 YYYY-MM-DD，可为 None
    """
    base = f"Day {day_offset + 1}"
    if not start_date_str:
        return base
    try:
        d = datetime.strptime(start_date_str, "%Y-%m-%d").date() + timedelta(days=day_offset)
        return f"{base}（{d.month}月{d.day}日 {'周' + '一二三四五六日'[d.weekday()]}）"
    except ValueError:
        return base


def _template_generate(
    prefs: dict[str, Any],
    pois: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    weather: list[dict[str, Any]],
    budget: dict[str, Any],
) -> str:
    """模板生成行程（LLM 不可用时的降级方案）。"""
    destination = prefs.get("destination", "目的地")
    days = prefs.get("days", 2)
    budget_total = prefs.get("budget", 0)
    interests = prefs.get("interests", [])
    accommodation = prefs.get("accommodation", "mid")
    companions = prefs.get("companions", "solo")
    start_date = prefs.get("start_date")

    companion_map = {"solo": "独自", "couple": "情侣", "family": "家庭", "friends": "朋友"}
    acc_map = {"budget": "经济型", "mid": "舒适型", "luxury": "豪华型"}

    date_line = ""
    if start_date:
        try:
            d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
            d1 = d0 + timedelta(days=max(days - 1, 0))
            date_line = (
                f"- 出行日期：{d0.month}月{d0.day}日（{'周' + '一二三四五六日'[d0.weekday()]}）"
                f" ~ {d1.month}月{d1.day}日（{'周' + '一二三四五六日'[d1.weekday()]}）"
            )
        except ValueError:
            date_line = ""

    summary_lines = [
        f"- 目的地：{destination}",
        f"- 天数：{days}天",
        f"- 预算：{budget_total}元" if budget_total else "- 预算：无限制",
        f"- 兴趣：{'、'.join(interests)}",
        f"- 同行人：{companion_map.get(companions, companions)}",
        f"- 住宿偏好：{acc_map.get(accommodation, accommodation)}",
    ]
    if date_line:
        summary_lines.insert(2, date_line)

    lines = [
        f"# {days}天{days - 1}晚 {destination}旅行计划",
        "",
        "## 📋 约束摘要",
        *summary_lines,
        "",
        "## 🗺️ 行程总览",
        "",
        "| 时间 | 地点 | 交通 | 预算 | 备注 |",
        "|------|------|------|------|------|",
    ]

    # 按天组织 POI
    daily_pois: list[list[dict[str, Any]]] = []
    chunk_size = max(3, len(pois) // days) if pois else 4
    for i in range(days):
        start = i * chunk_size
        daily_pois.append(pois[start:start + chunk_size])

    # 行程总览
    for day, day_pois in enumerate(daily_pois):
        label = _day_label(day, start_date)
        for j, poi in enumerate(day_pois):
            time_slot = ["上午", "中午", "下午", "晚上"][min(j, 3)]
            lines.append(
                f"| {label} {time_slot} | {poi.get('name', '')} | "
                f"步行/公交 | 约{budget.get('per_day', 0) / max(len(day_pois), 1):.0f}元 | "
                f"{poi.get('category', '')} |"
            )

    lines.append("")
    lines.append("## 📅 每日详细计划")

    for day, day_pois in enumerate(daily_pois):
        lines.append(f"### {_day_label(day, start_date)}")
        for j, poi in enumerate(day_pois):
            time_slot = ["上午", "中午", "下午", "晚上"][min(j, 3)]
            rating = poi.get("rating", "N/A")
            lines.append(f"- **{time_slot}**：{poi.get('name', '')}（评分 {rating}）")
        lines.append("")

    # 天气
    if weather:
        lines.append("## 🌤️ 天气提醒")
        for w in weather:
            lines.append(f"- {w.get('date', '')}: {w.get('condition', '')}，"
                        f"{w.get('temp_min', 'N/A')}°C ~ {w.get('temp_max', 'N/A')}°C")
        lines.append("")

    # 预算
    if budget:
        lines.append("## 💰 预算拆分")
        lines.append("| 项目 | 费用 |")
        lines.append("|------|------|")
        lines.append(f"| 交通 | 约{budget.get('transport', 0):.0f}元 |")
        lines.append(f"| 住宿 | 约{budget.get('accommodation', 0):.0f}元 |")
        lines.append(f"| 餐饮 | 约{budget.get('food', 0):.0f}元 |")
        lines.append(f"| 门票 | 约{budget.get('tickets', 0):.0f}元 |")
        lines.append(f"| 其他 | 约{budget.get('other', 0):.0f}元 |")
        lines.append(f"| **总计** | **约{budget.get('total', 0):.0f}元** |")
        if budget.get("note"):
            lines.append(f"\n{budget['note']}")
        lines.append("")

    # 注意事项
    lines.append("## ⚠️ 注意事项")
    lines.append("1. 以上价格为估算，实际消费以当地为准")
    lines.append("2. 建议提前查看景点开放时间，部分景点需预约")
    lines.append("3. 出行前请关注当地天气预报，做好防晒/防雨准备")
    lines.append("4. 保管好个人财物，注意人身安全")

    return "\n".join(lines)


def _clean_markdown(text: str) -> str:
    """清理 LLM 输出中的 markdown 代码块包裹。"""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    elif text.startswith("```md"):
        text = text[len("```md"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text