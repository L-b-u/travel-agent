"""
偏好收集 Agent：将自然语言需求转为结构化偏好。

主路径：LLM 结构化输出（with_structured_output / function calling），
LLM 返回 Pydantic 实例而非自由文本，无需正则抠 JSON。
兜底路径：基于规则的抽取（LLM 不可用或输出不合法时），保证全流程可用。
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field, field_validator


class TravelPreferences(BaseModel):
    """结构化旅行偏好（LLM 结构化输出的目标 Schema）。

    枚举字段必须容忍 LLM 返回 null：模型对未提及项常输出 null 而非省略键，
    校验器将 None/空串归一化为默认值，避免整体解析失败。
    """

    destination: str = Field(default="杭州", description="目的地城市名（中国城市）")
    days: int = Field(default=2, ge=1, le=30, description="旅行天数")
    budget: float = Field(default=0, ge=0, description="总预算（元），0 表示无限制")
    interests: list[str] = Field(default_factory=list, description="兴趣类别（受控词表，供分类逻辑用）")
    interests_raw: list[str] = Field(
        default_factory=list,
        description="用户原话中的兴趣词（如'大熊猫'），未归类，供 POI 检索直接当关键词",
    )
    companions: Literal["solo", "couple", "family", "friends"] = Field(
        default="solo", description="同行人类型",
    )
    accommodation: Literal["budget", "mid", "luxury"] = Field(
        default="mid", description="住宿偏好等级",
    )
    start_date: str | None = Field(default=None, description="出发日期 YYYY-MM-DD，未提及为 null")
    notes: str = Field(default="", description="补充说明")

    @field_validator("interests_raw", mode="before")
    @classmethod
    def _clean_interests_raw(cls, v: Any) -> list[str]:
        """原话词只做去空格/去重/截断，不归类不丢弃。"""
        if not isinstance(v, list):
            return []
        seen: set = set()
        result: list[str] = []
        for item in v:
            word = str(item).strip()
            if word and len(word) <= 20 and word not in seen:
                seen.add(word)
                result.append(word)
        return result[:8]

    @field_validator("companions", mode="before")
    @classmethod
    def _default_companions(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "solo"
        return v

    @field_validator("accommodation", mode="before")
    @classmethod
    def _default_accommodation(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "mid"
        return v

    @field_validator("interests", mode="before")
    @classmethod
    def _normalize_interests(cls, v: Any) -> list[str]:
        """兴趣归一化为受控词表。

        三级匹配：精确命中 → 英文别名 → 关键词模糊归类（如"大熊猫"→自然），
        避免 LLM 返回具体名词时被当作无效项静默丢弃。
        """
        allowed = {"景点", "美食", "博物馆", "自然", "历史", "购物", "宗教", "建筑", "公园", "咖啡"}
        alias = {
            "museum": "博物馆", "food": "美食", "nature": "自然",
            "history": "历史", "shopping": "购物", "temple": "宗教",
            "architecture": "建筑", "park": "公园", "cafe": "咖啡",
        }
        # 子串归类：LLM 常返回具体名词而非类别词
        hints = [
            ("熊猫", "自然"), ("动物园", "自然"), ("瀑布", "自然"), ("森林", "自然"),
            ("雪山", "自然"), ("湖", "自然"), ("海", "自然"), ("山", "自然"), ("江", "自然"),
            ("古镇", "历史"), ("古城", "历史"), ("遗址", "历史"), ("文物", "历史"),
            ("寺", "宗教"), ("庙", "宗教"), ("教堂", "宗教"),
            ("火锅", "美食"), ("小面", "美食"), ("串串", "美食"), ("小吃", "美食"),
            ("烤肉", "美食"), ("菜", "美食"), ("吃", "美食"),
            ("博物", "博物馆"), ("展馆", "博物馆"), ("美术馆", "博物馆"),
            ("商场", "购物"), ("免税", "购物"), ("乐园", "景点"), ("主题园", "景点"),
        ]
        result: list[str] = []
        if not isinstance(v, list):
            return result
        for item in v:
            word = str(item).strip()
            word = alias.get(word.lower(), word)
            if word in allowed:
                pass  # 精确命中
            else:
                matched = next((cat for kw, cat in hints if kw in word), None)
                word = matched or ""  # 无法归类则丢弃
            if word and word in allowed and word not in result:
                result.append(word)
        return result or ["景点", "美食"]

    @field_validator("start_date", mode="before")
    @classmethod
    def _validate_start_date(cls, v: Any) -> str | None:
        if v in (None, "", "null"):
            return None
        text = str(v)
        return text if re.match(r"^\d{4}-\d{2}-\d{2}$", text) else None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


PREFERENCE_SYSTEM_PROMPT = """你是一个旅行偏好分析助手。从用户输入中提取结构化旅行偏好，以 JSON 对象输出。

规则：
- destination 必须是中国城市名
- days 默认为 2，范围 1-30
- budget 为 0 表示无限制
- interests 从以下选择：景点、美食、博物馆、自然、历史、购物、宗教、建筑、公园、咖啡
- interests_raw 原样保留用户提到的具体兴趣词（如"大熊猫""洪崖洞""火锅"），不归类不翻译
- 类别映射参考："大熊猫/动物园/雪山"→自然，"火锅/小面/小吃"→美食，"古镇/故宫"→历史
- 只提取用户明确提到的兴趣，不要根据目的地推断用户未提及的兴趣
- companions: solo/couple/family/friends；accommodation: budget/mid/luxury
- 缺失字段使用合理默认值
- start_date 必须是具体日期（YYYY-MM-DD）。把用户的相对时间表达结合"当前日期与星期"
  换算成确切日期："今天"=当天；"明天"=+1 天；"后天"=+2 天；
  "这周末/周末去"=最近的周六（若今天是周六则为今天）；"这周六""这周日"同理；
  "下周末"=下一周的周六；"下周X""星期X"按日历推算。完全未提及出行时间才填 null"""


def _weekday_cn(d: date) -> str:
    """日期转中文星期（如"周三"）。"""
    return "周" + "一二三四五六日"[d.weekday()]


def _resolve_relative_date(user_input: str, today: date) -> str | None:
    """
    规则兜底的相对日期解析（LLM 不可用时）。

    支持：今天/明天/后天/大后天、这周末/本周末、下周末、（这|本|下）周X/星期X。
    "这周末"取最近的周六（含今天）；周日说"这周末"视为当天。
    返回 YYYY-MM-DD 或 None。
    """

    def fmt(d: date) -> str:
        return d.isoformat()

    if any(w in user_input for w in ["今天", "今晚"]):
        return fmt(today)
    if "明天" in user_input:
        return fmt(today + timedelta(days=1))
    if "大后天" in user_input:
        return fmt(today + timedelta(days=3))
    if "后天" in user_input:
        return fmt(today + timedelta(days=2))

    # 周内具体某天：（这|本|下）（周|星期）X / 周X / 星期X
    weekday_names = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    m = re.search(r"(这|本|下)?(?:周|星期)([一二三四五六日天])", user_input)
    if m:
        prefix, name = m.group(1), m.group(2)
        target = weekday_names[name]
        delta = (target - today.weekday()) % 7
        candidate = today + timedelta(days=delta)
        if delta == 0:
            return fmt(candidate)  # 今天就是目标星期
        if prefix == "下":
            # "下周X"：下一个完整周的同一天（至少再过一周内）
            candidate = candidate + timedelta(days=7) if delta <= 6 - today.weekday() else candidate
            return fmt(candidate)
        if delta < (7 - today.weekday()):  # 本周内
            return fmt(candidate)
        return fmt(candidate)

    # 周末表达："这周末/本周末" → 最近周六（含今天）；"下周末" → 下周的周六
    if any(w in user_input for w in ["周末"]):
        if "下" in user_input.replace("下个月", ""):
            next_monday = today + timedelta(days=7 - today.weekday())
            return fmt(next_monday + timedelta(days=5))
        days_to_sat = (5 - today.weekday()) % 7
        return fmt(today + timedelta(days=days_to_sat))

    return None


async def collect_preferences_node(state: dict, config: RunnableConfig) -> dict[str, Any]:
    """
    节点 1：偏好收集。

    主路径：LLM 结构化输出直接返回 TravelPreferences 实例。
    兜底路径：LLM 不可用/失败时，规则抽取（城市匹配、正则提数字与关键词）。
    """
    user_input = state.get("user_input", "")

    defaults = TravelPreferences().to_dict()
    if not user_input.strip():
        return {"preferences": defaults, "error": "用户输入为空"}

    # LLM 通过 RunnableConfig 注入（ModelRouter，含 ainvoke_structured 能力）
    llm = config.get("configurable", {}).get("llm")

    preferences = defaults
    if llm:
        try:
            messages = [
                {"role": "system", "content": PREFERENCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"当前日期：{date.today().isoformat()}（{_weekday_cn(date.today())}）\n"
                        f"用户需求：\n{user_input}"
                    ),
                },
            ]
            parsed: TravelPreferences = await llm.ainvoke_structured(messages, TravelPreferences, temperature=0.1)
            preferences = parsed.to_dict()
            logger.info("偏好收集完成(结构化输出): {}", preferences)
            return {"preferences": preferences}
        except Exception as e:
            logger.warning("LLM 结构化偏好提取失败，降级到规则兜底: {}", e)

    # 规则兜底
    fallback_prefs = dict(defaults)
    _fallback_parse(user_input, fallback_prefs)
    logger.info("偏好收集完成(规则兜底): {}", fallback_prefs)
    return {"preferences": fallback_prefs}


def _fallback_parse(user_input: str, prefs: dict[str, Any]) -> None:
    """基于规则的偏好提取兜底。"""
    # 相对日期 → 具体出发日（如"这周末"→最近周六）
    resolved = _resolve_relative_date(user_input, date.today())
    if resolved:
        prefs["start_date"] = resolved

    # 提取目的地（从已知城市列表匹配，避免误用默认值"杭州"）
    from app.core.travel.tools._poi_data import CITY_COORDS
    for city in CITY_COORDS:
        if city in user_input:
            prefs["destination"] = city
            break

    # 提取天数
    days_patterns = [
        r"(\d+)\s*天", r"(\d+)\s*日", r"玩\s*(\d+)", r"(\d+)\s*晚",
    ]
    for pat in days_patterns:
        m = re.search(pat, user_input)
        if m:
            prefs["days"] = max(1, min(30, int(m.group(1))))
            break

    # 提取预算
    budget_patterns = [
        r"预算\s*(\d+)", r"(\d+)\s*元", r"(\d+)\s*块", r"不超过\s*(\d+)",
    ]
    for pat in budget_patterns:
        m = re.search(pat, user_input)
        if m:
            prefs["budget"] = float(m.group(1))
            break

    # 提取兴趣
    interest_keywords = {
        "博物馆": "博物馆", "museum": "博物馆",
        "美食": "美食", "吃": "美食", "小吃": "美食", "餐厅": "美食",
        "火锅": "美食", "小面": "美食", "美食街": "美食", "特色菜": "美食",
        "自然": "自然", "风景": "自然", "美景": "自然", "山水": "自然", "公园": "自然", "海": "自然",
        "购物": "购物", "买": "购物", "逛街": "购物",
        "历史": "历史", "古迹": "历史", "古城": "历史", "古镇": "历史",
        "宗教": "宗教", "寺庙": "宗教", "教堂": "宗教", "佛": "宗教",
        "咖啡": "咖啡", "cafe": "咖啡",
        "建筑": "建筑",
    }
    detected = []
    raw_hits = []
    for kw, cat in interest_keywords.items():
        if kw.lower() in user_input.lower():
            raw_hits.append(kw)
            if cat not in detected:
                detected.append(cat)
    if detected:
        prefs["interests"] = detected
    if raw_hits:
        prefs["interests_raw"] = list(dict.fromkeys(raw_hits))[:8]

    # 同行人
    if any(w in user_input for w in ["情侣", "女朋友", "男朋友", "对象", "couple"]):
        prefs["companions"] = "couple"
    elif any(w in user_input for w in ["家庭", "孩子", "小孩", "亲子", "family"]):
        prefs["companions"] = "family"
    elif any(w in user_input for w in ["朋友", "同学", "friends", "闺蜜", "兄弟"]):
        prefs["companions"] = "friends"

    # 住宿偏好
    if any(w in user_input for w in ["豪华", "五星", "高端", "luxury"]):
        prefs["accommodation"] = "luxury"
    elif any(w in user_input for w in ["经济", "便宜", "青旅", "穷游", "budget"]):
        prefs["accommodation"] = "budget"
