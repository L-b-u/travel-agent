"""
Travel Agent 评估器。

评估核心对象：最终输出的 Markdown 旅行计划（itinerary）。
辅助评估对象：安全审查结果（safety_result）。

评估维度：
1. 结构完整性：标题、日程、预算拆分
2. 约束满足：目的地、天数、预算是否在行程中体现
3. 偏好纳入：用户兴趣是否在行程中体现
4. 信息来源：POI 是否出现在行程中（基于真实数据而非编造）
5. 不确定性说明：天气信息、备选方案
6. 安全合规：敏感操作是否被拦截
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.travel.graph import pending_confirmation, run_travel_agent
from app.core.travel.output import load_itinerary, save_itinerary


@dataclass
class EvalCase:
    """单条评估用例。"""

    id: str
    type: str
    input: str
    expected: dict[str, Any]


@dataclass
class EvalResult:
    """单条评估结果。"""

    case_id: str
    case_type: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class EvalRunner:
    """评估运行器：基于最终输出的 Markdown 行程进行评估。"""

    def __init__(self, cases_path: str | None = None) -> None:
        if cases_path is None:
            cases_path = str(Path(__file__).parent / "cases.json")
        self._cases = self._load_cases(cases_path)
        self._results: list[EvalResult] = []

    @staticmethod
    def _load_cases(path: str) -> list[EvalCase]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [
            EvalCase(id=item["id"], type=item["type"], input=item["input"],
                     expected=item.get("expected", {}))
            for item in data
        ]

    async def run(self, llm: Any = None, *, use_judge: bool = True) -> dict[str, Any]:
        """运行全部评估用例。每条用例生成 Markdown 文件，从文件读取评估。

        Args:
            llm: ModelRouter 实例（None 时全走规则兜底）
            use_judge: 是否在规则检查之外追加 LLM-as-judge 评分（需 llm）
        """
        from app.core.travel.eval.judge import judge_itinerary

        self._results = []
        for i, case in enumerate(self._cases, 1):
            print(f"\n{'='*60}")
            print(f"  [{i}/{len(self._cases)}] {case.id} ({case.type})")
            print(f"  输入: {case.input}")
            print(f"{'='*60}")

            try:
                result = await run_travel_agent(
                    user_input=case.input,
                    session_id=f"eval_{case.id}",
                    llm=llm,
                )

                # 被人工确认中断的用例：拒绝即视为终态（评估取消说明的合规性）
                if pending_confirmation(result):
                    result["__interrupted__"] = True

                # 保存行程为 Markdown 文件
                itinerary_raw = result.get("itinerary", "")
                filepath = save_itinerary(
                    itinerary=itinerary_raw,
                    session_id=f"eval_{case.id}",
                    user_input=case.input,
                    preferences=result.get("preferences", {}),
                )
                print(f"  文件: {filepath}")

                # 从文件读取行程内容（评估的是文件内容，不是内存中间状态）
                itinerary = load_itinerary(filepath)
                result["itinerary"] = itinerary

                passed, details = self._evaluate_case(case, result)
                details["filepath"] = filepath
                details["itinerary"] = itinerary

                # LLM-as-judge 补充评分（不参与 pass 判定，只做质量观测）
                if use_judge and llm is not None and case.type != "安全边界":
                    judge_result = await judge_itinerary(case.input, result.get("preferences", {}), itinerary, llm)
                    if judge_result:
                        details["judge"] = judge_result
                        print(f"  LLM 评审: {judge_result['total']}/25 分")

                self._results.append(EvalResult(
                    case_id=case.id, case_type=case.type,
                    passed=passed, details=details,
                ))

                # 打印行程摘要
                print(f"\n  --- 行程文件 ({len(itinerary)} 字符) ---")
                preview = itinerary[:400]
                for line in preview.split("\n"):
                    print(f"  {line}")
                if len(itinerary) > 400:
                    print(f"  ... (剩余 {len(itinerary) - 400} 字符)")

                # 打印评估结果
                status = "✅ 通过" if passed else "❌ 失败"
                print(f"\n  评估: {status}")
                failed_checks = details.get("failed_checks", [])
                if failed_checks:
                    print(f"  未通过: {', '.join(failed_checks)}")
                for check_name, check_passed in details.get("checks", {}).items():
                    mark = "✅" if check_passed else "❌"
                    print(f"    {mark} {check_name}")

            except Exception as e:
                logger.exception("评估异常: [{}]", case.id)
                self._results.append(EvalResult(
                    case_id=case.id, case_type=case.type,
                    passed=False, error=str(e),
                ))
                print(f"  ❌ 异常: {e}")

        return self._summarize()

    def _evaluate_case(self, case: EvalCase, result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """基于最终输出的 Markdown 行程评估。"""
        itinerary = result.get("itinerary", "")
        safety = result.get("safety_result", {})
        pois = result.get("pois", [])
        expected = case.expected

        # 评估前剥离头部 HTML 注释（元信息）。
        # 否则"目的地/预算在行程中"会匹配到注释里的元信息自身，造成假阳性。
        itinerary = _strip_itinerary_header(itinerary)

        checks: dict[str, bool] = {}
        details: dict[str, Any] = {"itinerary_length": len(itinerary)}

        # ============================================================
        # 通用检查：所有非安全边界类都评估行程文档质量
        # （与 evaluate_itinerary 共用 _check_structure，避免两套逻辑脱钩）
        # ============================================================
        if case.type != "安全边界":
            struct_checks, struct_details = _check_structure(itinerary)
            checks.update(struct_checks)
            details.update(struct_details)

        # ============================================================
        # 按类型深度评估（基于行程文本）
        # ============================================================
        if case.type == "安全边界":
            # 安全边界用例支持两个方向：
            #   safety_blocked=true（默认）：敏感操作必须被拦截/转人工确认
            #   safety_blocked=false：正常旅行请求必须放行（误伤回归检查）
            expect_blocked = expected.get("safety_blocked", True)
            blocked = safety.get("passed", True) is False
            requires_conf = result.get("requires_confirmation", False) or bool(result.get("__interrupted__"))

            if expect_blocked:
                checks["安全拦截"] = blocked or requires_conf
                details["blocked_keywords"] = safety.get("blocked_keywords", [])
            else:
                checks["无误伤放行"] = (not blocked) and (not requires_conf)
                details["warnings"] = safety.get("warnings", [])

        elif case.type == "常规规划":
            # 常规规划：目的地、天数在行程中体现，POI 出现在行程中
            if "destination" in expected:
                dest = expected["destination"]
                checks["目的地在行程中"] = dest in itinerary

            if "days" in expected:
                # 检查行程中是否体现了天数（"2天""两天""Day 1/Day 2"等）
                # 中文数字变体覆盖 "二/两" 口语差异，避免 LLM 输出"三天"时漏判
                days = expected["days"]
                day_patterns = [f"{days}天", f"{days}日", f"{days}晚", f"Day {days}", f"day {days}"]
                for cn in _chinese_number_variants(days):
                    day_patterns += [f"{cn}天", f"{cn}日", f"{cn}晚", f"第{cn}天"]
                checks["天数在行程中"] = any(p in itinerary for p in day_patterns)

            # POI 出现在行程中（信息来源标注：基于真实搜索数据）
            poi_names = [p.get("name", "") for p in pois[:8] if p.get("name")]
            poi_in_itinerary = sum(1 for name in poi_names if name and name in itinerary)
            checks["POI在行程中"] = poi_in_itinerary >= 2
            details["poi_in_itinerary"] = poi_in_itinerary

        elif case.type == "预算约束":
            # 预算约束：用户预算在行程中体现，预算拆分合理
            if "budget" in expected:
                budget = expected["budget"]
                # 行程中是否提到预算金额（含中文数字变体，如 "1000元" / "一千元" / "一千块"）
                budget_int = int(budget)
                budget_patterns = [str(budget_int), f"{budget_int}元", f"{budget_int}块"]
                for cn in _chinese_number_variants(budget_int):
                    budget_patterns += [f"{cn}元", f"{cn}块"]
                checks["预算在行程中"] = any(p in itinerary for p in budget_patterns)

            if "accommodation" in expected:
                # 住宿等级：luxury → "五星/豪华/高档", budget → "经济/青旅/客栈"
                acc = expected["accommodation"]
                if acc == "luxury":
                    acc_keywords = ["五星", "豪华", "高档", "高端", "星级酒店"]
                else:
                    acc_keywords = ["经济", "青旅", "客栈", "民宿", "性价比", "便宜"]
                checks["住宿等级匹配"] = any(kw in itinerary for kw in acc_keywords)

        elif case.type == "偏好约束":
            # 偏好约束：用户兴趣在行程中体现
            interests = expected.get("interests", [])
            interest_keywords = {
                "博物馆": ["博物馆", "博物院", "展馆", "展览"],
                "历史": ["历史", "古迹", "古城", "遗址", "文物", "古", "文化"],
                "美食": ["美食", "小吃", "餐厅", "火锅", "特色菜", "当地菜", "美食街"],
                "自然": ["自然", "山", "湖", "风景", "公园", "景区", "瀑布"],
                "景点": ["景点", "景区", "名胜", "地标", "打卡"],
            }
            if interests:
                matched = 0
                for interest in interests:
                    keywords = interest_keywords.get(interest, [interest])
                    if any(kw in itinerary for kw in keywords):
                        matched += 1
                checks["兴趣在行程中"] = matched >= len(interests) * 0.5
                details["interests_matched"] = f"{matched}/{len(interests)}"

            # 不喜欢购物
            if expected.get("no_shopping"):
                shopping_keywords = ["购物中心", "商场购物", "免税店", "购物天堂"]
                checks["未推荐购物"] = not any(kw in itinerary for kw in shopping_keywords)

        elif case.type == "变化处理":
            # 变化处理：行程中有天气信息、备选方案
            if expected.get("has_weather_plan"):
                weather_keywords = ["天气", "下雨", "雨", "晴", "阴", "气温", "降水", "降雨"]
                checks["有天气应对"] = any(kw in itinerary for kw in weather_keywords)

            if expected.get("has_backup"):
                backup_keywords = ["备选", "备用", "替代", "如果", "万一", "可选", "方案"]
                checks["有备选方案"] = any(kw in itinerary for kw in backup_keywords)

            if expected.get("has_health_note"):
                health_keywords = ["高反", "高原反应", "健康", "身体", "氧气", "海拔", "注意"]
                checks["有健康提醒"] = any(kw in itinerary for kw in health_keywords)

        # 所有检查项都通过才算 passed
        passed = all(checks.values()) if checks else False
        details["checks"] = checks
        details["failed_checks"] = [k for k, v in checks.items() if not v]

        return passed, details

    def _summarize(self) -> dict[str, Any]:
        """汇总评估结果。"""
        from app.core.travel.eval.judge import aggregate_judge_scores

        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed
        errors = sum(1 for r in self._results if r.error)

        by_type: dict[str, dict[str, int]] = {}
        for r in self._results:
            if r.case_type not in by_type:
                by_type[r.case_type] = {"total": 0, "passed": 0}
            by_type[r.case_type]["total"] += 1
            if r.passed:
                by_type[r.case_type]["passed"] += 1

        details_payload = [
            {
                "case_id": r.case_id,
                "type": r.case_type,
                "passed": r.passed,
                "error": r.error,
                "details": r.details,
            }
            for r in self._results
        ]

        return {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": f"{passed / total * 100:.1f}%" if total > 0 else "N/A",
            "by_type": by_type,
            "judge_summary": aggregate_judge_scores([d["details"] for d in details_payload]),
            "details": details_payload,
        }

    def print_report(self, summary: dict[str, Any]) -> None:
        """打印评估报告。"""
        print("\n" + "=" * 60)
        print("  Travel Agent Eval Report（基于 Markdown 行程输出评估）")
        print("=" * 60)
        print(f"  Total Cases: {summary['total_cases']}")
        print(f"  Passed:      {summary['passed']} ✅")
        print(f"  Failed:      {summary['failed']} ❌")
        print(f"  Errors:      {summary['errors']} ⚠️")
        print(f"  Pass Rate:   {summary['pass_rate']}")
        print("-" * 60)
        print("  By Type:")
        for t, stats in summary.get("by_type", {}).items():
            rate = f"{stats['passed'] / stats['total'] * 100:.0f}%" if stats["total"] > 0 else "N/A"
            print(f"    {t}: {stats['passed']}/{stats['total']} ({rate})")

        judge = summary.get("judge_summary", {})
        if judge.get("count"):
            print("-" * 60)
            lo, hi = judge["min_total"], judge["max_total"]
            print(f"  LLM 评审 (rubric /25): 均分 {judge['avg_total']}，区间 [{lo}, {hi}]")
            for dim, avg in judge.get("avg_by_dimension", {}).items():
                print(f"    {dim}: {avg}")
        print("-" * 60)
        for detail in summary.get("details", []):
            status = "✅" if detail["passed"] else "❌"
            err = f" (Error: {detail['error']})" if detail.get("error") else ""
            print(f"  [{status}] {detail['case_id']} ({detail['type']}){err}")
            if not detail["passed"] and not detail.get("error"):
                failed_checks = detail.get("details", {}).get("failed_checks", [])
                if failed_checks:
                    print(f"         未通过: {', '.join(failed_checks)}")
        print("=" * 60)


async def run_eval(llm: Any = None) -> dict[str, Any]:
    """便捷函数：运行评估并打印报告。"""
    runner = EvalRunner()
    summary = await runner.run(llm=llm)
    runner.print_report(summary)
    return summary


# ============================================================
# 独立评估接口：对任意 Markdown 行程文档做质量评估
# 不依赖测试用例、不调用 LLM、不联网，纯规则毫秒级返回。
# ============================================================

# 通用兴趣关键词（从用户输入提取兴趣，并检查行程是否体现）
_INTEREST_KEYWORDS: dict[str, list[str]] = {
    "博物馆": ["博物馆", "博物院", "展馆", "展览"],
    "历史": ["历史", "古迹", "古城", "遗址", "文物", "文化"],
    "美食": ["美食", "小吃", "餐厅", "火锅", "特色菜", "当地菜", "美食街"],
    "自然": ["自然", "山", "湖", "风景", "公园", "景区", "瀑布"],
    "景点": ["景点", "景区", "名胜", "地标", "打卡"],
    "购物": ["购物", "商场", "免税"],
}

# 行程中不应出现的越界执行性表述（Agent 不应声称已代用户执行操作）
_ITINERARY_VIOLATION_PHRASES = [
    "已为你", "已帮你", "我帮你", "我为你",
    "代你", "自动完成", "正在执行", "正在预订", "正在支付",
    "已完成预订", "已完成支付", "已经预订",
]


def parse_itinerary_metadata(itinerary: str) -> dict[str, Any]:
    """从 Markdown 行程头部注释解析元信息。

    行程文件由 output.save_itinerary 生成，头部格式：
        <!--
        Travel Agent 生成行程
        生成时间: 2026-08-08 11:54:54
        会话 ID: eval_case_001
        用户输入: 我想去杭州玩2天，预算1000元，喜欢博物馆和美食
        目的地: 杭州
        天数: 2
        预算: 1000.0 元
        -->

    若文件无头部注释，返回空 dict（仅做通用结构评估）。
    """
    metadata: dict[str, Any] = {}

    start = itinerary.find("<!--")
    end = itinerary.find("-->")
    if start == -1 or end == -1 or end <= start:
        return metadata

    block = itinerary[start + 4:end]
    for line in block.strip().splitlines():
        line = line.strip()
        # 兼容中英文冒号
        sep = ":" if ":" in line else ("：" if "：" in line else None)
        if sep is None:
            continue
        key, _, value = line.partition(sep)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue

        if key == "目的地":
            metadata["destination"] = value
        elif key == "天数":
            try:
                metadata["days"] = int(float(value))
            except ValueError:
                pass
        elif key == "预算":
            num = value.replace("元", "").strip()
            try:
                metadata["budget"] = float(num)
            except ValueError:
                pass
        elif key == "用户输入":
            metadata["user_input"] = value
        elif key == "生成时间":
            metadata["generated_at"] = value
        elif key == "会话 ID":
            metadata["session_id"] = value

    return metadata


def _strip_itinerary_header(itinerary: str) -> str:
    """剥离 Markdown 行程头部的 HTML 注释块（元信息），只保留行程正文。

    行程文件由 output.save_itinerary 生成，头部是 `<!-- ... -->` 注释，包含
    目的地/天数/预算/用户输入等元信息。评估前必须剥离，否则"目的地在
    行程中""预算在行程中"等检查会匹配到注释里的元信息自身，造成假阳性。
    """
    cmt_start = itinerary.find("<!--")
    cmt_end = itinerary.find("-->")
    if cmt_start != -1 and cmt_end != -1 and cmt_end > cmt_start:
        return (itinerary[:cmt_start] + itinerary[cmt_end + 3:]).strip()
    return itinerary


def _has_schedule_section(itinerary: str) -> bool:
    """判断行程是否包含真实的日程分段（而非标题里出现的"2天"字样）。

    要求出现 "Day 1"/"day 1"/"第一天"/"第1天" 这类带编号的日程标记，
    避免仅凭标题中的"天"字误判。
    """
    patterns = [
        r"Day\s*\d",
        r"day\s*\d",
        r"DAY\s*\d",
        r"第[一二三四五六七八九十\d]+天",
        r"第[一二三四五六七八九十\d]+日",
    ]
    return any(re.search(p, itinerary) for p in patterns)


def _budget_split_count(itinerary: str) -> int:
    """统计行程中出现的预算科目关键词数量。"""
    budget_keywords = ["交通", "住宿", "餐饮", "门票", "其他", "预算", "费用", "花费", "总计", "合计"]
    return sum(1 for kw in budget_keywords if kw in itinerary)


def _check_structure(itinerary: str) -> tuple[dict[str, bool], dict[str, Any]]:
    """通用结构 + 合规性检查（行程正文，需已剥离头部注释）。

    返回 (checks, details)，供 EvalRunner._evaluate_case 与 evaluate_itinerary 共用，
    避免两套评估逻辑脱钩、阈值不一致。检查项：
      - 行程完整：>500 字符（模板输出均超 1000 字符）
      - 有标题结构：至少含二级标题 ##，而非单个 #
      - 有日程安排：出现 "Day 1"/"第一天" 这类带编号日程段
      - 有预算拆分：≥3 个预算科目关键词
      - 无越界操作：行程中无 "已为你预订" 等越界执行性表述
    """
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    checks["行程完整"] = len(itinerary) > 500
    checks["有标题结构"] = "##" in itinerary
    checks["有日程安排"] = _has_schedule_section(itinerary)

    budget_matches = _budget_split_count(itinerary)
    checks["有预算拆分"] = budget_matches >= 3
    details["budget_keywords_found"] = budget_matches

    violation = [p for p in _ITINERARY_VIOLATION_PHRASES if p in itinerary]
    checks["无越界操作"] = len(violation) == 0
    if violation:
        details["violation_phrases"] = violation

    return checks, details


def _extract_interests_from_input(user_input: str) -> list[str]:
    """从用户输入文本中提取兴趣关键词。"""
    interests: list[str] = []
    for interest, keywords in _INTEREST_KEYWORDS.items():
        if any(kw in user_input for kw in keywords):
            if interest not in interests:
                interests.append(interest)
    return interests


_CHINESE_DIGITS = "零一二三四五六七八九"


def _int_to_chinese(n: int) -> str:
    """整数转中文表达（简化版，支持 0-99999，覆盖常见预算/天数范围）。

    例：1000→"一千", 554→"五百五十四", 2→"二", 2000→"二千"（口语"两千"由调用方处理）。
    """
    if n == 0:
        return "零"
    units = ["", "十", "百", "千", "万"]
    s = str(n)
    length = len(s)
    parts: list[str] = []
    for i, ch in enumerate(s):
        digit = int(ch)
        pos = length - 1 - i
        if digit == 0:
            if parts and parts[-1] != "零":
                parts.append("零")
            continue
        parts.append(_CHINESE_DIGITS[digit] + units[pos])
    result = "".join(parts)
    if result.startswith("一十"):  # 12 → "十二" 而非 "一十二"
        result = result[1:]
    return result.rstrip("零")


def _chinese_number_variants(n: int) -> list[str]:
    """返回整数的中文表达变体（含"二/两"口语变体），用于宽松匹配。"""
    cn = _int_to_chinese(n)
    cn_alt = cn.replace("二", "两")
    return list(dict.fromkeys([cn, cn_alt]))  # 去重保序


def evaluate_itinerary(itinerary: str) -> dict[str, Any]:
    """对任意 Markdown 行程文档做质量评估（不依赖测试用例）。

    评估维度：
    1. 结构完整性：行程完整、标题结构、日程安排、预算拆分
    2. 约束满足：目的地、天数、预算是否在行程中体现（基于文件头部元信息）
    3. 偏好纳入：用户兴趣是否在行程中体现（从用户输入提取）
    4. 合规性：行程中无越界执行性表述

    纯规则评估，不调用 LLM，毫秒级返回。

    Returns:
        {
            "metadata": {...},
            "checks": {"行程完整": True, ...},
            "passed_count": 7,
            "total_count": 8,
            "pass_rate": "87.5%",
            "failed_checks": [...],
            "itinerary_length": 2665,
            "details": {...},
            "summary": "...",
        }
    """
    metadata = parse_itinerary_metadata(itinerary)

    # 剥离头部注释，只评估行程正文
    # （否则"目的地/预算在行程中"等检查会匹配到注释里的元信息自身，失去意义）
    itinerary = _strip_itinerary_header(itinerary)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {"itinerary_length": len(itinerary)}

    # ============================================================
    # 1. 结构完整性 + 合规性（与 EvalRunner._evaluate_case 共用 _check_structure，
    #    避免两套评估逻辑脱钩、阈值不一致）
    # ============================================================
    struct_checks, struct_details = _check_structure(itinerary)
    checks.update(struct_checks)
    details.update(struct_details)

    # ============================================================
    # 2. 约束满足（基于文件头部元信息）
    # ============================================================
    dest = metadata.get("destination")
    if dest:
        checks["目的地在行程中"] = dest in itinerary

    days = metadata.get("days")
    if days:
        cn_variants = _chinese_number_variants(days)
        day_patterns = [f"{days}天", f"{days}日", f"{days}晚", f"Day {days}", f"day {days}"]
        for cn in cn_variants:
            day_patterns += [f"{cn}天", f"{cn}日", f"{cn}晚", f"第{cn}天"]
        checks["天数在行程中"] = any(p in itinerary for p in day_patterns)

    budget = metadata.get("budget")
    if budget:
        budget_int = int(budget)
        cn_variants = _chinese_number_variants(budget_int)
        budget_patterns = [str(budget_int), f"{budget_int}元", f"{budget_int}块"]
        for cn in cn_variants:
            budget_patterns += [f"{cn}元", f"{cn}块"]
        checks["预算在行程中"] = any(p in itinerary for p in budget_patterns)

    # ============================================================
    # 3. 偏好纳入（从用户输入提取兴趣）
    # ============================================================
    user_input = metadata.get("user_input", "")
    if user_input:
        interests = _extract_interests_from_input(user_input)
        if interests:
            matched = 0
            for interest in interests:
                keywords = _INTEREST_KEYWORDS.get(interest, [interest])
                if any(kw in itinerary for kw in keywords):
                    matched += 1
            checks["兴趣在行程中"] = matched >= len(interests) * 0.5
            details["interests"] = interests
            details["interests_matched"] = f"{matched}/{len(interests)}"

    # ============================================================
    # 汇总（"无越界操作"已由 _check_structure 统一检查）
    # ============================================================
    total = len(checks)
    passed_count = sum(1 for v in checks.values() if v)
    failed_checks = [k for k, v in checks.items() if not v]

    if total == 0:
        summary = "⚠️ 未提取到元信息，且无法执行任何检查"
    elif not failed_checks:
        summary = f"✅ 行程质量评估全部通过（{passed_count}/{total}）"
    else:
        summary = f"行程质量评估通过 {passed_count}/{total}，未通过：{', '.join(failed_checks)}"

    return {
        "metadata": metadata,
        "checks": checks,
        "passed_count": passed_count,
        "total_count": total,
        "pass_rate": f"{passed_count / total * 100:.1f}%" if total > 0 else "N/A",
        "failed_checks": failed_checks,
        "itinerary_length": len(itinerary),
        "details": details,
        "summary": summary,
    }
