"""
Travel Agent 评估脚本。

运行 20 条 Eval Case，评估约束满足率、安全合规率等指标。

用法:
    python run_eval.py              # 运行全部 20 条
    python run_eval.py --type 安全边界  # 只运行安全边界类
    python run_eval.py --output results.json  # 保存结果到文件
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.api.routes import travel
from app.config import build_llm_router, get_settings
from app.core.travel.eval.evaluator import EvalRunner


async def main(args: argparse.Namespace) -> int:
    # 初始化 LLM
    get_settings.cache_clear()
    router = build_llm_router()
    if router is not None:
        travel.set_llm_router(router)
        llm = travel.get_llm_router()
        logger.info("LLM 已就绪: {}", get_settings().openai_model)
    else:
        llm = None
        logger.warning("未配置 LLM，将走规则兜底路径（评估结果可能不准确）")

    # 加载评估用例
    runner = EvalRunner()

    # 按类型过滤
    if args.type:
        runner._cases = [c for c in runner._cases if c.type == args.type]
        logger.info("过滤类型: {}，剩余 {} 条", args.type, len(runner._cases))

    logger.info("开始评估，共 {} 条用例（预计 {} 分钟）",
                len(runner._cases), len(runner._cases) * 1)

    # 运行评估
    start_time = datetime.now()
    summary = await runner.run(llm=llm, use_judge=not args.no_judge)
    elapsed = (datetime.now() - start_time).total_seconds()

    # 打印报告
    runner.print_report(summary)
    print(f"\n  耗时: {elapsed:.0f} 秒 ({elapsed / 60:.1f} 分钟)")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存: {output_path}")

    # 返回退出码：通过率 >= 80% 视为成功
    pass_rate = summary["passed"] / summary["total_cases"] if summary["total_cases"] > 0 else 0
    return 0 if pass_rate >= 0.8 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Travel Agent 评估脚本")
    parser.add_argument("--type", type=str, default=None,
                        help="只运行指定类型的用例（常规规划/预算约束/偏好约束/变化处理/安全边界）")
    parser.add_argument("--no-judge", action="store_true",
                        help="跳过 LLM-as-judge 评分（只跑规则检查，速度快）")
    parser.add_argument("--output", type=str, default=None,
                        help="评估结果保存路径（JSON）")
    args = parser.parse_args()

    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
