# -*- coding: utf-8 -*-
"""完整 Eval 运行器：使用真实 LLM 路由器，清理代理环境变量。"""
import asyncio
import os
import sys

# 清理代理环境变量（避免干扰 API 调用，见 project_memory 教训）
for k in list(os.environ):
    if k.lower() in {"http_proxy", "https_proxy", "all_proxy", "http_proxies", "https_proxies"}:
        del os.environ[k]
        print(f"  已清理环境变量: {k}")

sys.path.insert(0, r"d:\AIProject\Project")

from app.config import build_llm_router
from app.api.routes.travel import _LLMWrapper
from app.core.travel.eval import run_eval


async def main():
    router = build_llm_router()
    if router is None:
        print("⚠️ 未构建 LLM 路由器，将以 llm=None（模板降级）模式运行 eval")
        llm = None
    else:
        llm = _LLMWrapper(router)
        print(f"✅ LLM 路由器已构建，将以真实 LLM 模式运行 eval")
    summary = await run_eval(llm=llm)
    print("\n\n========== 最终汇总 ==========")
    print(f"Total: {summary['total_cases']}, Passed: {summary['passed']}, "
          f"Failed: {summary['failed']}, Errors: {summary['errors']}, "
          f"Pass Rate: {summary['pass_rate']}")
    print("By type:")
    for t, stats in summary.get("by_type", {}).items():
        rate = f"{stats['passed'] / stats['total'] * 100:.0f}%" if stats["total"] > 0 else "N/A"
        print(f"  {t}: {stats['passed']}/{stats['total']} ({rate})")
    # 打印失败用例明细
    print("\n失败用例明细:")
    for d in summary.get("details", []):
        if not d["passed"]:
            err = f" (Error: {d['error']})" if d.get("error") else ""
            fc = d.get("details", {}).get("failed_checks", [])
            print(f"  ❌ {d['case_id']} ({d['type']}){err} 未通过: {', '.join(fc) if fc else 'N/A'}")


if __name__ == "__main__":
    asyncio.run(main())
