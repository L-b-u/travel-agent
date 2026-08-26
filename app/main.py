"""Travel Agent — FastAPI 应用入口。"""

import os
from datetime import datetime

from fastapi import FastAPI
from loguru import logger

from app.api.routes import eval as eval_routes
from app.api.routes import travel
from app.config import build_llm_router, get_settings


def setup_logging() -> None:
    """配置 loguru 日志：控制台 + 文件（每次运行创建一个日志文件）。"""
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/travel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )
    logger.info("日志文件: {}", log_file)


def create_app() -> FastAPI:
    setup_logging()

    settings = get_settings()
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)

    application = FastAPI(
        title="Travel Agent",
        description=(
            "LLM Agent 工程化实践：LangGraph 状态图编排 + ReAct Tool Calling "
            "+ RAG 攻略检索 + HITL 安全确认 + Eval 评估体系"
        ),
        version="0.2.0",
        debug=settings.debug,
    )
    application.include_router(travel.router, prefix=settings.api_prefix)
    application.include_router(eval_routes.router, prefix=settings.api_prefix)

    # 注入 LLM 路由器：若未配置 API Key 则返回 None，节点走规则兜底
    llm_router = build_llm_router()
    if llm_router is not None:
        travel.set_llm_router(llm_router)
        logger.info("LLM 路由器已注入 Travel Agent 路由")
    else:
        logger.warning("LLM 路由器未注入，所有 LLM 节点将走规则兜底路径")

    # 预热 RAG 索引（jieba 词典加载 + BM25 构建），避免首个请求承担 ~7s 冷启动
    from app.core.travel.rag import get_retriever

    logger.info("RAG 预热完成: {}", get_retriever().status)

    return application


app = create_app()
