# -*- coding: utf-8 -*-
"""行程评估 API 路由：上传/传入 Markdown 行程，即时评估质量。

评估与生成分离：
- /travel/plan        生成行程（调用 LLM，耗时约 1 分钟）
- /eval/itinerary     评估行程文本（纯规则，毫秒级返回，不调用 LLM）
- /eval/itinerary/upload  上传 .md 文件评估
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from app.core.travel.eval.evaluator import evaluate_itinerary
from app.models.travel_schemas import EvalItineraryRequest, EvalItineraryResponse

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/itinerary", response_model=EvalItineraryResponse)
async def eval_itinerary(request: EvalItineraryRequest):
    """评估 Markdown 行程文本（JSON 模式）。

    传入 Markdown 行程文本（含头部元信息注释时可做约束满足检查）。
    纯规则评估，不调用 LLM，毫秒级返回。
    """
    markdown = request.markdown.strip()
    if not markdown:
        raise HTTPException(status_code=422, detail="行程文本不能为空")

    logger.info("收到行程评估请求（JSON），长度: {}", len(markdown))
    result = evaluate_itinerary(markdown)
    logger.info(
        "评估完成: {}/{} 通过 ({})",
        result["passed_count"], result["total_count"], result["pass_rate"],
    )
    return EvalItineraryResponse(**result)


@router.post("/itinerary/upload", response_model=EvalItineraryResponse)
async def eval_itinerary_upload(file: UploadFile = File(...)):
    """上传 Markdown 行程文件并评估。

    支持 .md / .markdown / .txt 文件（UTF-8 或 GBK 编码）。
    文件头部应包含元信息注释（由 /travel/plan 接口自动生成），
    评估接口会解析元信息做约束满足检查。
    """
    if not file.filename:
        raise HTTPException(status_code=422, detail="未提供文件")

    suffix = (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "").lower()
    if suffix not in ("md", "markdown", "txt"):
        raise HTTPException(
            status_code=422,
            detail="仅支持 .md / .markdown / .txt 文件",
        )

    content = await file.read()
    try:
        markdown = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            markdown = content.decode("gbk")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=422,
                detail="文件编码不支持，请使用 UTF-8 编码",
            )

    logger.info("收到行程评估上传: filename={}, 长度={}", file.filename, len(markdown))
    result = evaluate_itinerary(markdown)
    result["details"]["filename"] = file.filename
    logger.info(
        "评估完成: filename={}, {}/{} 通过 ({})",
        file.filename, result["passed_count"], result["total_count"], result["pass_rate"],
    )
    return EvalItineraryResponse(**result)
