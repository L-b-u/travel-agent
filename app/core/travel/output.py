# -*- coding: utf-8 -*-
"""
行程输出工具：将 Travel Agent 生成的行程保存为 Markdown 文件。

文件保存到 output/ 目录，命名格式: {目的地}{天数}天行程_{时间戳}.md
例: 杭州2天行程_20260808_115454.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

# 输出目录
OUTPUT_DIR = Path("output")


def save_itinerary(
    itinerary: str,
    session_id: str = "default",
    user_input: str = "",
    preferences: Optional[Dict[str, Any]] = None,
) -> str:
    """
    将行程保存为 Markdown 文件。

    Args:
        itinerary: 行程 Markdown 文本
        session_id: 会话 ID
        user_input: 用户原始输入（写入文件头部注释）
        preferences: 偏好信息（写入文件头部注释）

    Returns:
        保存的文件路径
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 构建友好的中文文件名：{目的地}{天数}天行程_{时间戳}.md
    prefs = preferences or {}
    destination = str(prefs.get("destination", "")).strip()
    days = prefs.get("days", 0)

    # 清理目的地：保留中文/字母/数字，去除空格和特殊符号
    safe_dest = "".join(
        c for c in destination
        if c.isalnum() or "\u4e00" <= c <= "\u9fff"
    ) or "旅行"

    prefix = f"{safe_dest}{days}天行程" if days else f"{safe_dest}行程"
    filename = f"{prefix}_{timestamp}.md"
    filepath = OUTPUT_DIR / filename

    # 构建完整 Markdown 文件（含元信息注释）
    meta_lines = [
        f"<!--",
        f"Travel Agent 生成行程",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"会话 ID: {session_id}",
        f"用户输入: {user_input}",
    ]
    if preferences:
        dest = preferences.get("destination", "未知")
        days = preferences.get("days", 0)
        budget = preferences.get("budget", 0)
        meta_lines.append(f"目的地: {dest}")
        meta_lines.append(f"天数: {days}")
        meta_lines.append(f"预算: {budget} 元")
    meta_lines.append(f"-->")

    content = "\n".join(meta_lines) + "\n\n" + itinerary

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("行程已保存: {}", filepath)
    return str(filepath)


def load_itinerary(filepath: str) -> str:
    """
    从 Markdown 文件加载行程内容。

    Args:
        filepath: 文件路径

    Returns:
        行程 Markdown 文本（不含元信息注释）
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 移除头部元信息注释
    if content.startswith("<!--"):
        end = content.find("-->")
        if end != -1:
            content = content[end + 3:].strip()

    return content


def list_itineraries() -> list:
    """列出 output 目录下所有行程文件。"""
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
