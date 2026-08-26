"""外部 HTTP 工具的共享客户端工厂。

默认 trust_env=False 直连：高德/Open-Meteo 等外部 API 走系统代理时
容易被本机代理软件（Clash 等）拦截导致 ConnectTimeout/ReadTimeout。
需要代理出网的部署可通过 HTTPX_TRUST_ENV=true 打开。
"""

from __future__ import annotations

import httpx

from app.config import get_settings


def make_client(timeout: float) -> httpx.AsyncClient:
    """构建工具层统一配置的异步 HTTP 客户端。"""
    return httpx.AsyncClient(timeout=timeout, trust_env=get_settings().httpx_trust_env)
