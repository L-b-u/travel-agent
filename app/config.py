"""应用配置：基于 pydantic-settings，支持环境变量与 .env 文件。"""

from functools import lru_cache

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="travel-agent", description="服务名称")
    app_env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    # ---- LLM 配置 ----
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI 兼容 API Base",
    )
    openai_model: str = Field(default="gpt-4o-mini", description="默认对话模型")

    # ---- Travel Agent 配置 ----
    amap_api_key: str = Field(
        default="",
        description="高德地图 API Key（免费获取：https://lbs.amap.com/）",
    )
    travel_weather_enabled: bool = Field(
        default=True,
        description="是否启用天气查询",
    )

    log_level: str = Field(default="INFO", description="日志级别")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def build_llm_router():
    """
    根据 Settings 构造 ModelRouter 实例。

    Returns:
        ModelRouter 实例；若未配置 API Key 则返回 None（节点将走规则兜底）。
    """
    settings = get_settings()

    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your-"):
        logger.warning(
            "未配置有效的 OPENAI_API_KEY，LLM 节点将走规则兜底路径。"
            "请在 .env 中设置 OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL。"
        )
        return None

    # 延迟导入，避免无 LLM 时也要装 openai
    from app.infrastructure.llm import ModelConfig, ModelRouter

    primary = ModelConfig(
        model_id=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
    )

    logger.info("LLM 路由器已构建，主模型: {}", settings.openai_model)
    return ModelRouter([primary])