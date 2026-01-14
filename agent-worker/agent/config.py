"""
Agent Worker配置
Agent Worker Configuration
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""
    
    # 环境配置
    environment: str = "development"
    log_level: str = "INFO"
    
    # 平台API配置
    platform_api_url: str = "http://localhost:8080"
    
    # Redis配置
    redis_url: str = "redis://localhost:6379/0"
    
    # MinIO配置
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket: str = "simulation-files"
    minio_secure: bool = False
    
    # Agent配置
    agent_id: str = "agent-worker-1"
    workspace_dir: str = "/app/workspace"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# 全局配置实例
settings = Settings()
