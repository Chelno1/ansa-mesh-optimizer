"""
智能仿真平台 - Agent Worker主应用
Intelligent Simulation Platform - Agent Worker Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.api import health
from agent.config import settings

app = FastAPI(
    title="Agent Worker",
    description="Intelligent Simulation Platform Agent Worker",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, prefix="/api", tags=["health"])


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    print("=" * 50)
    print("  Agent Worker 启动成功!")
    print("  Agent Worker Started Successfully!")
    print(f"  Environment: {settings.environment}")
    print(f"  Platform API: {settings.platform_api_url}")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    print("Agent Worker shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
