"""
健康检查API
Health Check API
"""
from datetime import datetime
from fastapi import APIRouter

from agent.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "UP",
        "service": "Agent Worker",
        "version": "1.0.0",
        "agent_id": settings.agent_id,
        "environment": settings.environment,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/ready")
async def readiness_check():
    """就绪检查"""
    return {
        "status": "READY",
        "agent_id": settings.agent_id
    }


@router.get("/health/live")
async def liveness_check():
    """存活检查"""
    return {
        "status": "ALIVE",
        "agent_id": settings.agent_id
    }
