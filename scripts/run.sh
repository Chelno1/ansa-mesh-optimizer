#!/bin/bash
set -e

echo "========================================"
echo "  智能仿真平台 - 启动脚本"
echo "  Intelligent Simulation Platform"
echo "========================================"

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker未运行，请先启动Docker"
    echo "[ERROR] Docker is not running, please start Docker first"
    exit 1
fi

# 检查docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] docker-compose未安装"
    echo "[ERROR] docker-compose is not installed"
    exit 1
fi

echo "[1/4] 创建必要目录..."
mkdir -p data/postgres data/redis data/minio

echo "[2/4] 构建Docker镜像..."
docker-compose build

echo "[3/4] 启动基础服务..."
docker-compose up -d postgres redis minio

echo "[4/4] 等待服务就绪后启动应用..."
sleep 10
docker-compose up -d platform-api agent-worker

echo ""
echo "========================================"
echo "  启动完成!"
echo "  Startup Complete!"
echo "========================================"
echo ""
echo "服务地址 (Service URLs):"
echo "  - 平台API:     http://localhost:8080"
echo "  - API文档:     http://localhost:8080/doc.html"
echo "  - Agent:       http://localhost:8081"
echo "  - MinIO控制台: http://localhost:9001"
echo "    (用户名/密码: minioadmin/minioadmin123)"
echo ""
echo "查看日志: docker-compose logs -f"
echo "停止服务: ./scripts/stop.sh 或 make platform-down"
