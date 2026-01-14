#!/bin/bash
set -e

echo "========================================"
echo "  智能仿真平台 - 停止脚本"
echo "  Intelligent Simulation Platform"
echo "========================================"

# 检查docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] docker-compose未安装"
    echo "[ERROR] docker-compose is not installed"
    exit 1
fi

echo "停止所有服务..."
docker-compose down

echo ""
echo "========================================"
echo "  服务已停止!"
echo "  Services Stopped!"
echo "========================================"
echo ""
echo "如需清理数据卷，运行: docker-compose down -v"
echo "To clean volumes, run: docker-compose down -v"
