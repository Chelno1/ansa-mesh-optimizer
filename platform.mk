# Intelligent Simulation Platform Makefile
# 智能仿真平台管理命令

.PHONY: help platform-build platform-up platform-down platform-logs platform-clean platform-dev platform-test platform-restart

# 默认目标
help:
	@echo "智能仿真平台 - 可用命令："
	@echo "  make platform-build    - 构建所有Docker镜像"
	@echo "  make platform-up       - 启动所有服务"
	@echo "  make platform-down     - 停止所有服务"
	@echo "  make platform-restart  - 重启所有服务"
	@echo "  make platform-logs     - 查看服务日志"
	@echo "  make platform-clean    - 清理容器和卷"
	@echo "  make platform-test     - 运行测试"

# 构建镜像
platform-build:
	@echo "构建Docker镜像..."
	docker-compose build

# 启动服务
platform-up:
	@echo "启动服务..."
	docker-compose up -d
	@echo "等待服务就绪..."
	@sleep 5
	@echo ""
	@echo "服务已启动："
	@echo "  - 平台API:     http://localhost:8080"
	@echo "  - API文档:     http://localhost:8080/doc.html"
	@echo "  - Agent服务:   http://localhost:8081"
	@echo "  - MinIO控制台: http://localhost:9001"

# 停止服务
platform-down:
	@echo "停止服务..."
	docker-compose down

# 重启服务
platform-restart:
	@echo "重启服务..."
	docker-compose restart

# 查看日志
platform-logs:
	docker-compose logs -f

# 查看特定服务日志
platform-logs-api:
	docker-compose logs -f platform-api

platform-logs-agent:
	docker-compose logs -f agent-worker

# 清理
platform-clean:
	@echo "清理容器和数据卷..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# 运行测试
platform-test:
	@echo "运行平台测试..."
	@if [ -d "platform-api" ]; then \
		cd platform-api && ./mvnw test; \
	fi
	@if [ -d "agent-worker" ]; then \
		cd agent-worker && pytest tests/ -v; \
	fi

# 查看服务状态
platform-status:
	docker-compose ps

# 进入容器
platform-shell-api:
	docker exec -it sim-platform-api /bin/bash

platform-shell-agent:
	docker exec -it sim-agent-worker /bin/bash

# 查看数据库
platform-db:
	docker exec -it sim-postgres psql -U simuser -d simulation
