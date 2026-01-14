# 智能仿真平台 (Intelligent Simulation Platform)

智能仿真平台是一个面向仿真全过程、具备AI能力的企业级仿真管理系统。

## 项目概述

本平台提供了完整的仿真项目管理、数据管理、任务编排和执行能力，采用微服务架构设计。

### 技术架构

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| **平台主服务** | Java 17 + Spring Boot 3.2 | 项目/数据/流程元数据、任务编排、API、权限 |
| **Agent Worker** | Python 3.11 + FastAPI | 执行仿真任务、文件处理、结果回传 |
| **数据库** | PostgreSQL 15 | 元数据存储 |
| **缓存/队列** | Redis 7 | 任务队列、缓存、状态管理 |
| **对象存储** | MinIO | 仿真输入/输出文件存储 |
| **部署方式** | Docker Compose | Demo阶段轻量部署 |

## 快速开始

### 前置条件

- Docker 20.10+
- Docker Compose 2.0+
- Git

### 一键启动

```bash
# 克隆仓库（如果还未克隆）
git clone <repository-url>
cd ansa-mesh-optimizer

# 启动平台服务
./scripts/run.sh

# 或使用 make 命令
make platform-up
```

### 验证服务

启动完成后，访问以下地址验证服务：

| 服务 | 地址 | 说明 |
|------|------|------|
| 平台API | http://localhost:8080/api/health | 健康检查 |
| API文档 | http://localhost:8080/doc.html | Knife4j API文档 |
| Agent服务 | http://localhost:8081/api/health | Agent健康检查 |
| MinIO控制台 | http://localhost:9001 | 对象存储管理 (minioadmin/minioadmin123) |
| PostgreSQL | localhost:5432 | 数据库 (simuser/simpass123) |
| Redis | localhost:6379 | 缓存和队列 |

## 项目结构

```
.
├── platform-api/              # Java主服务
│   ├── src/
│   │   └── main/
│   │       ├── java/com/sim/platform/
│   │       │   ├── PlatformApplication.java
│   │       │   ├── config/         # 配置类
│   │       │   ├── controller/     # 控制器
│   │       │   ├── model/          # 实体和DTO
│   │       │   └── common/         # 通用类
│   │       └── resources/
│   │           └── application.yml
│   ├── pom.xml
│   └── Dockerfile
├── agent-worker/              # Python Agent服务
│   ├── agent/
│   │   ├── main.py            # FastAPI入口
│   │   ├── config.py          # 配置管理
│   │   ├── api/               # API路由
│   │   └── executor/          # 任务执行器
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── Dockerfile
├── sql/
│   └── init.sql               # 数据库初始化脚本
├── scripts/
│   ├── run.sh                 # 启动脚本
│   └── stop.sh                # 停止脚本
├── docker-compose.yml         # Docker编排配置
├── platform.mk                # 平台管理命令
└── PLATFORM_README.md         # 本文档
```

## 开发指南

### Java平台服务开发

```bash
cd platform-api

# 使用Maven构建
./mvnw clean package

# 本地运行（需要PostgreSQL和Redis）
./mvnw spring-boot:run
```

### Python Agent开发

```bash
cd agent-worker

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 本地运行
uvicorn agent.main:app --reload --port 8081
```

## 常用命令

使用平台管理命令（通过 [`platform.mk`](platform.mk:1)）：

```bash
# 查看可用命令
make help

# 构建镜像
make platform-build

# 启动服务
make platform-up

# 停止服务
make platform-down

# 重启服务
make platform-restart

# 查看日志
make platform-logs

# 查看特定服务日志
make platform-logs-api
make platform-logs-agent

# 清理环境（包括数据卷）
make platform-clean

# 查看服务状态
make platform-status

# 进入容器
make platform-shell-api
make platform-shell-agent

# 连接数据库
make platform-db
```

## 数据库结构

平台使用PostgreSQL存储元数据，主要表包括：

- **project** - 项目信息
- **dataset** - 数据集
- **workflow_template** - 工作流模板
- **workflow_instance** - 工作流实例
- **task** - 任务
- **artifact** - 产物
- **metric** - 指标
- **platform_user** - 用户

详细表结构请参考 [`sql/init.sql`](sql/init.sql:1)

## API文档

启动服务后，访问 http://localhost:8080/doc.html 查看完整的API文档。

主要API端点：

### 平台API (端口 8080)
- `GET /api/health` - 健康检查
- `GET /actuator/health` - Spring Actuator健康检查

### Agent API (端口 8081)
- `GET /api/health` - 健康检查
- `GET /api/health/ready` - 就绪检查
- `GET /api/health/live` - 存活检查

## 架构特性

### 1. 微服务架构
- 平台服务（Java）负责业务逻辑和元数据管理
- Agent Worker（Python）负责任务执行
- 服务间通过REST API和Redis队列通信

### 2. 异步任务处理
- 使用Redis Streams实现任务队列
- Agent Worker异步消费和执行任务
- 支持任务重试和错误处理

### 3. 对象存储
- MinIO存储仿真文件
- 支持大文件上传下载
- 文件版本管理

### 4. 扩展性设计
- 执行器模式，支持多种仿真类型
- 工作流引擎，支持复杂流程编排
- 水平扩展Agent Worker节点

## 配置说明

### 环境变量

**平台API服务：**
- `SPRING_PROFILES_ACTIVE` - 运行环境 (dev/prod)
- `SPRING_DATASOURCE_URL` - 数据库连接
- `SPRING_REDIS_HOST` - Redis主机
- `MINIO_ENDPOINT` - MinIO端点

**Agent Worker服务：**
- `PLATFORM_API_URL` - 平台API地址
- `REDIS_URL` - Redis连接URL
- `MINIO_ENDPOINT` - MinIO端点
- `LOG_LEVEL` - 日志级别

详细配置参考：
- 平台API: [`platform-api/src/main/resources/application.yml`](platform-api/src/main/resources/application.yml:1)
- Agent: [`agent-worker/agent/config.py`](agent-worker/agent/config.py:1)

## 故障排查

### 服务无法启动

1. 检查Docker是否运行：
   ```bash
   docker info
   ```

2. 查看服务日志：
   ```bash
   docker-compose logs -f
   ```

3. 检查端口占用：
   ```bash
   # 检查8080, 8081, 5432, 6379, 9000端口
   netstat -tuln | grep -E '8080|8081|5432|6379|9000'
   ```

### 数据库连接失败

确保PostgreSQL服务已启动并健康：
```bash
docker-compose ps postgres
docker-compose logs postgres
```

### MinIO连接问题

检查MinIO服务状态并测试连接：
```bash
curl http://localhost:9000/minio/health/live
```

## 扩展开发

### 添加新的执行器

1. 在 `agent-worker/agent/executor/` 创建新的执行器类
2. 继承 [`BaseExecutor`](agent-worker/agent/executor/base.py:10)
3. 实现 `execute()` 和 `validate()` 方法

示例：
```python
from agent.executor.base import BaseExecutor

class CustomExecutor(BaseExecutor):
    async def validate(self) -> bool:
        # 参数验证逻辑
        return True
    
    async def execute(self) -> Dict[str, Any]:
        # 执行逻辑
        return {"status": "SUCCESS"}
```

### 添加新的API端点

**Java平台服务：**
1. 在 `platform-api/src/main/java/com/sim/platform/controller/` 创建控制器
2. 使用 `@RestController` 和 `@RequestMapping` 注解

**Python Agent服务：**
1. 在 `agent-worker/agent/api/` 创建路由模块
2. 在 [`main.py`](agent-worker/agent/main.py:27) 中注册路由

## 性能优化建议

1. **数据库优化**
   - 为常用查询字段添加索引
   - 使用连接池管理
   - 定期分析慢查询

2. **缓存策略**
   - 使用Redis缓存热点数据
   - 设置合理的过期时间
   - 实现缓存预热

3. **Agent扩展**
   - 增加Agent Worker实例数量
   - 使用负载均衡
   - 实现任务优先级队列

## 安全建议

⚠️ **生产环境部署前必须修改：**

1. 修改默认密码：
   - PostgreSQL: simuser/simpass123
   - MinIO: minioadmin/minioadmin123
   - Spring Security: admin/admin123

2. 启用HTTPS
3. 配置防火墙规则
4. 实施访问控制
5. 定期备份数据

## 版本信息

- **当前版本**: 1.0.0-SNAPSHOT
- **发布日期**: 2026-01-14
- **兼容性**: Demo版本

## 许可证

MIT License

## 支持与反馈

如有问题或建议，请提交Issue或联系开发团队。

---

**注意**: 这是Demo版本，仅用于开发和测试。生产环境部署需要额外的安全加固和性能优化。
