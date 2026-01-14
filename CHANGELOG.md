# 更新日志

记录项目的重要变更。

## [1.0.0] - 2026-01-14
### 新增
- 🚀 智能仿真平台初始化
  - Java Spring Boot 3.2 平台API服务 (platform-api/)
  - Python FastAPI Agent Worker服务 (agent-worker/)
  - PostgreSQL数据库初始化脚本和完整表结构
  - Redis任务队列和缓存支持
  - MinIO对象存储集成
  - Docker Compose一键部署配置
  - 平台管理命令 (platform.mk)
  - 完整的API文档 (Knife4j/Swagger)
  - 健康检查和监控端点
  - 基础执行器框架
  - 一键启动脚本 (scripts/run.sh, scripts/stop.sh)
  - 平台使用文档 (PLATFORM_README.md)

### 架构
- 微服务架构：平台服务 + Agent Worker
- 异步任务处理：Redis Streams
- 对象存储：MinIO文件管理
- 数据库：PostgreSQL元数据存储
- API文档：Knife4j/Swagger集成

## [0.2.0] - 2025-01-01
### 新增
- 初始公开版本，提供多种优化算法（贝叶斯、随机、随机森林、遗传）、配置系统、缓存机制和命令行工具。

