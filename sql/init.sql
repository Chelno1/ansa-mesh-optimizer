-- 智能仿真平台数据库初始化脚本
-- Intelligent Simulation Platform Database Initialization

-- 项目表
CREATE TABLE IF NOT EXISTS project (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    product VARCHAR(200),
    simulation_type VARCHAR(50),
    description TEXT,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 数据集表
CREATE TABLE IF NOT EXISTS dataset (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT REFERENCES project(id),
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50),
    version VARCHAR(20),
    file_path VARCHAR(500),
    file_hash VARCHAR(64),
    file_size BIGINT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 工作流模板表
CREATE TABLE IF NOT EXISTS workflow_template (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    steps JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 工作流实例表
CREATE TABLE IF NOT EXISTS workflow_instance (
    id BIGSERIAL PRIMARY KEY,
    template_id BIGINT REFERENCES workflow_template(id),
    project_id BIGINT REFERENCES project(id),
    status VARCHAR(20) DEFAULT 'PENDING',
    current_step INT DEFAULT 0,
    context JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 任务表
CREATE TABLE IF NOT EXISTS task (
    id BIGSERIAL PRIMARY KEY,
    workflow_instance_id BIGINT REFERENCES workflow_instance(id),
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50),
    step_index INT,
    status VARCHAR(20) DEFAULT 'PENDING',
    priority INT DEFAULT 5,
    params JSONB,
    result JSONB,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    agent_id VARCHAR(100),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 产物表
CREATE TABLE IF NOT EXISTS artifact (
    id BIGSERIAL PRIMARY KEY,
    task_id BIGINT REFERENCES task(id),
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50),
    file_path VARCHAR(500),
    file_hash VARCHAR(64),
    file_size BIGINT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 指标表
CREATE TABLE IF NOT EXISTS metric (
    id BIGSERIAL PRIMARY KEY,
    task_id BIGINT REFERENCES task(id),
    artifact_id BIGINT REFERENCES artifact(id),
    name VARCHAR(100) NOT NULL,
    value NUMERIC,
    unit VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户表
CREATE TABLE IF NOT EXISTS platform_user (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(200),
    role VARCHAR(50) DEFAULT 'USER',
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_dataset_project ON dataset(project_id);
CREATE INDEX IF NOT EXISTS idx_task_workflow ON task(workflow_instance_id);
CREATE INDEX IF NOT EXISTS idx_task_status ON task(status);
CREATE INDEX IF NOT EXISTS idx_artifact_task ON artifact(task_id);
CREATE INDEX IF NOT EXISTS idx_metric_task ON metric(task_id);

-- 插入初始数据（可选）
INSERT INTO platform_user (username, password_hash, email, role)
VALUES ('admin', '$2a$10$dummyHashForDevOnly', 'admin@example.com', 'ADMIN')
ON CONFLICT (username) DO NOTHING;

-- 插入示例工作流模板
INSERT INTO workflow_template (name, description, steps)
VALUES (
    'Basic Simulation Workflow',
    'A basic workflow for running simulations',
    '[
        {"name": "prepare", "type": "PREPARATION", "executor": "validator"},
        {"name": "simulate", "type": "SIMULATION", "executor": "simulation"},
        {"name": "analyze", "type": "ANALYSIS", "executor": "simulation"}
    ]'::jsonb
)
ON CONFLICT DO NOTHING;
