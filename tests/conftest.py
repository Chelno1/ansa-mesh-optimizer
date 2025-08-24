#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest配置文件

作者: Chel
创建日期: 2025-07-07
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Generator

import importlib
import sys

import pytest

# Ensure the source package can be imported as ``ansa_mesh_optimizer``
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add the ``src`` directory to ``sys.path`` if it's not already there
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Alias the ``src`` package to ``ansa_mesh_optimizer`` so tests can import
# the package as if it were installed.
src_module = importlib.import_module("src")
sys.modules.setdefault("ansa_mesh_optimizer", src_module)


@pytest.fixture
def test_params() -> Dict[str, float]:
    """测试参数夹具"""
    return {"distortion_distance": 20}


@pytest.fixture
def temp_config_file() -> Path:
    """临时配置文件夹具"""
    config_data = {
        "optimization": {
            "n_calls": 10,
            "n_initial_points": 3,
            "optimizer": "genetic",
            "early_stopping": True,
            "patience": 3,
        },
        "ansa": {
            "min_element_length": 2.0,
            "max_element_length": 8.0,
            "quality_check_enabled": True,
        },
    }

    # 使用显式的文本模式和编码
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(config_data, f, indent=2)
        return Path(f.name)


@pytest.fixture
def mock_mesh_evaluator():
    """模拟网格评估器夹具"""
    from ansa_mesh_optimizer.evaluators.mesh_evaluator import create_mesh_evaluator

    return create_mesh_evaluator("mock")


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """临时工作空间夹具"""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # 创建必要的目录结构
        (workspace / "src").mkdir()
        (workspace / "tests").mkdir()
        (workspace / "output").mkdir()

        yield workspace


@pytest.fixture
def sample_optimization_history() -> list:
    """示例优化历史夹具"""
    return [
        {
            "params": {"distortion_distance": 20},
            "result": 100,
            "timestamp": "2025-07-07T12:00:00",
        },
        {
            "params": {"distortion_distance": 22},
            "result": 80,
            "timestamp": "2025-07-07T12:01:00",
        },
        {
            "params": {"distortion_distance": 25},
            "result": 50,
            "timestamp": "2025-07-07T12:02:00",
        },
    ]


def pytest_configure(config: pytest.Config) -> None:
    """Pytest配置函数"""
    # 添加自定义标记
    config.addinivalue_line("markers", "integration: 标记集成测试")
    config.addinivalue_line("markers", "slow: 标记耗时测试")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """修改收集的测试项"""
    for item in items:
        # 为集成测试添加标记
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # 为耗时测试添加标记
        if "test_optimization" in item.nodeid or "test_batch_mesh" in item.nodeid:
            item.add_marker(pytest.mark.slow)
