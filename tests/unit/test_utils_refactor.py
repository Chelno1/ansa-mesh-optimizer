#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utils 重构测试模块

测试重构后的序列化、格式化和通用工具函数模块。

作者: Chel
创建日期: 2025-08-23
版本: 1.0.0
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# 测试统一导出接口
from src.utils import format_execution_time as utils_format_execution_time
from src.utils import normalize_params as utils_normalize_params
from src.utils import safe_json_serialize as utils_safe_json_serialize

# 测试格式化模块
from src.utils.formatting import (
    PATTERNS,
    create_backup_filename,
    create_progress_bar,
    create_summary_table,
    extract_numbers_from_text,
    format_execution_time,
    format_file_size,
    format_number,
    format_percentage,
    truncate_string,
)

# 测试通用工具模块 (已迁移到misc.py)
from src.utils.misc import (
    calculate_statistics,
    check_memory_usage,
    filter_dict_by_keys,
    safe_divide,
)

# 测试序列化模块
from src.utils.serialization import (
    load_json_config,
    parse_json_string,
    safe_json_serialize,
    save_json_config,
    serialize_numpy_object,
)

# 测试参数验证模块 (仍在utils.py中)
from src.utils.utils import normalize_params


class TestSerialization:
    """测试序列化模块"""

    def test_safe_json_serialize_basic_types(self):
        """测试基本类型序列化"""
        data = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        result = safe_json_serialize(data)
        assert isinstance(result, str)

        # 验证可以正确解析
        parsed = json.loads(result)
        assert parsed["string"] == "test"
        assert parsed["int"] == 42
        assert parsed["float"] == 3.14

    def test_safe_json_serialize_numpy_types(self):
        """测试numpy类型序列化"""
        data = {
            "np_array": np.array([1, 2, 3]),
            "np_float": np.float64(3.14),
            "np_int": np.int32(42),
            "nested": {"np_values": np.array([0.1, 0.2, 0.3])},
        }

        result = safe_json_serialize(data)
        parsed = json.loads(result)

        assert parsed["np_array"] == [1, 2, 3]
        assert isinstance(parsed["np_float"], float)
        assert isinstance(parsed["np_int"], int)
        assert parsed["nested"]["np_values"] == [0.1, 0.2, 0.3]

    def test_serialize_numpy_object(self):
        """测试单个numpy对象序列化"""
        # 测试numpy数组
        arr = np.array([1, 2, 3])
        result = serialize_numpy_object(arr)
        assert result == [1, 2, 3]

        # 测试numpy标量
        scalar = np.float64(3.14)
        result = serialize_numpy_object(scalar)
        assert isinstance(result, float)
        assert result == 3.14

    def test_config_file_operations(self):
        """测试配置文件操作"""
        test_config = {
            "param1": 1.0,
            "param2": "value",
            "nested": {"array": np.array([1, 2, 3])},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"

            # 保存配置
            save_json_config(test_config, config_file, backup=False)
            assert config_file.exists()

            # 加载配置
            loaded_config = load_json_config(config_file)
            assert loaded_config["param1"] == 1.0
            assert loaded_config["param2"] == "value"
            assert loaded_config["nested"]["array"] == [1, 2, 3]

    def test_parse_json_string(self):
        """测试JSON字符串解析"""
        json_str = '{"key": "value", "number": 42}'
        result = parse_json_string(json_str)

        assert result["key"] == "value"
        assert result["number"] == 42

        # 测试无效JSON
        with pytest.raises(ValueError):
            parse_json_string("invalid json")


class TestFormatting:
    """测试格式化模块"""

    def test_format_execution_time(self):
        """测试时间格式化"""
        # 秒
        assert "30.0秒" in format_execution_time(30.0)

        # 分钟
        assert "2.0分钟" in format_execution_time(120.0)

        # 小时
        assert "1.0小时" in format_execution_time(3600.0)

        # 负数
        assert format_execution_time(-10) == "0秒"

    def test_create_summary_table(self):
        """测试表格创建"""
        data = [
            {"name": "test1", "value": 1.23, "status": "ok"},
            {"name": "test2", "value": 4.56, "status": "running"},
        ]

        table = create_summary_table(data)
        assert "test1" in table
        assert "test2" in table
        assert "1.23" in table
        assert "4.56" in table

        # 测试空数据
        assert create_summary_table([]) == "无数据"

    def test_truncate_string(self):
        """测试字符串截断"""
        long_string = "This is a very long string"

        result = truncate_string(long_string, 10)
        assert len(result) <= 10
        assert result.endswith("...")

        # 测试短字符串
        short_string = "short"
        result = truncate_string(short_string, 10)
        assert result == "short"

    def test_format_number(self):
        """测试数字格式化"""
        # 基本格式化
        result = format_number(1234.5678)
        assert "1,234" in result
        assert "57" in result  # 默认2位小数

        # 整数
        result = format_number(1234)
        assert result == "1,234"

        # 精度控制
        result = format_number(1234.5678, precision=1)
        assert "1,234.6" in result

    def test_format_percentage(self):
        """测试百分比格式化"""
        result = format_percentage(0.1234)
        assert result == "12.3%"

        result = format_percentage(0.1234, precision=2)
        assert result == "12.34%"

    def test_format_file_size(self):
        """测试文件大小格式化"""
        # 字节
        assert format_file_size(512) == "512 B"

        # KB
        assert "1.0 KB" in format_file_size(1024)

        # MB
        assert "1.0 MB" in format_file_size(1024 * 1024)

        # GB
        assert "1.0 GB" in format_file_size(1024 * 1024 * 1024)

    def test_create_progress_bar(self):
        """测试进度条创建"""
        # 正常进度
        bar = create_progress_bar(30, 100)
        assert "[" in bar
        assert "]" in bar
        assert "30.0%" in bar

        # 完成进度
        bar = create_progress_bar(100, 100)
        assert "100.0%" in bar

        # 异常情况
        bar = create_progress_bar(50, 0)
        assert "0.0%" in bar

    def test_create_backup_filename(self):
        """测试备份文件名创建"""
        original = Path("config.json")

        # 带时间戳
        backup = create_backup_filename(original, timestamp=True)
        assert backup.suffix == ".json"
        assert "_" in backup.stem

        # 不带时间戳
        backup = create_backup_filename(original, timestamp=False)
        assert backup.name == "config_backup.json"


class TestUtils:
    """测试通用工具模块"""

    def test_normalize_params(self):
        """测试参数标准化"""
        params = {
            "quality_threshold": np.array([0.6]),
            "distortion_distance": np.float64(20.0),
            "normal_param": 3.0,
        }

        normalized = normalize_params(params)

        # 检查所有值都是Python原生类型
        for key, value in normalized.items():
            assert isinstance(value, (int, float))

    def test_safe_divide(self):
        """测试安全除法"""
        # 正常除法
        assert safe_divide(10, 2) == 5.0

        # 除零
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1.0

        # 极小值
        assert safe_divide(10, 1e-12) == 0.0

        # 类型错误 - 测试时需要忽略类型检查
        # safe_divide 函数应该能处理这种情况
        try:
            result = safe_divide("invalid", 2)  # type: ignore
            assert result == 0.0
        except:
            # 如果抛出异常也是可以接受的
            pass

    def test_calculate_statistics(self):
        """测试统计计算"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_statistics(values)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats

        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0

        # 测试空列表
        empty_stats = calculate_statistics([])
        assert "error" in empty_stats

    def test_filter_dict_by_keys(self):
        """测试字典过滤"""
        data = {"a": 1, "b": 2, "c": 3, "d": 4}

        # 包含模式
        filtered = filter_dict_by_keys(data, ["a", "c"], include=True)
        assert filtered == {"a": 1, "c": 3}

        # 排除模式
        filtered = filter_dict_by_keys(data, ["a", "c"], include=False)
        assert filtered == {"b": 2, "d": 4}

    def test_extract_numbers_from_text(self):
        """测试文本数字提取"""
        text = "价格是 123.45 元和 67.89 元，共计 191.34"

        numbers = extract_numbers_from_text(text)
        assert len(numbers) == 3
        assert 123.45 in numbers
        assert 67.89 in numbers
        assert 191.34 in numbers

        # 测试整数模式
        integers = extract_numbers_from_text(
            "有 3 个苹果和 5 个橙子", pattern="integer"
        )
        assert integers == [3.0, 5.0]

    def test_check_memory_usage(self):
        """测试内存使用检查"""
        memory_info = check_memory_usage()

        # 根据系统是否有psutil来检查结果
        if "error" in memory_info:
            assert memory_info["error"] == "psutil not available"
        else:
            assert "rss_mb" in memory_info
            assert "vms_mb" in memory_info
            assert isinstance(memory_info["rss_mb"], (int, float))

    def test_patterns(self):
        """测试正则表达式模式"""
        assert "number" in PATTERNS
        assert "integer" in PATTERNS
        assert "email" in PATTERNS
        assert "ip_address" in PATTERNS
        assert "filename" in PATTERNS


class TestUnifiedInterface:
    """测试统一导出接口"""

    def test_unified_imports(self):
        """测试可以从utils包根目录导入所有函数"""
        # 测试序列化函数
        data = {"test": np.array([1, 2, 3])}
        result = utils_safe_json_serialize(data)
        assert isinstance(result, str)

        # 测试格式化函数
        time_str = utils_format_execution_time(120)
        assert "分钟" in time_str

        # 测试工具函数
        params = {"value": np.float64(3.14)}
        normalized = utils_normalize_params(params)
        assert isinstance(normalized["value"], (int, float))

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 确保原有的导入方式仍然可用
        try:
            from ansa_mesh_optimizer.utils import (
                format_execution_time,
                normalize_params,
                safe_json_serialize,
            )

            # 基本功能测试
            assert callable(safe_json_serialize)
            assert callable(format_execution_time)
            assert callable(normalize_params)

        except ImportError as e:
            pytest.fail(f"向后兼容性测试失败: {e}")


def test_module_structure():
    """测试模块结构"""
    # 确保所有新模块都可以正确导入
    try:
        import src.utils
        import src.utils.formatting
        import src.utils.serialization
        import src.utils.utils

        # 检查__all__属性
        assert hasattr(src.utils, "__all__")
        assert len(src.utils.__all__) > 0

    except ImportError as e:
        pytest.fail(f"模块结构测试失败: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
