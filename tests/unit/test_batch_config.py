#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa批处理配置模块的单元测试

作者: Chel
创建日期: 2025-08-24
版本: 2.0.0
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.batch.config import AnsaBatchConfig, create_default_config, load_config_from_file


class TestAnsaBatchConfig(unittest.TestCase):
    """AnsaBatchConfig类的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = AnsaBatchConfig(self.temp_dir)

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_initialization(self):
        """测试默认初始化"""
        self.assertEqual(self.config.min_length, 5.0)
        self.assertEqual(self.config.max_length, 15.0)
        self.assertEqual(self.config.timeout, 300)
        self.assertEqual(self.config.retry_attempts, 3)
        self.assertEqual(self.config.qual_file, "8mm_v23.ansa_qual")
        self.assertTrue(isinstance(self.config.quality_checks, dict))

    def test_find_mpar_file(self):
        """测试mpar文件查找"""
        # 创建一个测试mpar文件
        test_mpar = self.temp_dir / "test.ansa_mpar"
        test_mpar.touch()
        
        config = AnsaBatchConfig(self.temp_dir)
        self.assertEqual(config.mpar_file, "test.ansa_mpar")

    def test_find_mpar_file_default(self):
        """测试mpar文件查找失败时的默认值"""
        # 确保目录中没有mpar文件
        self.assertEqual(self.config.mpar_file, "mend.ansa_mpar")

    def test_load_from_file_existing(self):
        """测试从现有文件加载配置"""
        config_file = self.temp_dir / "test_config.json"
        test_config = {
            "min_length": 3.0,
            "max_length": 20.0,
            "timeout": 600,
            "retry_attempts": 5
        }
        
        with open(config_file, "w") as f:
            json.dump(test_config, f)
        
        self.config.load_from_file(config_file)
        
        self.assertEqual(self.config.min_length, 3.0)
        self.assertEqual(self.config.max_length, 20.0)
        self.assertEqual(self.config.timeout, 600)
        self.assertEqual(self.config.retry_attempts, 5)

    def test_load_from_file_nonexistent(self):
        """测试从不存在的文件加载配置"""
        config_file = self.temp_dir / "nonexistent.json"
        
        # 应该不会报错，使用默认配置
        self.config.load_from_file(config_file)
        self.assertEqual(self.config.min_length, 5.0)

    def test_save_to_file(self):
        """测试保存配置到文件"""
        config_file = self.temp_dir / "saved_config.json"
        
        # 修改一些配置值
        self.config.min_length = 4.0
        self.config.max_length = 16.0
        
        self.config.save_to_file(config_file)
        
        # 验证文件存在并包含正确内容
        self.assertTrue(config_file.exists())
        
        with open(config_file, "r") as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["min_length"], 4.0)
        self.assertEqual(saved_data["max_length"], 16.0)

    def test_validate_valid_config(self):
        """测试有效配置的验证"""
        is_valid, errors = self.config.validate()
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_invalid_min_length(self):
        """测试无效最小长度的验证"""
        self.config.min_length = -1.0
        
        is_valid, errors = self.config.validate()
        
        self.assertFalse(is_valid)
        self.assertIn("min_element_length must be positive", errors)

    def test_validate_invalid_max_length(self):
        """测试无效最大长度的验证"""
        self.config.min_length = 10.0
        self.config.max_length = 5.0  # 小于最小长度
        
        is_valid, errors = self.config.validate()
        
        self.assertFalse(is_valid)
        self.assertIn("max_element_length must be greater than min_element_length", errors)

    def test_validate_invalid_timeout(self):
        """测试无效超时时间的验证"""
        self.config.timeout = -100
        
        is_valid, errors = self.config.validate()
        
        self.assertFalse(is_valid)
        self.assertIn("timeout must be positive", errors)

    def test_validate_invalid_retry_attempts(self):
        """测试无效重试次数的验证"""
        self.config.retry_attempts = -1
        
        is_valid, errors = self.config.validate()
        
        self.assertFalse(is_valid)
        self.assertIn("retry_attempts must be non-negative", errors)

    def test_update_thresholds(self):
        """测试更新阈值"""
        new_thresholds = {
            "min_length": 2.5,
            "aspect_ratio": 4.0,
            "invalid_key": 999  # 这个键不应该被设置
        }
        
        self.config.update_thresholds(new_thresholds)
        
        self.assertEqual(self.config.min_length, 2.5)
        self.assertEqual(self.config.aspect_ratio, 4.0)
        self.assertFalse(hasattr(self.config, "invalid_key"))

    def test_get_effective_thresholds_default(self):
        """测试获取默认有效阈值"""
        thresholds = self.config.get_effective_thresholds()
        
        expected_keys = [
            "min_length", "max_length", "aspect_ratio", "skewness", "warping",
            "min_angle_quads", "max_angle_quads", "min_angle_trias", "max_angle_trias"
        ]
        
        for key in expected_keys:
            self.assertIn(key, thresholds)
        
        self.assertEqual(thresholds["min_length"], 5.0)
        self.assertEqual(thresholds["max_length"], 15.0)

    def test_get_effective_thresholds_custom(self):
        """测试获取自定义有效阈值"""
        custom_thresholds = {
            "min_length": 3.0,
            "aspect_ratio": 5.0
        }
        
        thresholds = self.config.get_effective_thresholds(custom_thresholds)
        
        self.assertEqual(thresholds["min_length"], 3.0)  # 被覆盖
        self.assertEqual(thresholds["aspect_ratio"], 5.0)  # 被覆盖
        self.assertEqual(thresholds["max_length"], 15.0)  # 保持默认值

    def test_get_config_summary(self):
        """测试获取配置摘要"""
        summary = self.config.get_config_summary()
        
        expected_keys = ["thresholds", "execution", "files", "quality_checks"]
        for key in expected_keys:
            self.assertIn(key, summary)
        
        self.assertIn("timeout", summary["execution"])
        self.assertIn("qual_file", summary["files"])
        self.assertIn("min_length", summary["quality_checks"])


class TestConfigUtilityFunctions(unittest.TestCase):
    """配置工具函数的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_default_config(self):
        """测试创建默认配置"""
        config = create_default_config(self.temp_dir)
        
        self.assertIsInstance(config, AnsaBatchConfig)
        self.assertEqual(config.cwd_dir, self.temp_dir)
        self.assertEqual(config.min_length, 5.0)

    def test_load_config_from_file(self):
        """测试从文件加载配置"""
        config_file = self.temp_dir / "test_config.json"
        test_config = {
            "min_length": 2.5,
            "max_length": 12.0
        }
        
        with open(config_file, "w") as f:
            json.dump(test_config, f)
        
        config = load_config_from_file(config_file, self.temp_dir)
        
        self.assertIsInstance(config, AnsaBatchConfig)
        self.assertEqual(config.min_length, 2.5)
        self.assertEqual(config.max_length, 12.0)


if __name__ == "__main__":
    unittest.main()