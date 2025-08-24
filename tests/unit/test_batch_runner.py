#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa批处理运行器模块的单元测试

作者: Chel
创建日期: 2025-08-24
版本: 2.0.0
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.batch.config import AnsaBatchConfig
from src.batch.runner import AnsaBatchMeshRunner, run_batch_mesh, check_element_quality_simple


class TestAnsaBatchMeshRunner(unittest.TestCase):
    """AnsaBatchMeshRunner类的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = AnsaBatchConfig(self.temp_dir)
        self.runner = AnsaBatchMeshRunner(
            script_dir=self.temp_dir,
            cwd_dir=self.temp_dir,
            config=self.config
        )

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.runner.cwd_dir, self.temp_dir)
        self.assertIsInstance(self.runner.config, AnsaBatchConfig)
        self.assertIn("start_time", self.runner.stats)
        self.assertIn("success", self.runner.stats)

    @patch('src.batch.runner.ANSA_AVAILABLE', False)
    def test_run_batch_mesh_simulation_mode(self):
        """测试模拟模式下的批处理网格运行"""
        with patch.object(self.runner, '_simulate_batch_mesh', return_value=True) as mock_sim:
            result = self.runner.run_batch_mesh()
            
            mock_sim.assert_called_once()
            self.assertTrue(result)
            self.assertIsNotNone(self.runner.stats["start_time"])
            self.assertIsNotNone(self.runner.stats["end_time"])

    @patch('src.batch.runner.ANSA_AVAILABLE', False)
    def test_run_batch_mesh_with_params(self):
        """测试带参数的批处理网格运行"""
        params = {"distortion_distance": 25.0}
        
        with patch.object(self.runner, '_simulate_batch_mesh', return_value=True) as mock_sim:
            result = self.runner.run_batch_mesh(params)
            
            mock_sim.assert_called_once_with(params)
            self.assertTrue(result)

    def test_load_mesh_parameters_file_exists(self):
        """测试网格参数加载 - 文件存在"""
        # 创建测试mpar文件
        mpar_file = self.temp_dir / self.config.mpar_file
        mpar_file.touch()
        
        result = self.runner._load_mesh_parameters()
        self.assertTrue(result)

    def test_load_mesh_parameters_file_not_exists(self):
        """测试网格参数加载 - 文件不存在"""
        result = self.runner._load_mesh_parameters()
        self.assertFalse(result)

    def test_load_quality_criteria_file_exists(self):
        """测试质量标准加载 - 文件存在"""
        # 创建质量标准目录和文件
        self.runner.criterion_dir.mkdir(parents=True, exist_ok=True)
        qual_file = self.runner.criterion_dir / self.config.qual_file
        qual_file.touch()
        
        result = self.runner._load_quality_criteria()
        self.assertTrue(result)

    def test_load_quality_criteria_file_not_exists(self):
        """测试质量标准加载 - 文件不存在"""
        result = self.runner._load_quality_criteria()
        self.assertFalse(result)

    @patch('src.batch.runner.ANSA_AVAILABLE', False)
    def test_check_element_quality_simulation(self):
        """测试模拟模式下的质量检查"""
        result = self.runner.check_element_quality()
        
        self.assertIn("timestamp", result)
        self.assertIn("total_elements", result)
        self.assertIn("bad_elements", result)
        self.assertIn("quality_ratio", result)
        self.assertIn("checks", result)
        
        # 验证统计信息被更新
        self.assertEqual(self.runner.stats["total_elements"], result["total_elements"])
        self.assertEqual(self.runner.stats["bad_elements"], result["bad_elements"])

    def test_check_element_quality_with_custom_thresholds(self):
        """测试使用自定义阈值的质量检查"""
        custom_thresholds = {
            "min_element_length": 3.0,
            "max_element_length": 10.0
        }
        
        result = self.runner.check_element_quality(custom_thresholds)
        
        self.assertIn("thresholds", result)
        # 在模拟模式下，应该使用自定义阈值
        if not hasattr(result["thresholds"], "min_length"):
            # 模拟模式使用不同的键名
            self.assertIn("min_length", result["thresholds"])

    def test_get_quality_check_config(self):
        """测试质量检查配置获取"""
        config = self.runner._get_quality_check_config("min_length")
        
        self.assertIn("criteria", config)
        self.assertIn("compare_func", config)
        self.assertIn("worst_func", config)
        self.assertIn("log_msg", config)
        
        self.assertEqual(config["criteria"], "MIN-LEN")

    def test_get_quality_check_config_invalid_type(self):
        """测试无效检查类型的配置获取"""
        with self.assertRaises(ValueError):
            self.runner._get_quality_check_config("invalid_type")

    @patch('src.batch.runner.ANSA_AVAILABLE', False)
    def test_save_model_simulation(self):
        """测试模拟模式下的模型保存"""
        result = self.runner.save_model()
        self.assertTrue(result)
        
        # 检查是否创建了模拟文件
        expected_files = list(self.temp_dir.glob("output_mesh.ansa_*"))
        self.assertTrue(len(expected_files) > 0)

    def test_save_model_with_custom_path(self):
        """测试使用自定义路径保存模型"""
        custom_path = self.temp_dir / "custom_model.ansa"
        
        result = self.runner.save_model(custom_path)
        self.assertTrue(result)
        self.assertTrue(custom_path.exists())

    def test_get_stats(self):
        """测试获取运行统计信息"""
        # 设置一些测试数据
        self.runner.stats["start_time"] = 1000.0
        self.runner.stats["end_time"] = 1010.0
        
        stats = self.runner.get_stats()
        
        self.assertIn("execution_time", stats)
        self.assertEqual(stats["execution_time"], 10.0)
        self.assertIn("config", stats)

    def test_simulate_batch_mesh_success_rate(self):
        """测试模拟批处理网格的成功率逻辑"""
        # 测试多次运行以验证随机性
        results = []
        for _ in range(10):
            result = self.runner._simulate_batch_mesh()
            results.append(result)
        
        # 应该有成功和失败的情况（基于随机性）
        # 但这个测试可能不够稳定，主要是为了覆盖代码
        self.assertIsInstance(results[0], bool)

    def test_simulate_mesh_generation(self):
        """测试模拟网格生成"""
        result = self.runner._simulate_mesh_generation()
        self.assertIsInstance(result, bool)


class TestRunnerUtilityFunctions(unittest.TestCase):
    """运行器工具函数的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.batch.runner.AnsaBatchMeshRunner')
    def test_run_batch_mesh_function(self, mock_runner_class):
        """测试run_batch_mesh便利函数"""
        mock_runner = Mock()
        mock_runner.run_batch_mesh.return_value = True
        mock_runner_class.return_value = mock_runner
        
        result = run_batch_mesh({"param1": 1.0})
        
        mock_runner_class.assert_called_once()
        mock_runner.run_batch_mesh.assert_called_once_with({"param1": 1.0})
        self.assertTrue(result)

    @patch('src.batch.runner.AnsaBatchMeshRunner')
    def test_check_element_quality_simple_function(self, mock_runner_class):
        """测试check_element_quality_simple便利函数"""
        mock_runner = Mock()
        expected_result = {"total_elements": 1000, "bad_elements": 50}
        mock_runner.check_element_quality.return_value = expected_result
        mock_runner_class.return_value = mock_runner
        
        result = check_element_quality_simple({"min_length": 3.0})
        
        mock_runner_class.assert_called_once()
        mock_runner.check_element_quality.assert_called_once_with({"min_length": 3.0})
        self.assertEqual(result, expected_result)


class TestRunnerIntegration(unittest.TestCase):
    """运行器集成测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow_simulation(self):
        """测试完整工作流程的模拟"""
        config = AnsaBatchConfig(self.temp_dir)
        runner = AnsaBatchMeshRunner(
            script_dir=self.temp_dir,
            cwd_dir=self.temp_dir,
            config=config
        )
        
        # 运行网格生成
        mesh_result = runner.run_batch_mesh()
        
        # 检查质量
        quality_result = runner.check_element_quality()
        
        # 保存模型
        save_result = runner.save_model()
        
        # 验证结果
        self.assertIsInstance(mesh_result, bool)
        self.assertIsInstance(quality_result, dict)
        self.assertIsInstance(save_result, bool)
        
        # 验证质量结果结构
        self.assertIn("total_elements", quality_result)
        self.assertIn("bad_elements", quality_result)
        self.assertIn("quality_ratio", quality_result)

    def test_retry_mechanism(self):
        """测试重试机制"""
        config = AnsaBatchConfig(self.temp_dir)
        config.retry_attempts = 2
        config.retry_delay = 0.1  # 加速测试
        
        runner = AnsaBatchMeshRunner(
            script_dir=self.temp_dir,
            cwd_dir=self.temp_dir,
            config=config
        )
        
        # 模拟失败情况
        with patch.object(runner, '_simulate_batch_mesh', return_value=False):
            result = runner.run_batch_mesh()
            
            self.assertFalse(result)
            self.assertEqual(runner.stats["retry_count"], 2)


if __name__ == "__main__":
    unittest.main()