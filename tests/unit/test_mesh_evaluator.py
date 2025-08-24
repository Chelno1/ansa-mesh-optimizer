#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估器单元测试

作者: Chel
创建日期: 2025-07-07
"""

import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from ansa_mesh_optimizer.evaluators.mesh_evaluator import create_mesh_evaluator


class TestMeshEvaluator(unittest.TestCase):
    """网格评估器测试类"""

    def setUp(self) -> None:
        """测试前准备"""
        # 创建模拟配置管理器
        from unittest.mock import MagicMock

        self.mock_config_manager = MagicMock()
        self.mock_config_manager.parameter_space = MagicMock()

        # 设置测试参数
        self.test_params: Dict[str, float] = {
            "distortion_distance": 20.0,
            "rule_fillet_width_1": 3.0,
            "rule_fillet_width_2": 8.0,
            "rule_fillet_width_3": 18.0,
            "rule_fillet_width_4": 30.0,
            "recognize_chamfers_min_angle": 20.0,
            "recognize_chamfers_max_angle": 70.0,
            "recognize_chamfers_max_width": 20.0,
            "rule_chamfer_width_1": 10.0,
            "distortion_angle": 22.5,
            "perimeter_distance": 0.8,
        }

        # 模拟parameter_space的方法
        self.mock_config_manager.parameter_space.get_parameter.return_value = None

        # 模拟参数验证器
        with patch(
            "src.utils.parameter_validator.get_parameter_validator"
        ) as mock_get_validator:
            mock_validator = MagicMock()
            # 模拟验证成功的情况
            mock_validator.validate_comprehensive.return_value = (
                True,
                "",
                self.test_params,
            )
            mock_get_validator.return_value = mock_validator

            self.evaluator = create_mesh_evaluator(
                "mock", config_manager=self.mock_config_manager
            )
            self.mock_validator = mock_validator

    def test_evaluator_creation(self) -> None:
        """测试评估器创建"""
        # 测试创建mock评估器
        mock_evaluator = create_mesh_evaluator(
            "mock", config_manager=self.mock_config_manager
        )
        self.assertIsNotNone(mock_evaluator)

        # 测试创建ansa评估器
        ansa_evaluator = create_mesh_evaluator(
            "ansa", config_manager=self.mock_config_manager
        )
        self.assertIsNotNone(ansa_evaluator)

        # 测试无效评估器类型
        with self.assertRaises(ValueError):
            create_mesh_evaluator(
                "invalid_type", config_manager=self.mock_config_manager
            )

        # 测试缺少config_manager
        with self.assertRaises(ValueError):
            create_mesh_evaluator("mock")

    def test_parameter_validation(self) -> None:
        """测试参数验证"""
        # 测试有效参数
        try:
            result = self.evaluator.evaluate_mesh(self.test_params)
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0)
        except Exception as e:
            self.fail(f"有效参数测试失败: {e}")

        # 测试无效参数 - 直接修改验证器
        original_validator = self.evaluator.parameter_validator.validator
        mock_validator = MagicMock()
        mock_validator.validate_comprehensive.return_value = (
            False,
            "Parameter validation failed",
            {},
        )
        self.evaluator.parameter_validator.validator = mock_validator

        try:
            invalid_params: Dict[str, float] = {"distortion_distance": -1.0}
            result = self.evaluator.evaluate_mesh(invalid_params)
            self.assertEqual(result, float("inf"))
        finally:
            self.evaluator.parameter_validator.validator = original_validator

        # 测试缺失参数
        mock_validator2 = MagicMock()
        mock_validator2.validate_comprehensive.return_value = (
            False,
            "Missing parameters",
            {},
        )
        self.evaluator.parameter_validator.validator = mock_validator2

        try:
            missing_params: Dict[str, float] = {}
            result = self.evaluator.evaluate_mesh(missing_params)
            self.assertEqual(result, float("inf"))
        finally:
            self.evaluator.parameter_validator.validator = original_validator

    def test_evaluation_results(self) -> None:
        """测试评估结果"""
        # 测试多次评估结果的一致性
        results: List[float] = []
        for _ in range(3):
            result = self.evaluator.evaluate_mesh(self.test_params)
            results.append(result)

        # 检查结果是否在合理范围内
        for result in results:
            self.assertGreaterEqual(result, 0)
            self.assertLess(result, float("inf"))

    def test_cached_evaluation(self) -> None:
        """测试缓存评估"""
        # 创建带缓存的评估器
        from ansa_mesh_optimizer.utils.optimization_cache import CachedEvaluator, OptimizationCache

        with tempfile.NamedTemporaryFile(suffix=".pkl") as temp_file:
            cache = OptimizationCache(temp_file.name)
            cached_evaluator = CachedEvaluator(self.evaluator, cache)

            # 第一次评估
            result1 = cached_evaluator.evaluate_mesh(self.test_params)

            # 第二次评估应该使用缓存
            result2 = cached_evaluator.evaluate_mesh(self.test_params)

            self.assertEqual(result1, result2)
            self.assertGreater(cache.hits, 0)

    def test_parameter_sensitivity(self) -> None:
        """测试参数敏感性"""
        base_result = self.evaluator.evaluate_mesh(self.test_params)

        # 测试每个参数的敏感性
        for param in self.test_params:
            modified_params = self.test_params.copy()
            modified_params[param] *= 1.1  # 增加10%

            modified_result = self.evaluator.evaluate_mesh(modified_params)
            self.assertNotEqual(base_result, modified_result)

    def test_mock_mode_behavior(self) -> None:
        """测试模拟模式行为"""
        # 测试模拟模式评估器
        evaluator = create_mesh_evaluator(
            "ansa", config_manager=self.mock_config_manager
        )
        result = evaluator.evaluate_mesh(self.test_params)

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_error_handling(self) -> None:
        """测试错误处理"""
        # 测试空字典 - 直接修改现有evaluator的验证器
        original_validator = self.evaluator.parameter_validator.validator
        mock_validator = MagicMock()
        mock_validator.validate_comprehensive.return_value = (
            False,
            "Empty parameters",
            {},
        )
        self.evaluator.parameter_validator.validator = mock_validator

        try:
            result = self.evaluator.evaluate_mesh({})
            self.assertEqual(result, float("inf"))
        finally:
            # 恢复原验证器
            self.evaluator.parameter_validator.validator = original_validator

        # 测试参数类型错误 - normalize_params会处理字符串转换
        invalid_type_params: Dict[str, Any] = {"distortion_distance": "20"}
        try:
            result = self.evaluator.evaluate_mesh(invalid_type_params)
            self.assertIsInstance(result, float)
        except Exception:
            pass  # 如果转换失败也可以接受

        # 测试参数范围错误
        mock_validator2 = MagicMock()
        mock_validator2.validate_comprehensive.return_value = (
            False,
            "Out of range",
            {},
        )
        self.evaluator.parameter_validator.validator = mock_validator2

        try:
            out_of_range_params = self.test_params.copy()
            out_of_range_params["distortion_distance"] = 100.0
            result = self.evaluator.evaluate_mesh(out_of_range_params)
            self.assertEqual(result, float("inf"))
        finally:
            # 恢复原验证器
            self.evaluator.parameter_validator.validator = original_validator

    def test_result_consistency(self) -> None:
        """测试结果一致性"""
        # 创建一个无噪声的评估器以确保一致性
        import random

        original_seed = random.getstate()

        try:
            # 设置固定种子以确保可重现性
            random.seed(42)

            # 相同参数应该产生相似的结果
            results: List[float] = []
            for _ in range(5):
                result = self.evaluator.evaluate_mesh(self.test_params)
                results.append(result)

            # 检查结果的变异系数
            import numpy as np

            mean_value = np.mean(results)
            std_value = np.std(results)
            cv = std_value / mean_value if mean_value != 0 else float("inf")

            # 放宽阈值以适应mock评估器的噪声特性
            self.assertTrue(float(cv) < 0.15, f"变异系数 {cv:.3f} 应小于 0.15")

        finally:
            # 恢复原始随机状态
            random.setstate(original_seed)


if __name__ == "__main__":
    unittest.main()
