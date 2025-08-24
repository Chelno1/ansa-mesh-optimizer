#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估接口模块 - 重构版本

作者: Chel
创建日期: 2025-06-19
版本: 2.0.0
更新日期: 2025-08-24
重构: 拆分单体模块为多个专注模块
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple, Any, Union
import subprocess
import os
import logging
from pathlib import Path
import time
import random

# 导入配置管理器
from src.config.config import UnifiedConfigManager

# 导入参数验证器
from src.utils.parameter_validator import get_parameter_validator

# 导入参数替换策略
from src.evaluators.parameter_replacement_strategies import ParameterReplacementManager, format_mpar_parameter_value

# 导入重构后的工具模块
from src.evaluators.utils import (
    normalize_params,
    create_timestamped_temp_dir,
    copy_mpar_files_to_temp_dir,
    create_temp_config_in_dir,
    create_temp_config,
    parse_ansa_output,
    cleanup_temp_files,
    cleanup_temp_directory,
    process_parameter_files_in_temp_dir,
    simulate_evaluation
)

# 导入环境验证模块
from src.evaluators.environment import (
    validate_ansa_environment,
    check_input_files,
    run_ansa_batch,
    handle_ansa_returncode,
    AnsaEnvironmentValidator
)

logger = logging.getLogger(__name__)


class MeshEvaluator(ABC):
    """网格评估器抽象基类"""
    
    @abstractmethod
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        评估网格质量
        
        Args:
            params: 网格参数字典
            
        Returns:
            网格质量评分（越小越好）
        """
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, float]) -> bool:
        """
        验证参数有效性
        
        Args:
            params: 网格参数字典
            
        Returns:
            参数是否有效
        """
        pass


class AnsaMeshEvaluator(MeshEvaluator):
    """Ansa网格评估器 - 重构版本"""
    
    def __init__(self, config_manager=None):
        if config_manager is None:
            raise ValueError("AnsaMeshEvaluator requires a config_manager instance")
        self.config_manager = config_manager
        self.config = config_manager.ansa_config
        self.param_mapping = config_manager.parameter_space.get_ansa_mapping()
        self.validator = get_parameter_validator(config_manager.parameter_space)
        self.cwd_dir = Path.cwd().resolve()
        self.criterion_dir = self.cwd_dir / 'criterion'
        
        # 初始化参数替换策略管理器
        self.parameter_replacer = ParameterReplacementManager(config_manager)
        self._format_parameter_value = format_mpar_parameter_value
        
        # 初始化环境验证器
        self.env_validator = AnsaEnvironmentValidator(self.config)
        self._validate_environment()
    
    def _validate_environment(self) -> None:
        """验证Ansa环境"""
        try:
            self.env_validator.validate()
        except Exception as e:
            logger.warning(f"环境验证过程中发生异常: {e}")
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数有效性"""
        try:
            is_valid, error_msg, _ = self.validator.validate_comprehensive(params)
            if not is_valid:
                raise ValueError(f"Parameter validation failed: {error_msg}")
            return is_valid
        except ValueError:
            # 重新抛出ValueError以便测试能够捕获
            raise
        except Exception as e:
            logger.error(f"参数验证异常: {e}")
            raise ValueError(f"Parameter validation error: {e}")
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        评估网格质量 - 重构版本
        
        Args:
            params: 网格参数字典
            
        Returns:
            不合格网格单元数量
        """
        # 标准化和验证参数
        try:
            normalized_params = normalize_params(params)
            is_valid, error_msg, cleaned_params = self.validator.validate_comprehensive(normalized_params)
            
            if not is_valid:
                logger.error(f"参数验证失败: {error_msg}")
                return float('inf')
            
        except Exception as e:
            logger.error(f"参数处理失败: {e}")
            return float('inf')
        
        try:
            # 创建带时间戳的临时文件夹
            temp_dir = create_timestamped_temp_dir()
            
            # 将*.ansa_mpar文件拷贝到临时文件夹
            copied_mpar_files = copy_mpar_files_to_temp_dir(
                temp_dir, self.criterion_dir, self.config.mpar_file_pattern
            )
            
            # 在临时文件夹中创建临时配置文件
            config_file = create_temp_config_in_dir(temp_dir, cleaned_params, self._format_parameter_value)
            
            # 处理mpar参数文件替换（在临时文件夹中）
            process_parameter_files_in_temp_dir(temp_dir, cleaned_params, self.parameter_replacer)
            
            # 运行Ansa批处理
            bad_elements_count = self._run_ansa_batch(temp_dir)
            
            logger.info(f"网格评估完成: {bad_elements_count} 个不合格单元")
            return float(bad_elements_count)
            
        except Exception as e:
            logger.error(f"网格评估失败: {e}")
            return float('inf')
        
        finally:
            # 清理临时文件和文件夹
            temp_files = [
                locals().get('config_file'),
                locals().get('final_config_file')
            ]
            temp_dir_var = locals().get('temp_dir')
            
            cleanup_temp_files(temp_files)
            # 不清理临时文件夹
            # if temp_dir_var:
            #     cleanup_temp_directory(temp_dir_var)
    
    def _run_ansa_batch(self, temp_dir: str) -> float:
        """运行Ansa批处理 - 使用环境模块"""
        try:
            # 构建Ansa命令
            ansa_command = [
                self.config.ansa_executable,
                '-b',
                '-execpy', f"load_script: '{self.config.script_dir / self.config.batch_script}'",
                '-i', f"{self.cwd_dir / self.config.input_model}",
                '-changedir', temp_dir
            ]
            
            # 检查输入文件
            input_model_path = str(self.cwd_dir / self.config.input_model)
            batch_script_path = str(self.config.script_dir / self.config.batch_script)
            
            is_valid, error_msg = check_input_files(input_model_path, batch_script_path)
            if not is_valid:
                logger.error(error_msg)
                return simulate_evaluation()
            
            # 执行命令
            result = run_ansa_batch(ansa_command, temp_dir, self.config.execution_timeout)
            
            # 处理返回代码
            is_success, error_msg = handle_ansa_returncode(result)
            if not is_success:
                logger.error(error_msg)
                return simulate_evaluation()
            
            # 解析成功的输出
            bad_elements_count = parse_ansa_output(result.stdout)
            return bad_elements_count
            
        except subprocess.TimeoutExpired:
            logger.error(f"Ansa执行超时({self.config.execution_timeout}秒)")
            return simulate_evaluation()
        except FileNotFoundError:
            logger.error(f"Ansa可执行文件未找到: {self.config.ansa_executable}")
            logger.info("使用模拟模式")
            return simulate_evaluation()
        except PermissionError:
            logger.error("没有权限执行Ansa")
            return simulate_evaluation()
        except Exception as e:
            logger.exception(f"Ansa执行时发生意外错误: {e}")
            return simulate_evaluation()


class MockMeshEvaluator(MeshEvaluator):
    """模拟网格评估器（用于测试）- 重构版本"""
    
    def __init__(self, landscape_type: str = 'rosenbrock', add_noise: bool = True, config_manager=None):
        self.landscape_type = landscape_type
        self.add_noise = add_noise
        self.evaluation_count = 0
        
        if config_manager is None:
            raise ValueError("MockMeshEvaluator requires a config_manager instance")
        self.config_manager = config_manager
        self.validator = get_parameter_validator(config_manager.parameter_space)
        
        # 设置随机种子以便可重现
        random.seed(42)
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数有效性 - 修复版本"""
        try:
            is_valid, error_msg, _ = self.validator.validate_comprehensive(params)
            if not is_valid:
                # 对于测试，如果参数验证失败，抛出ValueError
                raise ValueError(f"Parameter validation failed: {error_msg}")
            return is_valid
        except ValueError:
            # 重新抛出ValueError以便测试能够捕获
            raise
        except Exception as e:
            logger.warning(f"Mock evaluator parameter validation failed: {e}")
            # 对于其他异常，也抛出ValueError以保持一致性
            raise ValueError(f"Parameter validation error: {e}")
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        模拟评估网格质量 - 增强版本
        
        使用多种数学函数模拟复杂的优化景观
        """
        self.evaluation_count += 1
        
        # 标准化参数
        try:
            normalized_params = normalize_params(params)
            is_valid, error_msg, cleaned_params = self.validator.validate_comprehensive(normalized_params)
            
            if not is_valid:
                logger.warning(f"Mock evaluator: invalid parameters - {error_msg}")
                # 与AnsaMeshEvaluator行为一致，返回inf而不是抛出异常
                return float('inf')
            
        except Exception as e:
            logger.error(f"Mock evaluator parameter processing failed: {e}")
            return float('inf')
        
        # 添加现实的延迟模拟
        if self.add_noise:
            time.sleep(random.uniform(0.05, 0.2))
        
        # 根据景观类型计算结果
        try:
            if self.landscape_type == 'rosenbrock':
                result = self._rosenbrock_function(cleaned_params)
            elif self.landscape_type == 'ackley':
                result = self._ackley_function(cleaned_params)
            elif self.landscape_type == 'rastrigin':
                result = self._rastrigin_function(cleaned_params)
            elif self.landscape_type == 'mesh_realistic':
                result = self._mesh_realistic_function(cleaned_params)
            else:
                result = self._rosenbrock_function(cleaned_params)
            
            # 添加噪声
            if self.add_noise:
                noise = random.gauss(0, 0.1 * result)
                result = max(0, result + noise)
            
            logger.debug(f"Mock evaluation #{self.evaluation_count}: {result:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"Mock evaluation function failed: {e}")
            return float('inf')
        
    def _fill_missing_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """填充缺失的参数"""
        default_params = self._get_default_params()

        # 使用提供的参数，缺失的用默认值填充
        filled_params = default_params.copy()
        filled_params.update(params)

        return filled_params
    
    def _get_default_params(self) -> Dict[str, float]:
        """获取默认参数"""
        return {
            'distortion_distance': 25
        }
    
    def _rosenbrock_function(self, params: Dict[str, float]) -> float:
        """Rosenbrock函数的变形（适合网格优化）"""
        x1 = params.get('distortion_distance', 20)
        x2 = params.get('perimeter_distance', 0.667)
        x3 = params.get('rule_fillet_width_1', 3.0)
        
        # 标准化到[-2, 2]范围
        x1_norm = (x1 - 20) / 10.0   # distortion_distance center at 20
        x2_norm = (x2 - 0.667) * 3.0 # perimeter_distance center at 0.667
        x3_norm = (x3 - 3.0) / 5.0   # rule_fillet_width_1 center at 3.0
        
        result = (
            100 * (x2_norm - x1_norm**2)**2 +
            (1 - x1_norm)**2 +
            50 * (x3_norm - 0.5)**2 +
            10  # base offset
        )
        
        return max(0, result)
    
    def _ackley_function(self, params: Dict[str, float]) -> float:
        """Ackley函数（多峰值景观）"""
        import math
        
        x1 = params.get('distortion_distance', 20)
        x2 = params.get('perimeter_distance', 0.667)
        x3 = params.get('rule_fillet_width_1', 3.0)
        
        # 标准化
        x = [x1 - 20, (x2 - 0.667) * 30, x3 - 3.0]
        n = len(x)
        
        sum_sq = sum(xi**2 for xi in x)
        sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x)
        
        result = (
            -20 * math.exp(-0.2 * math.sqrt(sum_sq / n)) -
            math.exp(sum_cos / n) + 20 + math.e
        ) * 10
        
        return max(0, result)
    
    def _rastrigin_function(self, params: Dict[str, float]) -> float:
        """Rastrigin函数（高度多峰值）"""
        import math
        
        x1 = params.get('distortion_distance', 20) - 20
        x2 = (params.get('perimeter_distance', 0.667) - 0.667) * 30
        x3 = params.get('rule_fillet_width_1', 3.0) - 3.0
        
        A = 10
        result = A * 3 + sum(
            xi**2 - A * math.cos(2 * math.pi * xi)
            for xi in [x1, x2, x3]
        )
        
        return max(0, result * 5)
    
    def _mesh_realistic_function(self, params: Dict[str, float]) -> float:
        """模拟真实网格优化函数"""
        x1 = params.get('distortion_distance', 20)
        x2 = params.get('perimeter_distance', 0.667)
        x3 = params.get('rule_fillet_width_1', 3.0)
        x4 = params.get('distortion_angle', 0.0)
        
        # 模拟网格质量与参数的非线性关系
        # 扭曲距离的影响
        distortion_penalty = abs(x1 - 20)**2 * 5
        
        # 周边距离的影响
        perimeter_penalty = abs(x2 - 0.667)**2 * 100
        
        # 圆角宽度的影响
        fillet_penalty = abs(x3 - 3.0)**2 * 10
        
        # 扭曲角度的影响
        angle_penalty = abs(x4)**2 * 20
        
        result = (
            distortion_penalty +
            perimeter_penalty +
            fillet_penalty +
            angle_penalty +
            random.uniform(10, 50)  # 基础偏移
        )
        
        return float(max(1, result))
    
    def get_optimal_params(self) -> Dict[str, float]:
        """获取当前景观的最优参数（用于测试）"""
        optimal_params = {
            'rosenbrock': {
                'distortion_distance': 20,
                'perimeter_distance': 0.667,
                'rule_fillet_width_1': 3.0,
                'distortion_angle': 0.0
            },
            'ackley': {
                'distortion_distance': 20,
                'perimeter_distance': 0.667,
                'rule_fillet_width_1': 3.0,
                'distortion_angle': 0.0
            },
            'mesh_realistic': {
                'distortion_distance': 20,
                'perimeter_distance': 0.667,
                'rule_fillet_width_1': 3.0,
                'distortion_angle': 0.0
            }
        }
        
        return optimal_params.get(self.landscape_type, optimal_params['rosenbrock'])


def create_mesh_evaluator(evaluator_type: str = 'ansa', config_manager=None) -> MeshEvaluator:
    """
    创建网格评估器
    
    Args:
        evaluator_type: 评估器类型 ('ansa' 或 'mock')
        config_manager: 配置管理器实例
        
    Returns:
        网格评估器实例
    """
    if config_manager is None:
        raise ValueError("create_mesh_evaluator requires a config_manager instance")
        
    if evaluator_type.lower() == 'ansa':
        return AnsaMeshEvaluator(config_manager=config_manager)
    elif evaluator_type.lower() == 'mock':
        return MockMeshEvaluator(config_manager=config_manager)
    elif evaluator_type.lower().startswith('mock_'):
        # 支持不同的mock类型，例如 'mock_ackley'
        landscape = evaluator_type[5:]  # 去掉 'mock_' 前缀
        return MockMeshEvaluator(landscape_type=landscape, config_manager=config_manager)
    else:
        raise ValueError(f"不支持的评估器类型: {evaluator_type}")


# 工具函数
def test_evaluator(evaluator: MeshEvaluator, n_tests: int = 5) -> None:
    """测试评估器功能"""
    print(f"Testing {evaluator.__class__.__name__}...")
    
    # 获取最优参数（如果是Mock评估器）
    if isinstance(evaluator, MockMeshEvaluator) and hasattr(evaluator, 'get_optimal_params'):
        test_params = evaluator.get_optimal_params()
    else:
        # 使用默认测试参数
        test_params = {
            'distortion_distance': 20,
            'perimeter_distance': 0.667,
            'rule_fillet_width_1': 3.0,
            'distortion_angle': 0.0
        }
    
    print(f"Test parameters: {test_params}")
    print(f"Parameter validation: {evaluator.validate_params(test_params)}")
    
    results = []
    for i in range(n_tests):
        result = evaluator.evaluate_mesh(test_params)
        results.append(result)
        print(f"Test {i+1}: {result:.6f}")
    
    if results:
        avg_result = sum(results) / len(results)
        print(f"Average result: {avg_result:.6f}")
    
    print("Testing completed!\n")


if __name__ == "__main__":
    # 测试评估器
    print("=== Mesh Evaluator Testing ===")
    
    # 测试Mock评估器
    mock_evaluator = create_mesh_evaluator('mock')
    test_evaluator(mock_evaluator, n_tests=3)
    
    # 测试不同景观的Mock评估器
    for landscape in ['ackley', 'mesh_realistic']:
        mock_eval = create_mesh_evaluator(f'mock_{landscape}')
        test_evaluator(mock_eval, n_tests=2)
    
    # 测试Ansa评估器（如果可用）
    try:
        ansa_evaluator = create_mesh_evaluator('ansa')
        print("Ansa evaluator created successfully")
        print(f"Parameter validation test: {ansa_evaluator.validate_params({'distortion_distance': 20, 'perimeter_distance': 0.8, 'rule_fillet_width_1': 3.0})}")
    except Exception as e:
        print(f"Ansa evaluator test skipped: {e}")
    
    print("All evaluator tests completed!")
