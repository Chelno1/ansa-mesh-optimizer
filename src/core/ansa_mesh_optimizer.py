#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa Batch Mesh Optimizer (重构版本)

优化有限元网格参数，最小化不合格网格数量
使用策略模式和模块化架构

作者: Chel
创建日期: 2025-06-09
版本: 2.0.0
更新日期: 2025-07-07
重构: 应用策略模式，模块化架构，SOLID原则
"""

import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 本地模块导入
try:
    # 使用重构后的配置
    from src.config.config import UnifiedConfigManager, OptimizationConfig
    
    # 创建兼容性包装器
    class ConfigManagerWrapper:
        def __init__(self, unified_manager):
            self.unified_manager = unified_manager
            self.optimization_config = unified_manager.optimization_config
            self.parameter_space = ParameterSpaceWrapper(unified_manager.parameter_space)
        
        def load_config(self, config_file):
            return self.unified_manager.load_config(config_file)
    
    class ParameterSpaceWrapper:
        def __init__(self, unified_param_space):
            self.unified_param_space = unified_param_space
        
        def get_param_names(self):
            return self.unified_param_space.get_parameter_names()
        
        def get_parameter_names(self):
            return self.unified_param_space.get_parameter_names()
        
        def get_bounds(self):
            return self.unified_param_space.get_bounds()
        
        def get_param_types(self):
            # 转换枚举类型为Python类型
            param_types = []
            for param_type in self.unified_param_space.get_parameter_types():
                if hasattr(param_type, 'value'):
                    if param_type.value == 'float':
                        param_types.append(float)
                    elif param_type.value == 'integer':
                        param_types.append(int)
                    else:
                        param_types.append(str)
                else:
                    param_types.append(param_type)
            return param_types
        
        def get_parameter_types(self):
            return self.unified_param_space.get_parameter_types()
        
        def to_skopt_space(self):
            return self.unified_param_space.to_skopt_space()
    
    logger.info("配置系统类已导入")
        
    # 导入重构后的模块
    from src.evaluators.mesh_evaluator import create_mesh_evaluator, MeshEvaluator
    from src.utils.optimization_cache import OptimizationCache, CachedEvaluator
    from src.core.early_stopping import create_early_stopping, EarlyStopping
    from src.utils.utils import normalize_params, validate_param_types, performance_monitor

    # 导入新的优化器策略模块
    from src.optimizers import (
        OptimizerFactory,
        OptimizerConfig,
        OptimizationResult,
        create_default_config
    )
    
    # 导入可视化和报告模块
    from src.visualization.optimization_visualizer import OptimizationVisualizer
    from src.reports.optimization_reporter import OptimizationReporter
    
except ImportError as e:
    logger.error(f"本地模块导入失败: {e}")
    logger.error("请确保所有必需的模块文件存在")
    raise

class MeshOptimizer:
    """网格参数优化器主类 - 重构版本
    
    使用策略模式实现不同的优化算法
    使用依赖注入实现模块化架构
    """
    
    def __init__(self,
                 config: Optional[OptimizationConfig] = None,
                 evaluator_type: str = 'ansa',
                 use_cache: bool = True,
                 optimizer_config: Optional[OptimizerConfig] = None,
                 config_manager: Optional[ConfigManagerWrapper] = None):
        """
        初始化优化器
        
        Args:
            config: 优化配置对象
            evaluator_type: 评估器类型 ('ansa' 或 'mock')
            use_cache: 是否使用缓存
            optimizer_config: 优化器配置对象
            config_manager: 配置管理器实例
        """
        if config_manager is None:
            raise ValueError("必须提供配置管理器实例")
        
        self.config = config or config_manager.optimization_config
        self.param_space = config_manager.parameter_space
        self.optimizer_config = optimizer_config or create_default_config()
        
        # 验证配置
        try:
            self.config.validate()
        except Exception as e:
            raise ValueError(f"配置验证失败: {e}")
        
        # 创建评估器
        self.base_evaluator = create_mesh_evaluator(evaluator_type, config_manager=config_manager.unified_manager)
        
        # 创建缓存（如果启用）
        if use_cache and self.config.use_cache:
            self.cache = OptimizationCache(self.config.cache_file)
            self.evaluator = CachedEvaluator(self.base_evaluator, self.cache)
        else:
            self.cache = None
            self.evaluator = self.base_evaluator
        
        # 创建早停机制
        if self.config.early_stopping:
            self.early_stopping = create_early_stopping(self.config)
        else:
            self.early_stopping = None
        
        # 延迟创建可视化器和报告器（避免创建不必要的空目录）
        self.visualizer = None
        self.reporter = None
        
        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_result: Optional[OptimizationResult] = None
        
        logger.info(f"优化器初始化完成 - 评估器: {evaluator_type}, 缓存: {use_cache}")
    
    def optimize(self, 
                 optimizer: str = 'bayesian',
                 n_calls: Optional[int] = None,
                 **kwargs) -> OptimizationResult:
        """
        执行优化
        
        Args:
            optimizer: 优化器类型
            n_calls: 优化迭代次数
            **kwargs: 其他优化器参数
            
        Returns:
            优化结果对象
        """
        n_calls = n_calls or self.config.n_calls
        
        logger.info(f"开始使用 {optimizer} 优化器进行网格参数优化")
        logger.info(f"迭代次数: {n_calls}")
        
        # 检查优化器可用性
        available_optimizers = OptimizerFactory.get_available_optimizers()
        if optimizer.lower() not in available_optimizers:
            raise ValueError(f"优化器 {optimizer} 不可用。可用优化器: {available_optimizers}")
        
        with performance_monitor(f"{optimizer} 优化"):
            try:
                start_time = time.time()
                
                # 创建优化器策略
                optimizer_strategy = OptimizerFactory.create_optimizer(
                    optimizer_type=optimizer,
                    param_space=self.param_space,
                    evaluator=self.evaluator,
                    config=self.optimizer_config
                )
                
                # 执行优化
                result = optimizer_strategy.optimize(n_calls, **kwargs)
                
                # 计算执行时间
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                result.success = True
                
                # 创建优化结果对象
                # result = OptimizationResult(
                #     best_params=result_dict['best_params'],
                #     best_value=result_dict['best_value'],
                #     optimizer_name=result_dict['optimizer_name'],
                #     optimization_history=result_dict.get('optimization_history', []),
                #     convergence_info=result_dict.get('convergence_info'),
                #     execution_time=execution_time,
                #     n_evaluations=len(result_dict.get('optimization_history', [])),
                #     success=True
                # )
                
                # 更新历史记录
                self.optimization_history = result.optimization_history
                self.best_result = result
                
                # 生成报告（传递原始result_dict以保留skopt_result）
                try:
                    report_dir = self._generate_optimization_report(result)
                    logger.info(f"详细报告已保存到: {report_dir}")
                except Exception as e:
                    logger.warning(f"报告生成失败: {e}")
                
                logger.info(f"优化完成")
                logger.info(f"最佳目标值: {result.best_value:.6f}")
                logger.info(f"执行时间: {execution_time:.2f}秒")
                
                return result
                
            except Exception as e:
                logger.error(f"优化过程中发生错误: {e}")
                
                # 创建失败结果
                result = OptimizationResult(
                    best_params={},
                    best_value=float('inf'),
                    optimizer_name=optimizer,
                    optimization_history=[],
                    success=False,
                    error_message=str(e)
                )
                
                return result
                
            finally:
                # 保存缓存
                if self.cache:
                    try:
                        self.cache._save_cache()
                    except Exception as e:
                        logger.warning(f"缓存保存失败: {e}")
    
    def _generate_optimization_report(self, result: OptimizationResult) -> str:
        """生成优化报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        optimizer_name = result.optimizer_name.replace(' ', '_')
        report_dir = Path(f"optimization_reports/{timestamp}_{optimizer_name}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建报告器和可视化器（使用指定目录）
        reporter = OptimizationReporter(report_dir)
        visualizer = OptimizationVisualizer(report_dir)
        
        # 使用报告器生成报告
        reporter.generate_optimization_report(
            result=result.to_dict(),
            optimization_history=self.optimization_history,
            config=self.config,
            param_space=self.param_space,
            cache_stats=self.cache.get_stats() if self.cache else None,
            early_stopping_info=self.early_stopping.get_best_result() if self.early_stopping else None
        )
        
        # 使用可视化器生成图表
        try:
            # visualization_result = result
            visualizer.generate_optimization_plots(
                result=result,
                optimization_history=self.optimization_history,
                early_stopping=self.early_stopping
            )
        except Exception as e:
            logger.warning(f"生成可视化图表失败: {e}")
        
        return str(report_dir)
    
    def sensitivity_analysis(self, 
                           best_params: Optional[Dict[str, float]] = None,
                           n_trials: int = 5,
                           noise_level: float = 0.1) -> Dict[str, List[tuple]]:
        """
        参数敏感性分析
        
        Args:
            best_params: 最佳参数（如果为None则使用最近的优化结果）
            n_trials: 每个参数的试验次数
            noise_level: 参数扰动幅度
            
        Returns:
            敏感性分析结果
        """
        if best_params is None:
            if self.best_result is None:
                raise ValueError("没有可用的最佳参数，请先运行优化或提供参数")
            best_params = self.best_result.best_params
        
        # 确保 best_params 不为 None
        if best_params is None:
            raise ValueError("最佳参数为空，无法进行敏感性分析")
        
        logger.info("开始参数敏感性分析...")
        
        with performance_monitor("敏感性分析"):
            sensitivity_results = {}
            bounds = self.param_space.get_bounds()
            param_names = self.param_space.get_param_names()
            param_types = self.param_space.get_param_types()
            
            for i, param_name in enumerate(param_names):
                param_value = best_params[param_name]
                param_type = param_types[i]
                low, high = bounds[i]
                
                logger.info(f"分析参数: {param_name}")
                
                # 确定参数类型并设置合适的扰动范围
                if param_type == float:
                    min_val = max(low, param_value * (1 - noise_level))
                    max_val = min(high, param_value * (1 + noise_level))
                    test_values = np.linspace(min_val, max_val, n_trials)
                else:  # int
                    range_size = int(max(1, param_value * noise_level))
                    min_val = max(low, int(param_value - range_size))
                    max_val = min(high, int(param_value + range_size))
                    test_values = np.linspace(min_val, max_val, n_trials, dtype=int)
                
                # 测试不同参数值的影响
                results = []
                for test_val in test_values:
                    test_params = best_params.copy()
                    test_params[param_name] = test_val
                    
                    try:
                        result = self.evaluator.evaluate_mesh(test_params)
                        results.append((test_val, result))
                        logger.debug(f"  {param_name}={test_val:.4f} -> {result:.4f}")
                    except Exception as e:
                        logger.warning(f"敏感性分析评估失败: {e}")
                        results.append((test_val, float('inf')))
                
                sensitivity_results[param_name] = results
            
            # 生成敏感性分析图表
            try:
                if self.best_result:
                    # 使用当前优化的报告目录
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    optimizer_name = self.best_result.optimizer_name.replace(' ', '_')
                    report_dir = Path(f"optimization_reports/{timestamp}_{optimizer_name}_sensitivity")
                    report_dir.mkdir(parents=True, exist_ok=True)
                    
                    visualizer = OptimizationVisualizer(report_dir)
                    visualizer.plot_sensitivity_analysis(
                        sensitivity_results=sensitivity_results,
                        best_params=best_params,
                        save_path=str(report_dir / "sensitivity_analysis.png")
                    )
                    
            except Exception as e:
                logger.warning(f"生成敏感性分析图表失败: {e}")
        
        logger.info("参数敏感性分析完成")
        return sensitivity_results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要信息"""
        summary = {
            'total_evaluations': len(self.optimization_history),
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'config': {
                'optimizer_config': self.optimizer_config.to_dict(),
                'optimization_config': {
                    'early_stopping': self.config.early_stopping,
                    'use_cache': self.config.use_cache,
                    'n_calls': self.config.n_calls
                },
                'available_optimizers': OptimizerFactory.get_available_optimizers()
            }
        }
        
        if self.cache:
            summary['cache_stats'] = self.cache.get_stats()
        
        if self.early_stopping and hasattr(self.early_stopping, 'get_best_result'):
            summary['early_stopping_info'] = self.early_stopping.get_best_result()
        
        # 添加性能统计
        if self.optimization_history:
            results = [entry['result'] for entry in self.optimization_history]
            summary['performance_stats'] = {
                'best_value': min(results),
                'worst_value': max(results),
                'mean_value': np.mean(results),
                'std_value': np.std(results),
                'improvement_count': sum(1 for i in range(1, len(results)) if results[i] < results[i-1])
            }
        
        return summary
    
    # def save_best_params(self, filename: Optional[str] = None) -> str:
    #     """
    #     保存最佳参数到文件
        
    #     Args:
    #         filename: 保存文件名（可选）
            
    #     Returns:
    #         保存的文件路径
    #     """
    #     if self.best_result is None:
    #         raise ValueError("没有可用的最佳参数，请先运行优化")
        
    #     if filename is None:
    #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #         optimizer_name = self.best_result.optimizer_name.replace(' ', '_').lower()
    #         report_dir = Path(f"optimization_reports/{timestamp}_{optimizer_name}")
    #         report_dir.mkdir(parents=True, exist_ok=True)
    #         filename = str(report_dir / "best_parameters.txt")
        
    #     try:
    #         with open(filename, 'w', encoding='utf-8') as f:
    #             f.write(f"# Best Mesh Parameters - {self.best_result.optimizer_name}\n")
    #             f.write(f"# Generated: {datetime.now().isoformat()}\n")
    #             f.write(f"# Best Objective Value: {self.best_result.best_value:.6f}\n")
    #             f.write(f"# Total Evaluations: {self.best_result.n_evaluations}\n")
    #             f.write(f"# Execution Time: {self.best_result.execution_time:.2f}s\n\n")
                
    #             for key, value in self.best_result.best_params.items():
    #                 f.write(f"{key} = {value}\n")
            
    #         logger.info(f"最佳参数已保存到: {filename}")
    #         return filename
            
    #     except Exception as e:
    #         logger.error(f"保存最佳参数失败: {e}")
    #         raise

def optimize_mesh_parameters(
    n_calls: int = 20,
    optimizer: str = 'bayesian',
    evaluator_type: str = 'ansa',
    config_file: Optional[str] = None,
    use_cache: bool = True,
    config_manager: Optional[ConfigManagerWrapper] = None,
    **kwargs
) -> OptimizationResult:
    """
    便捷的网格参数优化函数
    
    Args:
        n_calls: 优化迭代次数
        optimizer: 优化器类型
        evaluator_type: 评估器类型
        config_file: 配置文件路径
        use_cache: 是否使用缓存
        config_manager: 配置管理器实例
        **kwargs: 其他优化器参数
        
    Returns:
        优化结果对象
    """
    # 检查配置管理器
    if config_manager is None:
        if config_file is None:
            raise ValueError("必须提供配置文件或配置管理器实例")
        # 创建配置管理器
        unified_manager = UnifiedConfigManager(config_file=config_file, require_config=True)
        config_manager = ConfigManagerWrapper(unified_manager)
    
    # 创建优化器
    mesh_optimizer = MeshOptimizer(
        evaluator_type=evaluator_type,
        use_cache=use_cache,
        config_manager=config_manager
    )
    
    # 执行优化
    result = mesh_optimizer.optimize(
        optimizer=optimizer,
        n_calls=n_calls,
        **kwargs
    )
    
    # 运行敏感性分析（如果启用）
    if config_manager.optimization_config.sensitivity_analysis:
        try:
            mesh_optimizer.sensitivity_analysis(
                n_trials=config_manager.optimization_config.sensitivity_trials,
                noise_level=config_manager.optimization_config.noise_level
            )
        except Exception as e:
            logger.warning(f"敏感性分析失败: {e}")
    
    # 保存最佳参数
    # try:
    #     mesh_optimizer.save_best_params()
    # except Exception as e:
    #     logger.warning(f"保存最佳参数失败: {e}")
    
    return result

def get_available_optimizers() -> List[str]:
    """获取可用的优化器列表"""
    return OptimizerFactory.get_available_optimizers()

def check_dependencies() -> Dict[str, Any]:
    """检查依赖库状态"""
    from src.optimizers.optimizer_strategies import SKOPT_MODULES
    
    result = {
        'available_optimizers': get_available_optimizers(),
        'optimizer_factory_available': True,
        'visualization_available': True,
        'reporting_available': True,
        'skopt_available': SKOPT_MODULES['available']  # 添加缺失的键
    }
    
    # 添加调试日志
    logger.info(f"依赖检查结果: {result}")
    
    return result

if __name__ == "__main__":
    # 示例用法
    logger.info("Ansa网格优化器示例 - 重构版本")
    
    # 检查依赖
    deps = check_dependencies()
    print(f"可用优化器: {deps['available_optimizers']}")
    
    # 单个优化器测试
    try:
        result = optimize_mesh_parameters(
            n_calls=10,  # 减少迭代次数以便快速测试
            optimizer='genetic',  # 使用总是可用的遗传算法
            evaluator_type='mock'
        )
        
        print(f"\n最佳参数: {result.best_params}")
        print(f"最佳值: {result.best_value:.6f}")
        print(f"执行时间: {result.execution_time:.2f}秒")
        print(f"评估次数: {result.n_evaluations}")
        
    except Exception as e:
        logger.error(f"优化测试失败: {e}")
    
    print("\n示例运行完成!")