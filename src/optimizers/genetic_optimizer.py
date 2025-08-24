#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重构的遗传算法优化器 - 精简协调版本

负责遗传算法优化的总体协调，将具体的进化循环和结果分析委托给专门的模块。

作者: Chel
创建日期: 2025-06-19
版本: 3.0.0
更新日期: 2025-08-24
重构: 模块化设计，职责分离，精简主类为协调器
"""

import logging
import random
import numpy as np
from typing import Dict, List, Optional, Any

# 导入模块化组件
from .genetic_config import GeneticConfig
from .evolution import EvolutionEngine
from .evolution_loop import EvolutionLoop
from .optimizer_analysis import OptimizerAnalyzer
from .genetic_visualization import GeneticVisualizer, export_evolution_data
from .optimizer_config import OptimizationResult

logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """
    遗传算法优化器 - 协调器版本
    
    职责：
    - 初始化和协调各个模块
    - 提供统一的优化接口
    - 管理全局配置和状态
    """
    
    def __init__(
        self, 
        param_space, 
        evaluator, 
        config: Optional[Any] = None, 
        genetic_config: Optional[GeneticConfig] = None
    ):
        """
        初始化遗传算法优化器
        
        Args:
            param_space: 参数空间定义
            evaluator: 目标函数评估器
            config: 通用优化配置
            genetic_config: 遗传算法专用配置
        """
        self.param_space = param_space
        self.evaluator = evaluator
        self.config = config
        self.genetic_config = genetic_config or GeneticConfig()
        
        # 验证配置
        self._validate_configuration()
        
        # 获取参数空间信息
        self.bounds = param_space.get_bounds()
        self.param_types = param_space.get_param_types()
        self.param_names = param_space.get_param_names()
        
        # 初始化核心组件
        self._initialize_components()
        
        # 设置随机种子
        self._setup_random_seed()
        
        logger.info(f"遗传算法优化器初始化完成 - 种群大小: {self.genetic_config.population_size}")
    
    def optimize(self, n_calls: int, **kwargs) -> OptimizationResult:
        """
        执行遗传算法优化
        
        Args:
            n_calls: 总评估次数限制
            **kwargs: 其他优化参数
            
        Returns:
            OptimizationResult: 优化结果对象
        """
        # 计算动态参数
        population_size = min(
            self.genetic_config.population_size, 
            max(10, n_calls // 5)
        )
        max_generations = min(
            self.genetic_config.max_generations, 
            n_calls // population_size
        )
        
        logger.info(f"开始遗传算法优化: 种群={population_size}, 最大代数={max_generations}")
        
        try:
            # 执行进化循环
            total_generations, total_evaluations, execution_time = self.evolution_loop.run_evolution(
                n_calls=n_calls,
                population_size=population_size,
                max_generations=max_generations,
                verbose=self.config.verbose if self.config else False
            )
            
            # 生成优化结果
            result = self.analyzer.generate_optimization_result(
                best_individual=self.evolution_loop.get_best_individual(),
                generation_stats=self.evolution_loop.get_generation_stats(),
                evolution_engine=self.evolution_engine,
                total_generations=total_generations,
                total_evaluations=total_evaluations,
                execution_time=execution_time
            )
            
            # 记录优化完成信息
            best_score = result.best_score
            convergence_info = result.convergence_info or {}
            
            logger.info(
                f"遗传算法优化完成: 最佳适应度={best_score:.6f}, "
                f"总代数={total_generations}, 总评估次数={total_evaluations}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"遗传算法优化过程异常: {e}")
            # 返回空结果
            return self.analyzer.generate_optimization_result(
                best_individual=None,
                generation_stats=[],
                evolution_engine=self.evolution_engine,
                total_generations=0,
                total_evaluations=0,
                execution_time=0.0
            )
    
    def plot_evolution(self, save_path: Optional[str] = None, show_diversity: bool = True) -> None:
        """
        绘制进化过程
        
        Args:
            save_path: 保存路径
            show_diversity: 是否显示多样性曲线
        """
        generation_stats = self.evolution_loop.get_generation_stats()
        if not generation_stats:
            logger.warning("没有进化数据可以绘制")
            return
        
        # 获取进化数据
        diversity_metrics = self.evolution_engine.get_diversity_metrics()
        
        # 创建可视化器并绘制
        visualizer = GeneticVisualizer(
            best_fitness_history=diversity_metrics['best_fitness_history'],
            diversity_history=diversity_metrics['diversity_history'],
            generation_stats=generation_stats,
            genetic_config=self.genetic_config.to_dict()
        )
        
        visualizer.plot_evolution(save_path=save_path, show_diversity=show_diversity)
    
    def export_evolution_data(self, filename: str) -> None:
        """
        导出进化数据
        
        Args:
            filename: 导出文件名
        """
        try:
            diversity_metrics = self.evolution_engine.get_diversity_metrics()
            best_individual = self.evolution_loop.get_best_individual()
            generation_stats = self.evolution_loop.get_generation_stats()
            
            best_individual_info = {
                'genes': best_individual.genes if best_individual else None,
                'fitness': best_individual.fitness if best_individual else None,
                'params': best_individual.to_params(self.param_names) if best_individual else None,
                'generation': best_individual.generation if best_individual else None
            }
            
            metadata = {
                'total_generations': len(generation_stats),
                'parameter_names': self.param_names,
                'parameter_bounds': self.bounds,
                'parameter_types': [t.__name__ for t in self.param_types]
            }
            
            export_evolution_data(
                best_fitness_history=diversity_metrics['best_fitness_history'],
                diversity_history=diversity_metrics['diversity_history'],
                generation_stats=generation_stats,
                genetic_config=self.genetic_config.to_dict(),
                best_individual_info=best_individual_info,
                metadata=metadata,
                filename=filename
            )
            
        except Exception as e:
            logger.error(f"导出进化数据失败: {e}")
    
    def get_diversity_metrics(self) -> Dict[str, List[float]]:
        """获取种群多样性指标"""
        return self.evolution_engine.get_diversity_metrics()
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成优化总结报告"""
        generation_stats = self.evolution_loop.get_generation_stats()
        best_individual = self.evolution_loop.get_best_individual()
        
        if not best_individual:
            return {'error': 'No optimization results available'}
        
        # 创建临时结果对象用于报告生成
        result = self.analyzer.generate_optimization_result(
            best_individual=best_individual,
            generation_stats=generation_stats,
            evolution_engine=self.evolution_engine,
            total_generations=len(generation_stats),
            total_evaluations=0,  # 这里需要实际值，但为了简化暂时使用0
            execution_time=0.0    # 同上
        )
        
        return self.analyzer.generate_summary_report(result, generation_stats)
    
    def _validate_configuration(self) -> None:
        """验证配置有效性"""
        is_valid, errors = self.genetic_config.validate()
        if not is_valid:
            raise ValueError(f"遗传算法配置无效: {errors}")
    
    def _initialize_components(self) -> None:
        """初始化核心组件"""
        # 创建进化引擎
        self.evolution_engine = EvolutionEngine(
            bounds=self.bounds,
            param_types=self.param_types,
            genetic_config=self.genetic_config
        )
        
        # 创建进化循环管理器
        self.evolution_loop = EvolutionLoop(
            evolution_engine=self.evolution_engine,
            evaluator=self.evaluator,
            param_names=self.param_names
        )
        
        # 创建结果分析器
        self.analyzer = OptimizerAnalyzer(
            param_names=self.param_names,
            bounds=self.bounds,
            genetic_config=self.genetic_config
        )
    
    def _setup_random_seed(self) -> None:
        """设置随机种子"""
        if self.config and hasattr(self.config, 'random_state'):
            random.seed(self.config.random_state)
            np.random.seed(self.config.random_state)


def create_genetic_optimizer(
    param_space, 
    evaluator, 
    config: Optional[Any] = None, 
    **genetic_kwargs
) -> GeneticOptimizer:
    """
    创建遗传算法优化器的工厂函数
    
    Args:
        param_space: 参数空间定义
        evaluator: 目标函数评估器
        config: 通用优化配置
        **genetic_kwargs: 遗传算法特定参数
        
    Returns:
        GeneticOptimizer: 配置好的遗传算法优化器
    """
    genetic_config = GeneticConfig(**genetic_kwargs)
    return GeneticOptimizer(param_space, evaluator, config, genetic_config)


# 向后兼容性测试
if __name__ == "__main__":
    # 简化的测试代码
    logger.info("精简版遗传算法优化器测试")
    
    # 创建简单的测试问题
    class TestEvaluator:
        def evaluate_mesh(self, params):
            x = params.get('x', 0)
            y = params.get('y', 0)
            return 100 * (y - x**2)**2 + (1 - x)**2
    
    class TestParamSpace:
        def get_bounds(self):
            return [(-2, 2), (-2, 2)]
        
        def get_param_types(self):
            return [float, float]
        
        def get_param_names(self):
            return ['x', 'y']
    
    # 创建和运行测试
    param_space = TestParamSpace()
    evaluator = TestEvaluator()
    
    optimizer = create_genetic_optimizer(
        param_space, evaluator,
        population_size=20,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    result = optimizer.optimize(n_calls=100)
    
    print(f"最佳参数: {result.best_params}")
    print(f"最佳值: {result.best_score:.6f}")
    print("精简版遗传算法测试完成!")
