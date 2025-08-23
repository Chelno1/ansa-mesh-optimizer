#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重构的遗传算法优化器 - 模块化版本

作者: Chel
创建日期: 2025-06-19
版本: 2.0.0
更新日期: 2025-08-23
重构: 模块化设计，职责分离，维护性提升
"""

import logging
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# 导入新的模块化组件
from .genetic_config import GeneticConfig
from .individual import Individual
from .evolution import EvolutionEngine
from .genetic_visualization import GeneticVisualizer, export_evolution_data
from .optimizer_config import OptimizationResult

logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """遗传算法优化器 - 重构模块化版本"""
    
    def __init__(self, param_space, evaluator, config=None, genetic_config=None):
        """
        初始化遗传算法优化器
        
        Args:
            param_space: 参数空间
            evaluator: 评估器
            config: 优化配置
            genetic_config: 遗传算法配置
        """
        self.param_space = param_space
        self.evaluator = evaluator
        self.config = config
        self.genetic_config = genetic_config or GeneticConfig()
        
        # 验证遗传算法配置
        is_valid, errors = self.genetic_config.validate()
        if not is_valid:
            raise ValueError(f"遗传算法配置无效: {errors}")
        
        # 获取参数空间信息
        self.bounds = param_space.get_bounds()
        self.param_types = param_space.get_param_types()
        self.param_names = param_space.get_param_names()
        
        # 创建进化引擎
        self.evolution_engine = EvolutionEngine(
            bounds=self.bounds,
            param_types=self.param_types,
            genetic_config=self.genetic_config
        )
        
        # 优化历史和状态
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None
        self.best_ever_fitness = float('inf')
        self.best_ever_individual: Optional[Individual] = None
        
        # 设置随机种子
        if config and hasattr(config, 'random_state'):
            random.seed(config.random_state)
            np.random.seed(config.random_state)
        
        logger.info(f"遗传算法优化器初始化完成 - 种群大小: {self.genetic_config.population_size}")
    
    def optimize(self, n_calls: int, **kwargs) -> OptimizationResult:
        """
        执行遗传算法优化
        
        Args:
            n_calls: 总评估次数
            **kwargs: 其他参数
            
        Returns:
            优化结果
        """
        # 根据评估次数调整种群大小和代数
        population_size = min(self.genetic_config.population_size, max(10, n_calls // 5))
        max_generations = min(self.genetic_config.max_generations, n_calls // population_size)
        
        logger.info(f"遗传算法优化开始: 种群大小={population_size}, 最大代数={max_generations}")
        
        start_time = time.time()
        
        try:
            # 初始化种群
            population = self.evolution_engine.initialize_population(population_size)

            # 评估初始种群
            self._evaluate_population(population)

            # 检查初始种群是否有有效个体
            valid_individuals = [ind for ind in population if ind.fitness != float('inf')]
            if not valid_individuals:
                logger.warning("初始种群中没有有效个体，使用默认结果")
                return self._generate_result(0, population_size, time.time() - start_time)

            generation = 0
            total_evaluations = population_size

            for generation in range(max_generations):
                # 记录当前代信息
                stats = self.evolution_engine.record_generation_stats(population, generation)
                if stats:
                    # 添加最佳个体参数信息
                    best_individual = min(population, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
                    if best_individual and hasattr(best_individual, 'to_params'):
                        stats['params'] = best_individual.to_params(self.param_names)
                    self.generation_stats.append(stats)

                # 检查收敛
                if self.evolution_engine.check_convergence():
                    logger.info(f"在第{generation}代检测到收敛，提前停止")
                    break
                
                # 检查是否需要重启
                if self.evolution_engine.should_restart(generation):
                    logger.info(f"在第{generation}代执行种群重启")
                    population = self.evolution_engine.restart_population(population, population_size)
                    self._evaluate_population(population)
                    total_evaluations += population_size
                else:
                    # 进化操作
                    new_population = self.evolution_engine.evolve_population(
                        population, generation, max_generations
                    )

                    # 评估新种群中的新个体
                    new_evaluations = self._evaluate_new_individuals(new_population)
                    total_evaluations += new_evaluations

                    population = new_population

                # 更新最佳个体
                self.best_individual = self.evolution_engine.update_best_individual(
                    population, self.best_individual
                )
                self._update_best_ever_individual()

                # 检查评估次数限制
                if total_evaluations >= n_calls:
                    logger.info(f"达到评估次数限制 ({n_calls})，停止优化")
                    break
                
                if self.config and self.config.verbose and generation % 10 == 0:
                    best_fitness = self.best_individual.fitness if self.best_individual else float('inf')
                    diversity_metrics = self.evolution_engine.get_diversity_metrics()
                    current_diversity = diversity_metrics['diversity_history'][-1] if diversity_metrics['diversity_history'] else 0.0
                    logger.info(f"第{generation}代: 最佳适应度={best_fitness:.6f}, "
                              f"多样性={current_diversity:.4f}, 评估次数={total_evaluations}")

            execution_time = time.time() - start_time

            # 最终检查
            if self.best_individual is None:
                logger.warning("优化完成但未找到有效的最佳个体")

            # 生成结果
            result = self._generate_result(generation + 1, total_evaluations, execution_time)

            logger.info(f"遗传算法优化完成: 最佳适应度={result.best_score:.6f}, "
                       f"总代数={generation + 1}, 总评估次数={total_evaluations}")

            return result

        except Exception as e:
            logger.error(f"遗传算法优化过程异常: {e}")
            execution_time = time.time() - start_time
            return self._generate_result(0, 0, execution_time)
    
    def _evaluate_population(self, population: List[Individual]) -> None:
        """评估种群"""
        for individual in population:
            if individual.fitness is None:
                params = individual.to_params(self.param_names)
                try:
                    individual.fitness = self.evaluator.evaluate_mesh(params)
                except Exception as e:
                    logger.warning(f"个体评估失败: {e}")
                    individual.fitness = float('inf')
    
    def _evaluate_new_individuals(self, population: List[Individual]) -> int:
        """评估种群中新的个体，返回评估次数"""
        evaluation_count = 0
        for individual in population:
            if individual.fitness is None:
                params = individual.to_params(self.param_names)
                try:
                    individual.fitness = self.evaluator.evaluate_mesh(params)
                    evaluation_count += 1
                except Exception as e:
                    logger.warning(f"个体评估失败: {e}")
                    individual.fitness = float('inf')
                    evaluation_count += 1
        
        return evaluation_count
    
    def _update_best_ever_individual(self) -> None:
        """更新历史最佳个体"""
        if (self.best_individual and self.best_individual.fitness is not None and
            (self.best_ever_fitness is None or self.best_individual.fitness < self.best_ever_fitness)):
            self.best_ever_fitness = self.best_individual.fitness
            self.best_ever_individual = self.best_individual.copy()
    
    def _generate_result(self, total_generations: int, total_evaluations: int, execution_time: float) -> OptimizationResult:
        """生成优化结果"""
        if self.best_individual is None:
            logger.error("优化过程中未找到有效的最佳个体")
            # 创建一个默认的最佳结果
            default_genes = []
            for low, high in self.bounds:
                default_genes.append((low + high) / 2)  # 使用中点作为默认值
            
            # 创建默认历史记录
            default_history = [{
                'generation': 0,
                'parameters': {self.param_names[i]: default_genes[i] for i in range(len(self.param_names))},
                'result': float('inf'),
                'fitness': float('inf'),
                'stats': None
            }]
            
            return OptimizationResult.from_genetic_result(
                best_params=default_genes,
                best_value=float('inf'),
                optimization_history=default_history,
                parameter_names=self.param_names,
                parameter_ranges=self.bounds,
                generation_stats=self.generation_stats,
                convergence_info={
                    'converged': False,
                    'convergence_generation': -1,
                    'final_diversity': 0.0,
                    'error': 'No valid individuals found during optimization'
                }
            )

        # 构建历史记录
        history = []
        if self.generation_stats:
            for i, stats in enumerate(self.generation_stats):
                history.append({
                    'generation': i,
                    'parameters': self.best_individual.to_params(self.param_names),
                    'result': stats.get('best_fitness', float('inf')),
                    'fitness': stats.get('best_fitness', float('inf')),
                    'stats': stats
                })
        
        # 获取多样性指标
        diversity_metrics = self.evolution_engine.get_diversity_metrics()
        
        return OptimizationResult.from_genetic_result(
            best_params=self.best_individual.genes.copy(),
            best_value=self.best_individual.fitness if self.best_individual.fitness is not None else float('inf'),
            optimization_history=history,
            parameter_names=self.param_names,
            parameter_ranges=self.bounds,
            generation_stats=self.generation_stats,
            convergence_info={
                'converged': self.evolution_engine.convergence_counter >= self.genetic_config.convergence_patience,
                'convergence_generation': total_generations,
                'final_diversity': diversity_metrics['diversity_history'][-1] if diversity_metrics['diversity_history'] else 0.0,
                'total_generations': total_generations,
                'total_evaluations': total_evaluations,
                'execution_time': execution_time,
                'restart_count': self.evolution_engine.restart_count,
                'improvement_ratio': self.evolution_engine.calculate_improvement_ratio()
            }
        )
    
    def plot_evolution(self, save_path: Optional[str] = None, show_diversity: bool = True) -> None:
        """绘制进化过程"""
        if not self.generation_stats:
            logger.warning("没有进化数据可以绘制")
            return
        
        # 获取进化数据
        diversity_metrics = self.evolution_engine.get_diversity_metrics()
        
        # 创建可视化器并绘制
        visualizer = GeneticVisualizer(
            best_fitness_history=diversity_metrics['best_fitness_history'],
            diversity_history=diversity_metrics['diversity_history'],
            generation_stats=self.generation_stats,
            genetic_config=self.genetic_config.to_dict()
        )
        
        visualizer.plot_evolution(save_path=save_path, show_diversity=show_diversity)
    
    def get_diversity_metrics(self) -> Dict[str, List[float]]:
        """计算种群多样性指标"""
        return self.evolution_engine.get_diversity_metrics()
    
    def export_evolution_data(self, filename: str) -> None:
        """导出进化数据"""
        try:
            diversity_metrics = self.evolution_engine.get_diversity_metrics()
            
            best_individual_info = {
                'genes': self.best_individual.genes if self.best_individual else None,
                'fitness': self.best_individual.fitness if self.best_individual else None,
                'params': self.best_individual.to_params(self.param_names) if self.best_individual else None,
                'generation': self.best_individual.generation if self.best_individual else None
            }
            
            metadata = {
                'total_generations': len(self.generation_stats),
                'parameter_names': self.param_names,
                'parameter_bounds': self.bounds,
                'parameter_types': [t.__name__ for t in self.param_types]
            }
            
            export_evolution_data(
                best_fitness_history=diversity_metrics['best_fitness_history'],
                diversity_history=diversity_metrics['diversity_history'],
                generation_stats=self.generation_stats,
                genetic_config=self.genetic_config.to_dict(),
                best_individual_info=best_individual_info,
                metadata=metadata,
                filename=filename
            )
            
        except Exception as e:
            logger.error(f"导出进化数据失败: {e}")


# 工厂函数
def create_genetic_optimizer(param_space, evaluator, config=None, **genetic_kwargs) -> GeneticOptimizer:
    """
    创建遗传算法优化器的工厂函数
    
    Args:
        param_space: 参数空间
        evaluator: 评估器
        config: 优化配置
        **genetic_kwargs: 遗传算法特定参数
        
    Returns:
        配置好的遗传算法优化器
    """
    genetic_config = GeneticConfig(**genetic_kwargs)
    return GeneticOptimizer(param_space, evaluator, config, genetic_config)


# 向后兼容性支持
if __name__ == "__main__":
    # 测试遗传算法优化器
    logger.info("重构版遗传算法优化器测试")
    
    # 创建简单的测试问题
    class TestEvaluator:
        def evaluate_mesh(self, params):
            # Rosenbrock函数
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
    
    # 创建测试优化器
    param_space = TestParamSpace()
    evaluator = TestEvaluator()
    
    genetic_config = GeneticConfig(
        population_size=20,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        adaptive_mutation=True,
        diversity_preservation=True
    )
    
    optimizer = GeneticOptimizer(param_space, evaluator, genetic_config=genetic_config)
    
    # 运行优化
    result = optimizer.optimize(n_calls=100)
    
    # 输出结果
    print(f"最佳参数: {result.best_params}")
    print(f"最佳值: {result.best_score:.6f}")
    convergence_info = result.convergence_info or {}
    print(f"总代数: {convergence_info.get('total_generations', 0)}")
    print(f"重启次数: {convergence_info.get('restart_count', 0)}")
    
    # 绘制进化过程
    optimizer.plot_evolution("test_evolution_refactored.png")
    
    # 导出数据
    optimizer.export_evolution_data("test_evolution_data_refactored.json")
    
    print("重构版遗传算法测试完成!")