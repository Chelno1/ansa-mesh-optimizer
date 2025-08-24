#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法进化循环模块

负责处理遗传算法的核心迭代循环逻辑，包括种群进化、评估管理和收敛检测。

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple

from .individual import Individual
from .evolution import EvolutionEngine

logger = logging.getLogger(__name__)


class EvolutionLoop:
    """遗传算法进化循环管理器"""
    
    def __init__(self, evolution_engine: EvolutionEngine, evaluator, param_names: List[str]):
        """
        初始化进化循环管理器
        
        Args:
            evolution_engine: 进化引擎实例
            evaluator: 评估器实例
            param_names: 参数名称列表
        """
        self.evolution_engine = evolution_engine
        self.evaluator = evaluator
        self.param_names = param_names
        
        # 优化状态
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None
        self.best_ever_fitness = float('inf')
        self.best_ever_individual: Optional[Individual] = None
        
        logger.debug("进化循环管理器初始化完成")
    
    def run_evolution(
        self, 
        n_calls: int, 
        population_size: int, 
        max_generations: int,
        verbose: bool = False
    ) -> Tuple[int, int, float]:
        """
        执行完整的进化循环
        
        Args:
            n_calls: 总评估次数限制
            population_size: 种群大小
            max_generations: 最大代数
            verbose: 是否输出详细信息
            
        Returns:
            (最终代数, 总评估次数, 执行时间)
        """
        start_time = time.time()
        
        # 初始化种群
        population = self.evolution_engine.initialize_population(population_size)
        
        # 评估初始种群
        self._evaluate_population(population)
        
        # 检查初始种群是否有有效个体
        valid_individuals = [ind for ind in population if ind.fitness != float('inf')]
        if not valid_individuals:
            logger.warning("初始种群中没有有效个体")
            return 0, population_size, time.time() - start_time
        
        generation = 0
        total_evaluations = population_size
        
        # 主进化循环
        for generation in range(max_generations):
            # 记录当前代统计信息
            self._record_generation_stats(population, generation)
            
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
                # 执行进化操作
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
            
            # 输出进度信息
            if verbose and generation % 10 == 0:
                self._log_progress(generation, total_evaluations)
        
        execution_time = time.time() - start_time
        return generation + 1, total_evaluations, execution_time
    
    def _evaluate_population(self, population: List[Individual]) -> None:
        """
        评估整个种群
        
        Args:
            population: 要评估的种群
        """
        for individual in population:
            if individual.fitness is None:
                self._evaluate_individual(individual)
    
    def _evaluate_new_individuals(self, population: List[Individual]) -> int:
        """
        评估种群中新的个体
        
        Args:
            population: 种群
            
        Returns:
            实际评估的个体数量
        """
        evaluation_count = 0
        for individual in population:
            if individual.fitness is None:
                self._evaluate_individual(individual)
                evaluation_count += 1
        
        return evaluation_count
    
    def _evaluate_individual(self, individual: Individual) -> None:
        """
        评估单个个体
        
        Args:
            individual: 要评估的个体
        """
        params = individual.to_params(self.param_names)
        try:
            individual.fitness = self.evaluator.evaluate_mesh(params)
        except Exception as e:
            logger.warning(f"个体评估失败: {e}")
            individual.fitness = float('inf')
    
    def _record_generation_stats(self, population: List[Individual], generation: int) -> None:
        """
        记录当前代的统计信息
        
        Args:
            population: 当前种群
            generation: 当前代数
        """
        stats = self.evolution_engine.record_generation_stats(population, generation)
        if stats:
            # 添加最佳个体参数信息
            best_individual = min(
                population, 
                key=lambda x: x.fitness if x.fitness is not None else float('inf')
            )
            if best_individual and hasattr(best_individual, 'to_params'):
                stats['params'] = best_individual.to_params(self.param_names)
            self.generation_stats.append(stats)
    
    def _update_best_ever_individual(self) -> None:
        """更新历史最佳个体"""
        if (self.best_individual and self.best_individual.fitness is not None and
            (self.best_ever_fitness is None or self.best_individual.fitness < self.best_ever_fitness)):
            self.best_ever_fitness = self.best_individual.fitness
            self.best_ever_individual = self.best_individual.copy()
    
    def _log_progress(self, generation: int, total_evaluations: int) -> None:
        """
        记录进度信息
        
        Args:
            generation: 当前代数
            total_evaluations: 总评估次数
        """
        best_fitness = self.best_individual.fitness if self.best_individual else float('inf')
        diversity_metrics = self.evolution_engine.get_diversity_metrics()
        current_diversity = (
            diversity_metrics['diversity_history'][-1] 
            if diversity_metrics['diversity_history'] else 0.0
        )
        
        logger.info(
            f"第{generation}代: 最佳适应度={best_fitness:.6f}, "
            f"多样性={current_diversity:.4f}, 评估次数={total_evaluations}"
        )
    
    def get_generation_stats(self) -> List[Dict[str, Any]]:
        """获取代数统计信息"""
        return self.generation_stats.copy()
    
    def get_best_individual(self) -> Optional[Individual]:
        """获取当前最佳个体"""
        return self.best_individual
    
    def get_best_ever_individual(self) -> Optional[Individual]:
        """获取历史最佳个体"""
        return self.best_ever_individual
    
    def reset(self) -> None:
        """重置进化循环状态"""
        self.generation_stats.clear()
        self.best_individual = None
        self.best_ever_fitness = float('inf')
        self.best_ever_individual = None
        logger.debug("进化循环状态已重置")