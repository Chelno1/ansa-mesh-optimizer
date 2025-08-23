#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法进化操作模块

包含选择、交叉、变异、种群管理等进化操作
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
import logging

from .individual import Individual, create_individual, create_lhs_population, calculate_population_diversity
from .genetic_config import GeneticConfig

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """进化引擎 - 负责种群的进化操作"""
    
    def __init__(self, bounds: List[tuple], param_types: List[type], genetic_config: GeneticConfig):
        """
        初始化进化引擎
        
        Args:
            bounds: 参数边界
            param_types: 参数类型
            genetic_config: 遗传算法配置
        """
        self.bounds = bounds
        self.param_types = param_types
        self.genetic_config = genetic_config
        
        # 收敛检测
        self.convergence_counter = 0
        self.stagnation_counter = 0
        
        # 重启机制
        self.restart_count = 0
        
        # 历史记录
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
    
    def initialize_population(self, population_size: int) -> List[Individual]:
        """
        初始化种群 - 增强版本
        
        Args:
            population_size: 种群大小
            
        Returns:
            初始化的种群
        """
        population = []
        
        # 使用拉丁超立方抽样初始化一部分个体
        try:
            lhs_size = population_size // 2
            lhs_population = create_lhs_population(lhs_size, self.bounds, self.param_types)
            population.extend(lhs_population)
        except ImportError:
            logger.warning("scipy不可用，使用随机初始化")
        
        # 剩余个体使用随机初始化
        while len(population) < population_size:
            individual = create_individual(self.bounds, self.param_types)
            population.append(individual)
        
        return population
    
    def evolve_population(self, population: List[Individual], generation: int, 
                         max_generations: int) -> List[Individual]:
        """
        进化种群 - 增强版本
        
        Args:
            population: 当前种群
            generation: 当前代数
            max_generations: 最大代数
            
        Returns:
            进化后的种群
        """
        # 排序种群
        population.sort()
        
        # 计算种群多样性
        diversity = calculate_population_diversity(population)
        
        # 保留精英
        elite_size = self.genetic_config.elite_size
        new_population = [individual.copy() for individual in population[:elite_size]]
        
        # 多样性保持机制
        if self.genetic_config.diversity_preservation and diversity < 0.1:
            # 如果多样性太低，增加变异率
            mutation_rate = min(0.5, self.genetic_config.mutation_rate * 2)
            logger.debug(f"低多样性检测，增加变异率至 {mutation_rate:.3f}")
        else:
            mutation_rate = self.genetic_config.mutation_rate
        
        # 生成后代
        while len(new_population) < len(population):
            # 选择父母
            parent1 = self.selection(population)
            parent2 = self.selection(population)
            
            # 确保父母不同（如果可能）
            attempts = 0
            while parent1 is parent2 and attempts < 10:
                parent2 = self.selection(population)
                attempts += 1
            
            # 交叉
            child1, child2 = parent1.crossover(parent2, self.genetic_config.crossover_rate)
            
            # 变异
            child1.mutate(mutation_rate, generation, max_generations)
            child2.mutate(mutation_rate, generation, max_generations)
            
            # 设置代数
            child1.generation = generation + 1
            child2.generation = generation + 1
            
            new_population.extend([child1, child2])
        
        # 确保种群大小
        return new_population[:len(population)]
    
    def selection(self, population: List[Individual]) -> Individual:
        """选择操作 - 锦标赛选择"""
        tournament_size = min(self.genetic_config.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return min(tournament)  # 返回适应度最好的个体
    
    def roulette_wheel_selection(self, population: List[Individual]) -> Individual:
        """轮盘赌选择"""
        # 计算适应度值（转换为最大化问题）
        fitness_values = [1.0 / (1.0 + ind.fitness) if ind.fitness is not None else 0.0 
                         for ind in population]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return random.choice(population)
        
        # 轮盘赌选择
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_values):
            current += fitness
            if current >= pick:
                return population[i]
        
        return population[-1]  # 备用选择
    
    def rank_selection(self, population: List[Individual]) -> Individual:
        """排名选择"""
        sorted_population = sorted(population)
        n = len(population)
        
        # 线性排名概率
        ranks = list(range(1, n + 1))
        total_rank = sum(ranks)
        
        pick = random.uniform(0, total_rank)
        current = 0
        for i, rank in enumerate(ranks):
            current += rank
            if current >= pick:
                return sorted_population[n - 1 - i]  # 排名越高，适应度越好
        
        return sorted_population[0]  # 最好的个体
    
    def update_best_individual(self, population: List[Individual], 
                              best_individual: Optional[Individual]) -> Individual:
        """更新最佳个体"""
        current_best = min(population)
        
        # 更新当前代最佳
        if (best_individual is None or
            (current_best.fitness is not None and
             (best_individual.fitness is None or current_best.fitness < best_individual.fitness))):
            best_individual = current_best.copy()
        
        return best_individual
    
    def record_generation_stats(self, population: List[Individual], generation: int) -> Dict[str, Any]:
        """
        记录当前代统计信息
        
        Args:
            population: 当前种群
            generation: 代数
            
        Returns:
            统计信息字典
        """
        fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
        
        if fitness_values:
            diversity = calculate_population_diversity(population)
            best_individual = min(population, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
            
            stats = {
                'generation': generation,
                'best_fitness': min(fitness_values),
                'worst_fitness': max(fitness_values),
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'diversity': diversity,
                'population_size': len(population),
                'convergence_counter': self.convergence_counter,
                # 添加可视化需要的字段
                'score': min(fitness_values),  # 兼容可视化器
                'params': {}  # 需要在外部填充
            }
            
            # 更新历史记录
            self.best_fitness_history.append(stats['best_fitness'])
            self.diversity_history.append(diversity)
            
            # 限制历史记录长度
            if len(self.best_fitness_history) > self.genetic_config.max_history_size:
                self.best_fitness_history.pop(0)
                self.diversity_history.pop(0)
            
            return stats
        
        return {}
    
    def check_convergence(self) -> bool:
        """检查收敛性 - 增强版本"""
        if len(self.best_fitness_history) < self.genetic_config.convergence_patience:
            return False
        
        # 检查最近几代的改进
        recent_best = self.best_fitness_history[-self.genetic_config.convergence_patience:]
        variance = np.var(recent_best)
        
        if variance < self.genetic_config.convergence_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        
        # 检查停滞
        if len(self.best_fitness_history) >= 2:
            if abs(self.best_fitness_history[-1] - self.best_fitness_history[-2]) < self.genetic_config.convergence_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # 收敛条件
        return (self.convergence_counter >= self.genetic_config.convergence_patience or 
                self.stagnation_counter >= self.genetic_config.convergence_patience * 2)
    
    def should_restart(self, generation: int) -> bool:
        """检查是否应该重启种群"""
        if not self.genetic_config.restart_enabled:
            return False
        
        # 在特定代数间隔重启
        if generation > 0 and generation % self.genetic_config.restart_generations == 0:
            # 检查是否停滞
            if self.stagnation_counter >= self.genetic_config.restart_generations // 2:
                return True
        
        return False
    
    def restart_population(self, population: List[Individual], population_size: int) -> List[Individual]:
        """重启种群"""
        # 保留最佳个体
        elite_count = max(1, self.genetic_config.elite_size // 2)
        population.sort()
        new_population = [individual.copy() for individual in population[:elite_count]]
        
        # 重新初始化剩余个体
        while len(new_population) < population_size:
            individual = create_individual(self.bounds, self.param_types)
            new_population.append(individual)
        
        # 重置计数器
        self.convergence_counter = 0
        self.stagnation_counter = 0
        self.restart_count += 1
        
        logger.info(f"种群重启完成，保留 {elite_count} 个精英个体")
        
        return new_population
    
    def calculate_improvement_ratio(self) -> float:
        """计算改进比率"""
        if len(self.best_fitness_history) > 1:
            initial_fitness = self.best_fitness_history[0]
            final_fitness = self.best_fitness_history[-1]
            return (initial_fitness - final_fitness) / initial_fitness if initial_fitness != 0 else 0
        return 0.0
    
    def get_diversity_metrics(self) -> Dict[str, List[float]]:
        """计算种群多样性指标"""
        return {
            'diversity_history': self.diversity_history.copy(),
            'best_fitness_history': self.best_fitness_history.copy()
        }


def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    """
    锦标赛选择函数
    
    Args:
        population: 种群
        tournament_size: 锦标赛大小
        
    Returns:
        选中的个体
    """
    tournament_size = min(tournament_size, len(population))
    tournament = random.sample(population, tournament_size)
    return min(tournament)


def uniform_crossover(parent1: Individual, parent2: Individual, 
                     crossover_rate: float = 0.8) -> Tuple[Individual, Individual]:
    """
    均匀交叉操作
    
    Args:
        parent1: 父代1
        parent2: 父代2
        crossover_rate: 交叉率
        
    Returns:
        两个子代个体
    """
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    child1_genes = []
    child2_genes = []
    
    for i in range(len(parent1.genes)):
        if random.random() < 0.5:
            child1_genes.append(parent1.genes[i])
            child2_genes.append(parent2.genes[i])
        else:
            child1_genes.append(parent2.genes[i])
            child2_genes.append(parent1.genes[i])
    
    child1 = Individual(child1_genes, parent1.bounds, parent1.param_types)
    child2 = Individual(child2_genes, parent1.bounds, parent1.param_types)
    
    return child1, child2


def gaussian_mutation(individual: Individual, mutation_rate: float = 0.1, 
                     mutation_strength: float = 0.1) -> None:
    """
    高斯变异操作
    
    Args:
        individual: 个体
        mutation_rate: 变异率
        mutation_strength: 变异强度
    """
    for i in range(len(individual.genes)):
        if random.random() < mutation_rate:
            low, high = individual.bounds[i]
            
            if individual.param_types[i] == int:
                # 整数变异
                range_size = max(1, int((high - low) * mutation_strength))
                delta = random.randint(-range_size, range_size)
                individual.genes[i] += delta
            else:
                # 实数变异
                sigma = (high - low) * mutation_strength
                individual.genes[i] += random.gauss(0, sigma)
    
    individual._constrain_genes()
    individual.fitness = None  # 重置适应度
