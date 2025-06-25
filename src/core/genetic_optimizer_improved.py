#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的遗传算法优化器 - 增强版本

作者: Chel
创建日期: 2025-06-19
版本: 1.2.0
更新日期: 2025-06-20
修复: 内存优化，收敛检测，多样性保持
"""

import numpy as np
import random
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 安全导入matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib不可用，无法生成图表")

# 尝试导入字体装饰器模块
try:
    from utils.font_decorator import with_chinese_font, plotting_ready
    DECORATOR_AVAILABLE = True
except ImportError:
    logger.warning("字体装饰器模块未找到")
    DECORATOR_AVAILABLE = False
    
    # 创建空装饰器作为备用
    def with_chinese_font(func):
        return func
    
    def plotting_ready(**kwargs):
        def decorator(func):
            return func
        return decorator

@dataclass
class GeneticConfig:
    """遗传算法配置 - 增强版本"""
    population_size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    max_generations: int = 100
    convergence_threshold: float = 1e-6
    convergence_patience: int = 10
    
    # 新增配置选项
    adaptive_mutation: bool = True
    diversity_preservation: bool = True
    niching_enabled: bool = False
    restart_enabled: bool = True
    restart_generations: int = 20
    
    # 内存管理
    max_history_size: int = 50
    save_full_history: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        if self.population_size < 4:
            errors.append("population_size must be at least 4")
        if self.elite_size >= self.population_size:
            errors.append("elite_size must be less than population_size")
        if not 0 <= self.mutation_rate <= 1:
            errors.append("mutation_rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            errors.append("crossover_rate must be between 0 and 1")
        if self.tournament_size > self.population_size:
            errors.append("tournament_size must not exceed population_size")
        if self.convergence_threshold < 0:
            errors.append("convergence_threshold must be non-negative")
        if self.convergence_patience <= 0:
            errors.append("convergence_patience must be positive")
        
        return len(errors) == 0, errors

class Individual:
    """个体类 - 优化版本"""
    
    __slots__ = ['genes', 'bounds', 'param_types', 'fitness', 'age', 'generation']
    
    def __init__(self, genes: List[float], bounds: List[Tuple[float, float]], param_types: List[type]):
        """
        初始化个体
        
        Args:
            genes: 基因列表
            bounds: 参数边界
            param_types: 参数类型
        """
        self.genes = genes.copy()
        self.bounds = bounds
        self.param_types = param_types
        self.fitness: Optional[float] = None
        self.age: int = 0
        self.generation: int = 0
        
        # 确保基因在合法范围内
        self._constrain_genes()
    
    def _constrain_genes(self) -> None:
        """约束基因在合法范围内"""
        for i, (gene, (low, high), param_type) in enumerate(zip(self.genes, self.bounds, self.param_types)):
            if param_type == int:
                self.genes[i] = max(low, min(high, round(gene)))
            else:
                self.genes[i] = max(low, min(high, gene))
    
    def to_params(self, param_names: List[str]) -> Dict[str, float]:
        """转换为参数字典"""
        params = {}
        for i, name in enumerate(param_names):
            if self.param_types[i] == int:
                params[name] = int(round(self.genes[i]))
            else:
                params[name] = self.genes[i]
        return params
    
    def mutate(self, mutation_rate: float, generation: int = 0, max_generations: int = 100) -> None:
        """
        变异操作 - 自适应版本
        
        Args:
            mutation_rate: 基础变异率
            generation: 当前代数
            max_generations: 最大代数
        """
        # 自适应变异率：随代数增加而减少
        adaptive_rate = mutation_rate * (1 - generation / max_generations) ** 0.5
        
        for i in range(len(self.genes)):
            if random.random() < adaptive_rate:
                low, high = self.bounds[i]
                
                if self.param_types[i] == int:
                    # 整数变异
                    range_size = max(1, int((high - low) * 0.1))
                    delta = random.randint(-range_size, range_size)
                    self.genes[i] += delta
                else:
                    # 实数变异（自适应高斯变异）
                    mutation_strength = (high - low) * 0.1 * (1 - generation / max_generations)
                    self.genes[i] += random.gauss(0, mutation_strength)
        
        self._constrain_genes()
        self.fitness = None  # 重置适应度
        self.age += 1
    
    def crossover(self, other: 'Individual', crossover_rate: float) -> Tuple['Individual', 'Individual']:
        """
        交叉操作 - 增强版本
        
        Args:
            other: 另一个个体
            crossover_rate: 交叉率
            
        Returns:
            两个子代个体
        """
        if random.random() > crossover_rate:
            return Individual(self.genes, self.bounds, self.param_types), \
                   Individual(other.genes, self.bounds, self.param_types)
        
        # 模拟二进制交叉（SBX）用于实数参数
        child1_genes = []
        child2_genes = []
        
        for i in range(len(self.genes)):
            if self.param_types[i] == int:
                # 整数参数使用均匀交叉
                if random.random() < 0.5:
                    child1_genes.append(self.genes[i])
                    child2_genes.append(other.genes[i])
                else:
                    child1_genes.append(other.genes[i])
                    child2_genes.append(self.genes[i])
            else:
                # 实数参数使用SBX交叉
                p1, p2 = self.genes[i], other.genes[i]
                low, high = self.bounds[i]
                
                if abs(p1 - p2) > 1e-14:
                    # SBX交叉
                    if p1 > p2:
                        p1, p2 = p2, p1
                    
                    # 分布指数
                    eta = 2.0
                    u = random.random()
                    
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                    
                    # 确保在边界内
                    c1 = max(low, min(high, c1))
                    c2 = max(low, min(high, c2))
                    
                    child1_genes.append(c1)
                    child2_genes.append(c2)
                else:
                    child1_genes.append(p1)
                    child2_genes.append(p2)
        
        child1 = Individual(child1_genes, self.bounds, self.param_types)
        child2 = Individual(child2_genes, self.bounds, self.param_types)
        
        return child1, child2
    
    def distance_to(self, other: 'Individual') -> float:
        """计算与另一个个体的距离"""
        total_distance = 0.0
        for i, ((gene1, gene2), (low, high)) in enumerate(zip(
            zip(self.genes, other.genes), self.bounds)):
            # 标准化距离
            normalized_distance = abs(gene1 - gene2) / (high - low) if high > low else 0
            total_distance += normalized_distance ** 2
        
        return total_distance ** 0.5
    
    def __lt__(self, other: 'Individual') -> bool:
        """比较操作（用于排序）"""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness < other.fitness
    
    def copy(self) -> 'Individual':
        """创建个体的深拷贝"""
        new_individual = Individual(self.genes, self.bounds, self.param_types)
        new_individual.fitness = self.fitness
        new_individual.age = self.age
        new_individual.generation = self.generation
        return new_individual

class GeneticOptimizer:
    """遗传算法优化器 - 增强版本"""
    
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
        
        self.bounds = param_space.get_bounds()
        self.param_types = param_space.get_param_types()
        self.param_names = param_space.get_param_names()
        
        # 优化历史（内存优化）
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.best_individual: Optional[Individual] = None
        
        # 收敛检测
        self.convergence_counter = 0
        self.stagnation_counter = 0
        
        # 重启机制
        self.restart_count = 0
        self.best_ever_fitness = float('inf')
        self.best_ever_individual: Optional[Individual] = None
        
        # 设置随机种子
        if config and hasattr(config, 'random_state'):
            random.seed(config.random_state)
            np.random.seed(config.random_state)
        
        logger.info(f"遗传算法优化器初始化完成 - 种群大小: {self.genetic_config.population_size}")
    
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """
        执行遗传算法优化
        
        Args:
            n_calls: 总评估次数
            **kwargs: 其他参数
            
        Returns:
            优化结果字典
        """
        # 根据评估次数调整种群大小和代数
        population_size = min(self.genetic_config.population_size, max(10, n_calls // 5))
        max_generations = min(self.genetic_config.max_generations, n_calls // population_size)
        
        logger.info(f"遗传算法优化开始: 种群大小={population_size}, 最大代数={max_generations}")
        
        start_time = time.time()
        
        try:
            # 初始化种群
            population = self._initialize_population(population_size)

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
                self._record_generation_stats(population, generation)

                # 检查收敛
                if self._check_convergence():
                    logger.info(f"在第{generation}代检测到收敛，提前停止")
                    break
                
                # 检查是否需要重启
                if self._should_restart(generation):
                    logger.info(f"在第{generation}代执行种群重启")
                    population = self._restart_population(population, population_size)
                    self._evaluate_population(population)
                    total_evaluations += population_size
                    self.restart_count += 1
                else:
                    # 进化操作
                    new_population = self._evolve_population(population, generation, max_generations)

                    # 评估新种群中的新个体
                    new_evaluations = self._evaluate_new_individuals(new_population)
                    total_evaluations += new_evaluations

                    population = new_population

                # 更新最佳个体
                self._update_best_individual(population)

                # 检查评估次数限制
                if total_evaluations >= n_calls:
                    logger.info(f"达到评估次数限制 ({n_calls})，停止优化")
                    break
                
                if self.config and self.config.verbose and generation % 10 == 0:
                    best_fitness = self.best_individual.fitness if self.best_individual else float('inf')
                    diversity = self._calculate_population_diversity(population)
                    logger.info(f"第{generation}代: 最佳适应度={best_fitness:.6f}, "
                              f"多样性={diversity:.4f}, 评估次数={total_evaluations}")

            execution_time = time.time() - start_time

            # 最终检查
            if self.best_individual is None:
                logger.warning("优化完成但未找到有效的最佳个体")

            # 生成结果
            result = self._generate_result(generation + 1, total_evaluations, execution_time)

            logger.info(f"遗传算法优化完成: 最佳适应度={result['best_value']:.6f}, "
                       f"总代数={generation + 1}, 总评估次数={total_evaluations}")

            return result

        except Exception as e:
            logger.error(f"遗传算法优化过程异常: {e}")
            execution_time = time.time() - start_time
            return self._generate_result(0, 0, execution_time)
                                                             
    def _initialize_population(self, population_size: int) -> List[Individual]:
        """初始化种群 - 增强版本"""
        population = []
        
        # 使用拉丁超立方抽样初始化一部分个体
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=len(self.bounds), seed=42)
            lhs_samples = sampler.random(population_size // 2)
            
            # 缩放到参数边界
            for sample in lhs_samples:
                genes = []
                for i, (s, (low, high), param_type) in enumerate(zip(sample, self.bounds, self.param_types)):
                    if param_type == int:
                        gene = int(low + s * (high - low))
                    else:
                        gene = low + s * (high - low)
                    genes.append(gene)
                
                individual = Individual(genes, self.bounds, self.param_types)
                population.append(individual)
        
        except ImportError:
            logger.warning("scipy不可用，使用随机初始化")
        
        # 剩余个体使用随机初始化
        while len(population) < population_size:
            genes = []
            for (low, high), param_type in zip(self.bounds, self.param_types):
                if param_type == int:
                    gene = random.randint(low, high)
                else:
                    gene = random.uniform(low, high)
                genes.append(gene)
            
            individual = Individual(genes, self.bounds, self.param_types)
            population.append(individual)
        
        return population
    
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
    
    def _evolve_population(self, population: List[Individual], generation: int, max_generations: int) -> List[Individual]:
        """进化种群 - 增强版本"""
        # 排序种群
        population.sort()
        
        # 计算种群多样性
        diversity = self._calculate_population_diversity(population)
        
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
            parent1 = self._selection(population)
            parent2 = self._selection(population)
            
            # 确保父母不同（如果可能）
            attempts = 0
            while parent1 is parent2 and attempts < 10:
                parent2 = self._selection(population)
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
    
    def _selection(self, population: List[Individual]) -> Individual:
        """选择操作 - 锦标赛选择"""
        tournament_size = min(self.genetic_config.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return min(tournament)  # 返回适应度最好的个体
    
    def _update_best_individual(self, population: List[Individual]) -> None:
        """更新最佳个体"""
        current_best = min(population)
        
        # 更新当前代最佳
        if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best.copy()
        
        # 更新历史最佳
        if current_best.fitness < self.best_ever_fitness:
            self.best_ever_fitness = current_best.fitness
            self.best_ever_individual = current_best.copy()
    
    def _record_generation_stats(self, population: List[Individual], generation: int) -> None:
        """记录当前代统计信息 - 内存优化版本"""
        fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
        
        if fitness_values:
            diversity = self._calculate_population_diversity(population)
            
            stats = {
                'generation': generation,
                'best_fitness': min(fitness_values),
                'worst_fitness': max(fitness_values),
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'diversity': diversity,
                'population_size': len(population),
                'convergence_counter': self.convergence_counter
            }
            
            # 限制历史记录长度
            if len(self.generation_stats) >= self.genetic_config.max_history_size:
                self.generation_stats.pop(0)
            
            self.generation_stats.append(stats)
            self.best_fitness_history.append(stats['best_fitness'])
            self.diversity_history.append(diversity)
            
            # 限制历史列表长度
            if len(self.best_fitness_history) > self.genetic_config.max_history_size:
                self.best_fitness_history.pop(0)
                self.diversity_history.pop(0)
    
    def _calculate_population_diversity(self, population: List[Individual]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        # 计算种群中个体间的平均距离
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                total_distance += population[i].distance_to(population[j])
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _check_convergence(self) -> bool:
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
    
    def _should_restart(self, generation: int) -> bool:
        """检查是否应该重启种群"""
        if not self.genetic_config.restart_enabled:
            return False
        
        # 在特定代数间隔重启
        if generation > 0 and generation % self.genetic_config.restart_generations == 0:
            # 检查是否停滞
            if self.stagnation_counter >= self.genetic_config.restart_generations // 2:
                return True
        
        return False
    
    def _restart_population(self, population: List[Individual], population_size: int) -> List[Individual]:
        """重启种群"""
        # 保留最佳个体
        elite_count = max(1, self.genetic_config.elite_size // 2)
        population.sort()
        new_population = [individual.copy() for individual in population[:elite_count]]
        
        # 重新初始化剩余个体
        while len(new_population) < population_size:
            genes = []
            for (low, high), param_type in zip(self.bounds, self.param_types):
                if param_type == int:
                    gene = random.randint(low, high)
                else:
                    gene = random.uniform(low, high)
                genes.append(gene)
            
            individual = Individual(genes, self.bounds, self.param_types)
            new_population.append(individual)
        
        # 重置计数器
        self.convergence_counter = 0
        self.stagnation_counter = 0
        
        logger.info(f"种群重启完成，保留 {elite_count} 个精英个体")
        
        return new_population
    
    def _generate_result(self, total_generations: int, total_evaluations: int, execution_time: float) -> Dict[str, Any]:
        """生成优化结果"""
        if self.best_individual is None:
            logger.error("优化过程中未找到有效的最佳个体")
            # 创建一个默认的最佳结果
            default_params = {}
            param_names = self.param_names
            bounds = self.bounds

            for i, name in enumerate(param_names):
                low, high = bounds[i]
                default_params[name] = (low + high) / 2  # 使用中点作为默认值

            return {
                'best_params': default_params,
                'best_value': float('inf'),
                'optimizer_name': 'Genetic Algorithm',
                'total_generations': total_generations,
                'total_evaluations': total_evaluations,
                'execution_time': execution_time,
                'population_size': self.genetic_config.population_size,
                'convergence_detected': False,
                'restart_count': self.restart_count,
                'improvement_ratio': 0.0,
                'final_diversity': 0.0,
                'genetic_result': self,
                'error': 'No valid individuals found during optimization'
            }

        best_params = self.best_individual.to_params(self.param_names)
        
        # 计算改进统计
        if len(self.best_fitness_history) > 1:
            initial_fitness = self.best_fitness_history[0]
            final_fitness = self.best_fitness_history[-1]
            improvement = (initial_fitness - final_fitness) / initial_fitness if initial_fitness != 0 else 0
        else:
            improvement = 0
        
        return {
            'best_params': best_params,
            'best_value': self.best_individual.fitness,
            'optimizer_name': 'Genetic Algorithm',
            'total_generations': total_generations,
            'total_evaluations': total_evaluations,
            'execution_time': execution_time,
            'population_size': self.genetic_config.population_size,
            'convergence_detected': self.convergence_counter >= self.genetic_config.convergence_patience,
            'restart_count': self.restart_count,
            'improvement_ratio': improvement,
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'genetic_result': self  # 返回完整的遗传算法结果对象
        }
    
    @plotting_ready(backend='TkAgg', save_original=True)
    def plot_evolution(self, save_path: Optional[str] = None, show_diversity: bool = True) -> None:
        """绘制进化过程 - 使用增强装饰器"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib不可用，无法绘制进化图表")
            return
        
        try:
            fig_size = (15, 10) if show_diversity else (12, 8)
            fig, axes = plt.subplots(2, 2, figsize=fig_size)
            
            # 最佳适应度变化
            axes[0, 0].plot(self.best_fitness_history, 'b-', linewidth=2, label='最佳适应度')
            axes[0, 0].set_xlabel('代数')
            axes[0, 0].set_ylabel('适应度')
            axes[0, 0].set_title('最佳适应度进化曲线')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # 添加重启点标记
            if self.restart_count > 0:
                restart_points = []
                for i, gen_stat in enumerate(self.generation_stats):
                    if i > 0 and gen_stat['generation'] % self.genetic_config.restart_generations == 0:
                        restart_points.append(i)
                
                for point in restart_points:
                    if point < len(self.best_fitness_history):
                        axes[0, 0].axvline(x=point, color='red', linestyle='--', alpha=0.7, label='重启点')
            
            # 种群多样性变化
            if show_diversity and self.diversity_history:
                axes[0, 1].plot(self.diversity_history, 'g-', linewidth=2, label='种群多样性')
                axes[0, 1].set_xlabel('代数')
                axes[0, 1].set_ylabel('多样性')
                axes[0, 1].set_title('种群多样性变化')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # 适应度统计（最后一代）
            if self.generation_stats:
                final_stats = self.generation_stats[-1]
                
                # 模拟最终种群的适应度分布
                mean_fitness = final_stats['mean_fitness']
                std_fitness = final_stats['std_fitness']
                
                # 生成模拟分布数据
                simulated_fitness = np.random.normal(mean_fitness, std_fitness, 100)
                simulated_fitness = np.clip(simulated_fitness, final_stats['best_fitness'], final_stats['worst_fitness'])
                
                axes[1, 0].hist(simulated_fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 0].axvline(x=final_stats['best_fitness'], color='red', linestyle='--', 
                                 linewidth=2, label=f'最佳值: {final_stats["best_fitness"]:.4f}')
                axes[1, 0].axvline(x=mean_fitness, color='orange', linestyle='--', 
                                 linewidth=2, label=f'平均值: {mean_fitness:.4f}')
                axes[1, 0].set_xlabel('适应度')
                axes[1, 0].set_ylabel('频次')
                axes[1, 0].set_title('最终种群适应度分布（模拟）')
                axes[1, 0].legend()
            
            # 收敛性分析
            if len(self.best_fitness_history) > 10:
                # 计算滚动改进率
                window_size = min(10, len(self.best_fitness_history) // 4)
                improvement_rates = []
                
                for i in range(window_size, len(self.best_fitness_history)):
                    old_best = self.best_fitness_history[i - window_size]
                    new_best = self.best_fitness_history[i]
                    if old_best != 0:
                        improvement = (old_best - new_best) / old_best
                        improvement_rates.append(improvement)
                    else:
                        improvement_rates.append(0)
                
                axes[1, 1].plot(improvement_rates, 'purple', linewidth=2, label='改进率')
                axes[1, 1].set_xlabel('代数')
                axes[1, 1].set_ylabel('改进率')
                axes[1, 1].set_title(f'滚动改进率 (窗口大小: {window_size})')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"进化图表已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"绘制进化图表失败: {e}")
    
    def get_diversity_metrics(self) -> Dict[str, List[float]]:
        """计算种群多样性指标"""
        return {
            'diversity_history': self.diversity_history.copy(),
            'generation_stats': [stat['diversity'] for stat in self.generation_stats if 'diversity' in stat]
        }
    
    def export_evolution_data(self, filename: str) -> None:
        """导出进化数据 - 增强版本"""
        try:
            import json
            
            export_data = {
                'config': {
                    'population_size': self.genetic_config.population_size,
                    'mutation_rate': self.genetic_config.mutation_rate,
                    'crossover_rate': self.genetic_config.crossover_rate,
                    'tournament_size': self.genetic_config.tournament_size,
                    'max_generations': self.genetic_config.max_generations,
                    'adaptive_mutation': self.genetic_config.adaptive_mutation,
                    'diversity_preservation': self.genetic_config.diversity_preservation,
                    'restart_enabled': self.genetic_config.restart_enabled
                },
                'results': {
                    'best_fitness_history': self.best_fitness_history,
                    'diversity_history': self.diversity_history,
                    'generation_stats': self.generation_stats,
                    'restart_count': self.restart_count,
                    'convergence_counter': self.convergence_counter,
                    'stagnation_counter': self.stagnation_counter
                },
                'best_individual': {
                    'genes': self.best_individual.genes if self.best_individual else None,
                    'fitness': self.best_individual.fitness if self.best_individual else None,
                    'params': self.best_individual.to_params(self.param_names) if self.best_individual else None,
                    'generation': self.best_individual.generation if self.best_individual else None
                },
                'metadata': {
                    'total_generations': len(self.generation_stats),
                    'parameter_names': self.param_names,
                    'parameter_bounds': self.bounds,
                    'parameter_types': [t.__name__ for t in self.param_types],
                    'export_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"进化数据已导出: {filename}")
            
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

if __name__ == "__main__":
    # 测试遗传算法优化器
    logger.info("遗传算法优化器测试")
    
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
    
    print(f"最佳参数: {result['best_params']}")
    print(f"最佳值: {result['best_value']:.6f}")
    print(f"总代数: {result['total_generations']}")
    print(f"重启次数: {result['restart_count']}")
    
    # 绘制进化过程
    optimizer.plot_evolution("test_evolution.png")
    
    # 导出数据
    optimizer.export_evolution_data("test_evolution_data.json")
    
    print("遗传算法测试完成!")