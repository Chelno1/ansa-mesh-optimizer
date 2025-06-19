#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的遗传算法优化器

作者: Chel
创建日期: 2025-06-19
版本: 1.1.0
"""

import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class GeneticConfig:
    """遗传算法配置"""
    population_size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    max_generations: int = 100
    convergence_threshold: float = 1e-6
    convergence_patience: int = 10

class Individual:
    """个体类"""
    
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
    
    def mutate(self, mutation_rate: float) -> None:
        """变异操作"""
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                low, high = self.bounds[i]
                
                if self.param_types[i] == int:
                    # 整数变异
                    self.genes[i] += random.randint(-1, 1)
                else:
                    # 实数变异（高斯变异）
                    mutation_strength = (high - low) * 0.1
                    self.genes[i] += random.gauss(0, mutation_strength)
        
        self._constrain_genes()
        self.fitness = None  # 重置适应度
    
    def crossover(self, other: 'Individual', crossover_rate: float) -> Tuple['Individual', 'Individual']:
        """交叉操作"""
        if random.random() > crossover_rate:
            return Individual(self.genes, self.bounds, self.param_types), \
                   Individual(other.genes, self.bounds, self.param_types)
        
        # 均匀交叉
        child1_genes = []
        child2_genes = []
        
        for i in range(len(self.genes)):
            if random.random() < 0.5:
                child1_genes.append(self.genes[i])
                child2_genes.append(other.genes[i])
            else:
                child1_genes.append(other.genes[i])
                child2_genes.append(self.genes[i])
        
        child1 = Individual(child1_genes, self.bounds, self.param_types)
        child2 = Individual(child2_genes, self.bounds, self.param_types)
        
        return child1, child2
    
    def __lt__(self, other: 'Individual') -> bool:
        """比较操作（用于排序）"""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness < other.fitness

class GeneticOptimizer:
    """遗传算法优化器"""
    
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
        
        self.bounds = param_space.get_bounds()
        self.param_types = param_space.get_param_types()
        self.param_names = param_space.get_param_names()
        
        # 优化历史
        self.population_history: List[List[Individual]] = []
        self.fitness_history: List[List[float]] = []
        self.best_fitness_history: List[float] = []
        self.best_individual: Optional[Individual] = None
        
        # 设置随机种子
        if config and hasattr(config, 'random_state'):
            random.seed(config.random_state)
            np.random.seed(config.random_state)
    
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
        population_size = min(self.genetic_config.population_size, n_calls // 5)
        max_generations = min(self.genetic_config.max_generations, n_calls // population_size)
        
        logger.info(f"遗传算法优化开始: 种群大小={population_size}, 最大代数={max_generations}")
        
        # 初始化种群
        population = self._initialize_population(population_size)
        
        # 评估初始种群
        self._evaluate_population(population)
        
        convergence_counter = 0
        generation = 0
        
        for generation in range(max_generations):
            # 记录当前代信息
            self._record_generation(population, generation)
            
            # 检查收敛
            if self._check_convergence():
                convergence_counter += 1
                if convergence_counter >= self.genetic_config.convergence_patience:
                    logger.info(f"在第{generation}代检测到收敛，提前停止")
                    break
            else:
                convergence_counter = 0
            
            # 进化操作
            population = self._evolve_population(population)
            
            # 评估新种群
            self._evaluate_population(population)
            
            # 更新最佳个体
            self._update_best_individual(population)
            
            if self.config and self.config.verbose and generation % 10 == 0:
                best_fitness = self.best_individual.fitness
                logger.info(f"第{generation}代: 最佳适应度={best_fitness:.6f}")
        
        # 生成结果
        result = self._generate_result(generation + 1)
        
        logger.info(f"遗传算法优化完成: 最佳适应度={result['best_value']:.6f}")
        
        return result
    
    def _initialize_population(self, population_size: int) -> List[Individual]:
        """初始化种群"""
        population = []
        
        for _ in range(population_size):
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
                individual.fitness = self.evaluator.evaluate_mesh(params)
    
    def _evolve_population(self, population: List[Individual]) -> List[Individual]:
        """进化种群"""
        # 排序种群
        population.sort()
        
        # 保留精英
        elite = population[:self.genetic_config.elite_size]
        new_population = [Individual(ind.genes, self.bounds, self.param_types) for ind in elite]
        
        # 生成后代
        while len(new_population) < len(population):
            # 选择父母
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # 交叉
            child1, child2 = parent1.crossover(parent2, self.genetic_config.crossover_rate)
            
            # 变异
            child1.mutate(self.genetic_config.mutation_rate)
            child2.mutate(self.genetic_config.mutation_rate)
            
            new_population.extend([child1, child2])
        
        # 确保种群大小
        return new_population[:len(population)]
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(population, 
                                  min(self.genetic_config.tournament_size, len(population)))
        return min(tournament)  # 返回适应度最好的个体
    
    def _update_best_individual(self, population: List[Individual]) -> None:
        """更新最佳个体"""
        current_best = min(population)
        
        if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
            self.best_individual = Individual(current_best.genes, self.bounds, self.param_types)
            self.best_individual.fitness = current_best.fitness
    
    def _record_generation(self, population: List[Individual], generation: int) -> None:
        """记录当前代信息"""
        # 复制种群（避免引用问题）
        population_copy = [Individual(ind.genes, self.bounds, self.param_types) for ind in population]
        for i, ind in enumerate(population):
            population_copy[i].fitness = ind.fitness
        
        self.population_history.append(population_copy)
        
        # 记录适应度
        fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
        self.fitness_history.append(fitness_values)
        
        if fitness_values:
            self.best_fitness_history.append(min(fitness_values))
    
    def _check_convergence(self) -> bool:
        """检查收敛性"""
        if len(self.best_fitness_history) < 5:
            return False
        
        recent_best = self.best_fitness_history[-5:]
        variance = np.var(recent_best)
        
        return variance < self.genetic_config.convergence_threshold
    
    def _generate_result(self, total_generations: int) -> Dict[str, Any]:
        """生成优化结果"""
        best_params = self.best_individual.to_params(self.param_names)
        
        return {
            'best_params': best_params,
            'best_value': self.best_individual.fitness,
            'optimizer_name': 'Genetic Algorithm',
            'total_generations': total_generations,
            'population_size': len(self.population_history[0]) if self.population_history else 0,
            'convergence_detected': len(self.best_fitness_history) < total_generations,
            'genetic_result': self  # 返回完整的遗传算法结果对象
        }
    
    def plot_evolution(self, save_path: Optional[str] = None) -> None:
        """绘制进化过程"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 最佳适应度变化
            ax1.plot(self.best_fitness_history, 'b-', linewidth=2, label='最佳适应度')
            ax1.set_xlabel('代数')
            ax1.set_ylabel('适应度')
            ax1.set_title('最佳适应度进化曲线')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 种群适应度分布（最后一代）
            if self.fitness_history:
                final_fitness = self.fitness_history[-1]
                ax2.hist(final_fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.axvline(x=self.best_individual.fitness, color='red', linestyle='--', 
                           linewidth=2, label=f'最佳值: {self.best_individual.fitness:.4f}')
                ax2.set_xlabel('适应度')
                ax2.set_ylabel('个体数量')
                ax2.set_title('最终种群适应度分布')
                ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"进化图表已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"绘制进化图表失败: {e}")
    
    def get_diversity_metrics(self) -> Dict[str, List[float]]:
        """计算种群多样性指标"""
        diversity_metrics = {
            'genetic_diversity': [],
            'fitness_diversity': [],
            'phenotypic_diversity': []
        }
        
        for population in self.population_history:
            # 遗传多样性（基因差异）
            genes_matrix = np.array([ind.genes for ind in population])
            genetic_diversity = np.mean(np.std(genes_matrix, axis=0))
            diversity_metrics['genetic_diversity'].append(genetic_diversity)
            
            # 适应度多样性
            fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
            if fitness_values:
                fitness_diversity = np.std(fitness_values)
                diversity_metrics['fitness_diversity'].append(fitness_diversity)
            
            # 表型多样性（参数空间中的分布）
            param_matrix = np.array([list(ind.to_params(self.param_names).values()) for ind in population])
            phenotypic_diversity = np.mean(np.std(param_matrix, axis=0))
            diversity_metrics['phenotypic_diversity'].append(phenotypic_diversity)
        
        return diversity_metrics
    
    def export_evolution_data(self, filename: str) -> None:
        """导出进化数据"""
        try:
            import json
            
            export_data = {
                'config': {
                    'population_size': self.genetic_config.population_size,
                    'mutation_rate': self.genetic_config.mutation_rate,
                    'crossover_rate': self.genetic_config.crossover_rate,
                    'tournament_size': self.genetic_config.tournament_size
                },
                'best_fitness_history': self.best_fitness_history,
                'best_individual': {
                    'genes': self.best_individual.genes,
                    'fitness': self.best_individual.fitness,
                    'params': self.best_individual.to_params(self.param_names)
                },
                'diversity_metrics': self.get_diversity_metrics(),
                'total_generations': len(self.population_history)
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"进化数据已导出: {filename}")
            
        except Exception as e:
            logger.error(f"导出进化数据失败: {e}")

# 多目标遗传算法优化器
class MultiObjectiveGeneticOptimizer(GeneticOptimizer):
    """多目标遗传算法优化器（NSGA-II）"""
    
    def __init__(self, param_space, evaluators: List, config=None, genetic_config=None):
        """
        初始化多目标遗传算法优化器
        
        Args:
            param_space: 参数空间
            evaluators: 评估器列表（每个目标一个）
            config: 优化配置
            genetic_config: 遗传算法配置
        """
        # 使用第一个评估器初始化基类
        super().__init__(param_space, evaluators[0], config, genetic_config)
        self.evaluators = evaluators
        self.n_objectives = len(evaluators)
    
    def _evaluate_population(self, population: List[Individual]) -> None:
        """多目标评估种群"""
        for individual in population:
            if individual.fitness is None:
                params = individual.to_params(self.param_names)
                
                # 评估所有目标
                fitness_values = []
                for evaluator in self.evaluators:
                    fitness = evaluator.evaluate_mesh(params)
                    fitness_values.append(fitness)
                
                individual.fitness = fitness_values  # 多目标适应度
    
    def _non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """非支配排序"""
        fronts = []
        
        # 计算支配关系
        domination_count = [0] * len(population)
        dominated_solutions = [[] for _ in range(len(population))]
        
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                if i != j:
                    if self._dominates(ind1.fitness, ind2.fitness):
                        dominated_solutions[i].append(j)
                    elif self._dominates(ind2.fitness, ind1.fitness):
                        domination_count[i] += 1
        
        # 构建第一前沿
        current_front = []
        for i, count in enumerate(domination_count):
            if count == 0:
                current_front.append(i)
        
        # 构建后续前沿
        while current_front:
            fronts.append([population[i] for i in current_front])
            next_front = []
            
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        return fronts
    
    def _dominates(self, fitness1: List[float], fitness2: List[float]) -> bool:
        """检查fitness1是否支配fitness2"""
        at_least_one_better = False
        
        for f1, f2 in zip(fitness1, fitness2):
            if f1 > f2:  # 假设最小化问题
                return False
            elif f1 < f2:
                at_least_one_better = True
        
        return at_least_one_better
    
    def _crowding_distance(self, front: List[Individual]) -> List[float]:
        """计算拥挤距离"""
        n = len(front)
        distances = [0.0] * n
        
        for obj_idx in range(self.n_objectives):
            # 按目标函数值排序
            front_sorted = sorted(enumerate(front), 
                                key=lambda x: x[1].fitness[obj_idx])
            
            # 边界点设置为无穷大
            distances[front_sorted[0][0]] = float('inf')
            distances[front_sorted[-1][0]] = float('inf')
            
            # 计算中间点的拥挤距离
            obj_range = (front_sorted[-1][1].fitness[obj_idx] - 
                        front_sorted[0][1].fitness[obj_idx])
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distance = (front_sorted[i + 1][1].fitness[obj_idx] - 
                              front_sorted[i - 1][1].fitness[obj_idx]) / obj_range
                    distances[front_sorted[i][0]] += distance
        
        return distances
    
    def optimize(self, n_calls: int, **kwargs) -> Dict[str, Any]:
        """
        执行多目标遗传算法优化
        
        Args:
            n_calls: 总评估次数
            **kwargs: 其他参数
            
        Returns:
            优化结果字典
        """
        # 调整种群大小和代数
        population_size = min(self.genetic_config.population_size, n_calls // 5)
        max_generations = min(self.genetic_config.max_generations, n_calls // population_size)
        
        logger.info(f"多目标遗传算法优化开始: 种群大小={population_size}, 目标数={self.n_objectives}")
        
        # 初始化种群
        population = self._initialize_population(population_size)
        
        # 评估初始种群
        self._evaluate_population(population)
        
        for generation in range(max_generations):
            # 非支配排序
            fronts = self._non_dominated_sort(population)
            
            # 记录当前代信息
            self._record_generation(population, generation)
            
            # 生成下一代种群
            new_population = []
            
            for front in fronts:
                if len(new_population) + len(front) <= population_size:
                    new_population.extend(front)
                else:
                    # 需要选择部分个体
                    remaining = population_size - len(new_population)
                    distances = self._crowding_distance(front)
                    
                    # 按拥挤距离排序并选择
                    front_with_distances = list(zip(front, distances))
                    front_with_distances.sort(key=lambda x: x[1], reverse=True)
                    
                    new_population.extend([ind for ind, _ in front_with_distances[:remaining]])
                    break
            
            # 进化操作
            population = self._evolve_population(new_population)
            
            # 评估新种群
            self._evaluate_population(population)
            
            if self.config and self.config.verbose and generation % 10 == 0:
                logger.info(f"第{generation}代完成")
        
        # 获取帕累托前沿
        final_fronts = self._non_dominated_sort(population)
        pareto_front = final_fronts[0] if final_fronts else []
        
        # 生成结果
        result = {
            'pareto_front': [ind.to_params(self.param_names) for ind in pareto_front],
            'pareto_fitness': [ind.fitness for ind in pareto_front],
            'optimizer_name': 'Multi-Objective Genetic Algorithm (NSGA-II)',
            'total_generations': max_generations,
            'n_objectives': self.n_objectives,
            'population_size': population_size
        }
        
        logger.info(f"多目标优化完成: 帕累托前沿包含{len(pareto_front)}个解")
        
        return result

def optimize_with_genetic_algorithm(objective_func, param_space, bounds, param_types, 
                                   n_calls=20, random_state=42, verbose=True):
    """
    向后兼容的遗传算法优化函数
    
    Args:
        objective_func: 目标函数
        param_space: 参数空间
        bounds: 参数边界
        param_types: 参数类型
        n_calls: 评估次数
        random_state: 随机种子
        verbose: 是否显示详细信息
        
    Returns:
        优化结果字典
    """
    # 创建模拟评估器
    class LegacyEvaluator:
        def __init__(self, objective_func, param_names):
            self.objective_func = objective_func
            self.param_names = param_names
        
        def evaluate_mesh(self, params):
            return self.objective_func(params)
        
        def validate_params(self, params):
            return True
    
    # 创建模拟参数空间
    class LegacyParamSpace:
        def __init__(self, bounds, param_types, param_names):
            self._bounds = bounds
            self._param_types = param_types
            self._param_names = param_names
        
        def get_bounds(self):
            return self._bounds
        
        def get_param_types(self):
            return self._param_types
        
        def get_param_names(self):
            return self._param_names
    
    # 创建参数名称
    param_names = [dim.name for dim in param_space]
    
    # 创建评估器和参数空间
    evaluator = LegacyEvaluator(objective_func, param_names)
    legacy_param_space = LegacyParamSpace(bounds, param_types, param_names)
    
    # 创建配置
    class LegacyConfig:
        def __init__(self):
            self.random_state = random_state
            self.verbose = verbose
    
    config = LegacyConfig()
    
    # 执行优化
    optimizer = GeneticOptimizer(legacy_param_space, evaluator, config)
    result = optimizer.optimize(n_calls)
    
    # 转换结果格式以保持向后兼容
    class LegacyResult:
        def __init__(self, x, fun, x_iters, func_vals):
            self.x = x
            self.fun = fun
            self.x_iters = x_iters
            self.func_vals = func_vals
    
    # 构造历史数据
    x_iters = []
    func_vals = []
    
    for generation_fitness in optimizer.fitness_history:
        if generation_fitness:
            best_fitness = min(generation_fitness)
            func_vals.append(best_fitness)
            
            # 找到对应的个体
            for population in optimizer.population_history:
                for ind in population:
                    if ind.fitness == best_fitness:
                        x_iters.append(ind.genes)
                        break
                break
    
    legacy_result = LegacyResult(
        x=optimizer.best_individual.genes,
        fun=optimizer.best_individual.fitness,
        x_iters=x_iters,
        func_vals=func_vals
    )
    
    return {
        'best_params': result['best_params'],
        'best_value': result['best_value'],
        'optimizer': result['optimizer_name'],
        'execution_time': 0,  # 估算值
        'n_calls': n_calls,
        'result': legacy_result
    }