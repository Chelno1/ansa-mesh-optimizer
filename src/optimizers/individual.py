#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法个体类

定义遗传算法中个体的结构和操作方法，包括变异、交叉等核心操作
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Individual:
    """个体类 - 优化版本"""

    __slots__ = ["genes", "bounds", "param_types", "fitness", "age", "generation"]

    def __init__(
        self,
        genes: List[float],
        bounds: List[Tuple[float, float]],
        param_types: List[type],
    ):
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
        for i, (gene, (low, high), param_type) in enumerate(
            zip(self.genes, self.bounds, self.param_types)
        ):
            if param_type is int:
                self.genes[i] = max(low, min(high, round(gene)))
            else:
                self.genes[i] = max(low, min(high, gene))

    def to_params(self, param_names: List[str]) -> Dict[str, float]:
        """转换为参数字典"""
        params = {}
        for i, name in enumerate(param_names):
            if self.param_types[i] is int:
                params[name] = int(round(self.genes[i]))
            else:
                params[name] = self.genes[i]
        return params

    def mutate(
        self, mutation_rate: float, generation: int = 0, max_generations: int = 100
    ) -> None:
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

                if self.param_types[i] is int:
                    # 整数变异
                    range_size = max(1, int((high - low) * 0.1))
                    delta = random.randint(-range_size, range_size)
                    self.genes[i] += delta
                else:
                    # 实数变异（自适应高斯变异）
                    mutation_strength = (
                        (high - low) * 0.1 * (1 - generation / max_generations)
                    )
                    self.genes[i] += random.gauss(0, mutation_strength)

        self._constrain_genes()
        self.fitness = None  # 重置适应度
        self.age += 1

    def crossover(
        self, other: "Individual", crossover_rate: float
    ) -> Tuple["Individual", "Individual"]:
        """
        交叉操作 - 增强版本

        Args:
            other: 另一个个体
            crossover_rate: 交叉率

        Returns:
            两个子代个体
        """
        if random.random() > crossover_rate:
            return Individual(self.genes, self.bounds, self.param_types), Individual(
                other.genes, self.bounds, self.param_types
            )

        # 模拟二进制交叉（SBX）用于实数参数
        child1_genes = []
        child2_genes = []

        for i in range(len(self.genes)):
            if self.param_types[i] is int:
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

    def distance_to(self, other: "Individual") -> float:
        """计算与另一个个体的距离"""
        total_distance = 0.0
        for i, ((gene1, gene2), (low, high)) in enumerate(
            zip(zip(self.genes, other.genes), self.bounds)
        ):
            # 标准化距离
            normalized_distance = abs(gene1 - gene2) / (high - low) if high > low else 0
            total_distance += normalized_distance**2

        return total_distance**0.5

    def __lt__(self, other: "Individual") -> bool:
        """比较操作（用于排序）"""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness < other.fitness

    def copy(self) -> "Individual":
        """创建个体的深拷贝"""
        new_individual = Individual(self.genes, self.bounds, self.param_types)
        new_individual.fitness = self.fitness
        new_individual.age = self.age
        new_individual.generation = self.generation
        return new_individual

    def __str__(self) -> str:
        """字符串表示"""
        return f"Individual(fitness={self.fitness}, generation={self.generation}, age={self.age})"

    def __repr__(self) -> str:
        """详细表示"""
        return f"Individual(genes={self.genes}, fitness={self.fitness}, generation={self.generation})"


def create_individual(
    bounds: List[Tuple[float, float]], param_types: List[type]
) -> Individual:
    """
    创建随机个体

    Args:
        bounds: 参数边界
        param_types: 参数类型

    Returns:
        新创建的个体
    """
    genes = []
    for (low, high), param_type in zip(bounds, param_types):
        if param_type is int:
            gene = random.randint(int(low), int(high))
        else:
            gene = random.uniform(low, high)
        genes.append(gene)

    return Individual(genes, bounds, param_types)


def create_population(
    size: int, bounds: List[Tuple[float, float]], param_types: List[type]
) -> List[Individual]:
    """
    创建随机种群

    Args:
        size: 种群大小
        bounds: 参数边界
        param_types: 参数类型

    Returns:
        新创建的种群
    """
    return [create_individual(bounds, param_types) for _ in range(size)]


def create_lhs_population(
    size: int, bounds: List[Tuple[float, float]], param_types: List[type]
) -> List[Individual]:
    """
    使用拉丁超立方采样创建种群

    Args:
        size: 种群大小
        bounds: 参数边界
        param_types: 参数类型

    Returns:
        新创建的种群
    """
    population = []

    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=len(bounds))
        lhs_samples = sampler.random(size)

        # 缩放到参数边界
        for sample in lhs_samples:
            genes = []
            for i, (s, (low, high), param_type) in enumerate(
                zip(sample, bounds, param_types)
            ):
                if param_type is int:
                    gene = int(int(low) + s * (int(high) - int(low)))
                else:
                    gene = low + s * (high - low)
                genes.append(gene)

            individual = Individual(genes, bounds, param_types)
            population.append(individual)

    except ImportError:
        logger.warning("scipy不可用，使用随机初始化替代拉丁超立方采样")
        population = create_population(size, bounds, param_types)

    return population


def calculate_population_diversity(population: List[Individual]) -> float:
    """
    计算种群多样性

    Args:
        population: 种群

    Returns:
        多样性值（0-1之间）
    """
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
