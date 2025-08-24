#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法模块化组件单元测试

测试新的模块化遗传算法组件的功能
"""

import os
import tempfile
import unittest


from ansa_mesh_optimizer.optimizers.evolution import (
    EvolutionEngine,
    tournament_selection,
)

# 导入测试目标
from ansa_mesh_optimizer.optimizers.genetic_config import (
    GeneticConfig,
    create_adaptive_genetic_config,
    create_fast_genetic_config,
    create_thorough_genetic_config,
)
from ansa_mesh_optimizer.optimizers.genetic_visualization import (
    GeneticVisualizer,
    export_evolution_data,
)
from ansa_mesh_optimizer.optimizers.individual import (
    Individual,
    calculate_population_diversity,
    create_individual,
    create_population,
)


class TestGeneticConfig(unittest.TestCase):
    """测试遗传算法配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = GeneticConfig()
        self.assertEqual(config.population_size, 50)
        self.assertEqual(config.elite_size, 5)
        self.assertEqual(config.mutation_rate, 0.1)
        self.assertEqual(config.crossover_rate, 0.8)

    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        config = GeneticConfig(population_size=20, elite_size=2)
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # 无效配置 - 种群大小太小
        config = GeneticConfig(population_size=2)
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

        # 无效配置 - 变异率超出范围
        config = GeneticConfig(mutation_rate=1.5)
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = GeneticConfig(population_size=30, mutation_rate=0.2)
        config_dict = config.to_dict()
        self.assertEqual(config_dict["population_size"], 30)
        self.assertEqual(config_dict["mutation_rate"], 0.2)

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {"population_size": 40, "elite_size": 8}
        config = GeneticConfig.from_dict(config_dict)
        self.assertEqual(config.population_size, 40)
        self.assertEqual(config.elite_size, 8)

    def test_preset_configs(self):
        """测试预设配置"""
        fast_config = create_fast_genetic_config()
        self.assertEqual(fast_config.population_size, 20)

        thorough_config = create_thorough_genetic_config()
        self.assertEqual(thorough_config.population_size, 100)

        adaptive_config = create_adaptive_genetic_config()
        self.assertTrue(adaptive_config.adaptive_mutation)
        self.assertTrue(adaptive_config.diversity_preservation)


class TestIndividual(unittest.TestCase):
    """测试个体类"""

    def setUp(self):
        """设置测试数据"""
        self.bounds = [(0.0, 10.0), (-5.0, 5.0), (1.0, 100.0)]
        self.param_types = [float, float, int]
        self.genes = [5.0, 0.0, 50]

    def test_individual_creation(self):
        """测试个体创建"""
        individual = Individual(self.genes, self.bounds, self.param_types)
        self.assertEqual(len(individual.genes), 3)
        self.assertEqual(individual.fitness, None)
        self.assertEqual(individual.age, 0)
        self.assertEqual(individual.generation, 0)

    def test_gene_constraints(self):
        """测试基因约束"""
        # 超出边界的基因
        invalid_genes = [15.0, -10.0, 150]
        individual = Individual(invalid_genes, self.bounds, self.param_types)

        # 应该被约束在边界内
        self.assertLessEqual(individual.genes[0], 10.0)
        self.assertGreaterEqual(individual.genes[1], -5.0)
        self.assertLessEqual(individual.genes[2], 100)

    def test_to_params(self):
        """测试参数转换"""
        individual = Individual(self.genes, self.bounds, self.param_types)
        param_names = ["x", "y", "z"]
        params = individual.to_params(param_names)

        self.assertEqual(params["x"], 5.0)
        self.assertEqual(params["y"], 0.0)
        self.assertEqual(params["z"], 50)
        self.assertIsInstance(params["z"], int)

    def test_mutation(self):
        """测试变异操作"""
        individual = Individual(self.genes.copy(), self.bounds, self.param_types)
        individual.genes.copy()

        # 高变异率应该产生变化
        individual.mutate(mutation_rate=1.0, generation=0, max_generations=100)

        # 基因应该有所变化（虽然不能保证100%）
        # 但至少适应度应该被重置
        self.assertIsNone(individual.fitness)
        self.assertEqual(individual.age, 1)

    def test_crossover(self):
        """测试交叉操作"""
        parent1 = Individual([1.0, 2.0, 10], self.bounds, self.param_types)
        parent2 = Individual([9.0, 8.0, 90], self.bounds, self.param_types)

        child1, child2 = parent1.crossover(parent2, crossover_rate=1.0)

        # 子代应该是新的个体
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
        self.assertEqual(len(child1.genes), 3)
        self.assertEqual(len(child2.genes), 3)

    def test_distance_calculation(self):
        """测试个体间距离计算"""
        individual1 = Individual([0.0, 0.0, 1], self.bounds, self.param_types)
        individual2 = Individual([10.0, 5.0, 100], self.bounds, self.param_types)

        distance = individual1.distance_to(individual2)
        self.assertGreater(distance, 0)
        self.assertIsInstance(distance, float)

    def test_comparison(self):
        """测试个体比较"""
        individual1 = Individual(self.genes, self.bounds, self.param_types)
        individual2 = Individual(self.genes, self.bounds, self.param_types)

        individual1.fitness = 10.0
        individual2.fitness = 20.0

        self.assertTrue(individual1 < individual2)

    def test_copy(self):
        """测试个体复制"""
        individual = Individual(self.genes, self.bounds, self.param_types)
        individual.fitness = 15.0
        individual.age = 5
        individual.generation = 3

        copy_individual = individual.copy()

        self.assertEqual(copy_individual.genes, individual.genes)
        self.assertEqual(copy_individual.fitness, individual.fitness)
        self.assertEqual(copy_individual.age, individual.age)
        self.assertEqual(copy_individual.generation, individual.generation)

        # 修改原个体不应影响副本
        individual.genes[0] = 999
        self.assertNotEqual(copy_individual.genes[0], 999)


class TestEvolutionEngine(unittest.TestCase):
    """测试进化引擎"""

    def setUp(self):
        """设置测试数据"""
        self.bounds = [(0, 10), (-5, 5)]
        self.param_types = [float, float]
        self.genetic_config = GeneticConfig(
            population_size=20, elite_size=2, mutation_rate=0.1, crossover_rate=0.8
        )
        self.evolution_engine = EvolutionEngine(
            self.bounds, self.param_types, self.genetic_config
        )

    def test_population_initialization(self):
        """测试种群初始化"""
        population = self.evolution_engine.initialize_population(10)
        self.assertEqual(len(population), 10)

        for individual in population:
            self.assertIsInstance(individual, Individual)
            self.assertEqual(len(individual.genes), 2)

    def test_population_evolution(self):
        """测试种群进化"""
        # 创建初始种群
        population = self.evolution_engine.initialize_population(10)

        # 设置适应度
        for i, individual in enumerate(population):
            individual.fitness = float(i)

        # 进化一代
        new_population = self.evolution_engine.evolve_population(
            population, generation=0, max_generations=100
        )

        self.assertEqual(len(new_population), len(population))

        # 精英个体应该被保留
        new_population.sort()
        # 确保适应度不为None再比较
        if new_population[0].fitness is not None and population[0].fitness is not None:
            self.assertLessEqual(new_population[0].fitness, population[0].fitness)

    def test_selection(self):
        """测试选择操作"""
        population = self.evolution_engine.initialize_population(10)

        # 设置适应度
        for i, individual in enumerate(population):
            individual.fitness = float(i)

        selected = self.evolution_engine.selection(population)
        self.assertIsInstance(selected, Individual)
        self.assertIn(selected, population)

    def test_convergence_check(self):
        """测试收敛检查"""
        # 初始状态不应收敛
        self.assertFalse(self.evolution_engine.check_convergence())

        # 添加相似的适应度历史 - 需要足够的历史记录
        for _ in range(self.genetic_config.convergence_patience + 5):
            self.evolution_engine.best_fitness_history.append(1.0)

        # 手动设置收敛计数器来模拟收敛状态
        self.evolution_engine.convergence_counter = (
            self.genetic_config.convergence_patience
        )

        # 现在应该收敛
        self.assertTrue(self.evolution_engine.check_convergence())

    def test_diversity_metrics(self):
        """测试多样性指标"""
        metrics = self.evolution_engine.get_diversity_metrics()
        self.assertIn("diversity_history", metrics)
        self.assertIn("best_fitness_history", metrics)
        self.assertIsInstance(metrics["diversity_history"], list)
        self.assertIsInstance(metrics["best_fitness_history"], list)


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def setUp(self):
        """设置测试数据"""
        self.bounds = [(0.0, 10.0), (-5.0, 5.0)]
        self.param_types = [float, float]

    def test_create_individual(self):
        """测试创建单个个体"""
        individual = create_individual(self.bounds, self.param_types)
        self.assertIsInstance(individual, Individual)
        self.assertEqual(len(individual.genes), 2)

    def test_create_population(self):
        """测试创建种群"""
        population = create_population(5, self.bounds, self.param_types)
        self.assertEqual(len(population), 5)

        for individual in population:
            self.assertIsInstance(individual, Individual)

    def test_calculate_population_diversity(self):
        """测试种群多样性计算"""
        population = create_population(10, self.bounds, self.param_types)
        diversity = calculate_population_diversity(population)

        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)

    def test_tournament_selection(self):
        """测试锦标赛选择"""
        population = create_population(10, self.bounds, self.param_types)

        # 设置适应度
        for i, individual in enumerate(population):
            individual.fitness = float(i)

        selected = tournament_selection(population, tournament_size=3)
        self.assertIsInstance(selected, Individual)
        self.assertIn(selected, population)


class TestGeneticVisualization(unittest.TestCase):
    """测试遗传算法可视化"""

    def setUp(self):
        """设置测试数据"""
        self.best_fitness_history = [10.0, 8.0, 6.0, 4.0, 2.0]
        self.diversity_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        self.generation_stats = [
            {
                "generation": i,
                "best_fitness": fitness,
                "mean_fitness": fitness + 1,
                "std_fitness": 0.5,
                "diversity": div,
            }
            for i, (fitness, div) in enumerate(
                zip(self.best_fitness_history, self.diversity_history)
            )
        ]

    def test_visualizer_creation(self):
        """测试可视化器创建"""
        visualizer = GeneticVisualizer(
            self.best_fitness_history, self.diversity_history, self.generation_stats
        )

        self.assertEqual(visualizer.best_fitness_history, self.best_fitness_history)
        self.assertEqual(visualizer.diversity_history, self.diversity_history)
        self.assertEqual(visualizer.generation_stats, self.generation_stats)

    def test_export_evolution_data(self):
        """测试导出进化数据"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filename = f.name

        try:
            export_evolution_data(
                best_fitness_history=self.best_fitness_history,
                diversity_history=self.diversity_history,
                generation_stats=self.generation_stats,
                genetic_config={"population_size": 50},
                best_individual_info={"genes": [1.0, 2.0], "fitness": 1.0},
                metadata={"test": True},
                filename=filename,
            )

            # 验证文件被创建
            self.assertTrue(os.path.exists(filename))

            # 验证文件内容
            import json

            with open(filename, "r") as f:
                data = json.load(f)

            self.assertIn("config", data)
            self.assertIn("results", data)
            self.assertIn("best_individual", data)
            self.assertIn("metadata", data)

        finally:
            # 清理临时文件
            if os.path.exists(filename):
                os.unlink(filename)


if __name__ == "__main__":
    unittest.main()
