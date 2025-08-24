#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行遗传算法优化器 - 核心性能提升版本

作者: Chel
创建日期: 2025-07-15
版本: 2.0.0
功能: 实现完整的并行化遗传算法，包括并行评估、进化和多样性计算
"""

import logging
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from .genetic_config import GeneticConfig

# 导入基础组件
from .genetic_optimizer import GeneticOptimizer
from .individual import Individual
from .optimizer_config import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class ParallelGeneticConfig(GeneticConfig):
    """并行遗传算法配置 - 完整版本"""

    # === 基础并行配置 ===
    n_workers: int = 4  # 工作进程数
    use_multiprocessing: bool = True  # 是否使用多进程
    parallel_evaluation: bool = True  # 并行适应度评估
    parallel_diversity: bool = True  # 并行多样性计算
    parallel_evolution: bool = True  # 并行进化操作

    # === 批处理配置 ===
    evaluation_batch_size: int = 10  # 评估批次大小
    evolution_batch_size: int = 20  # 进化批次大小
    auto_batch_size: bool = True  # 自动调整批次大小

    # === 内存管理 ===
    max_memory_usage: float = 0.8  # 最大内存使用率
    shared_memory: bool = True  # 使用共享内存
    memory_monitoring: bool = True  # 内存监控

    # === 分布式配置 ===
    distributed_mode: bool = False  # 分布式模式
    island_count: int = 4  # 岛屿数量
    migration_interval: int = 10  # 迁移间隔
    migration_rate: float = 0.1  # 迁移率
    topology: str = "ring"  # 拓扑结构: ring, star, mesh

    # === 性能优化 ===
    vectorized_operations: bool = True  # 向量化操作
    cache_evaluations: bool = True  # 缓存评估结果
    lazy_evaluation: bool = True  # 惰性评估

    # === 容错配置 ===
    fault_tolerance: bool = True  # 容错机制
    max_retries: int = 3  # 最大重试次数
    timeout_seconds: int = 300  # 超时时间

    # === 性能监控 ===
    performance_monitoring: bool = True  # 性能监控
    profiling_enabled: bool = False  # 性能分析
    log_performance: bool = True  # 记录性能数据

    # === 收敛控制 ===
    max_stagnation_iterations: int = 20  # 最大停滞迭代数
    early_stopping: bool = True  # 早停机制

    # === 随机种子 ===
    random_state: int = 42  # 随机种子

    def __post_init__(self):
        """初始化后验证并行配置"""
        # 调用父类的验证方法
        GeneticConfig.validate(self)
        self._validate_parallel_config()

        # 自动调整工作进程数
        if self.n_workers <= 0:
            self.n_workers = min(multiprocessing.cpu_count(), 8)

    def _validate_parallel_config(self):
        """验证并行配置"""
        if self.n_workers > multiprocessing.cpu_count():
            logger.warning(
                f"工作进程数({self.n_workers})超过CPU核心数({multiprocessing.cpu_count()})"
            )

        if self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size必须大于0")

        if not 0 < self.max_memory_usage <= 1:
            raise ValueError("max_memory_usage必须在(0,1]范围内")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须大于0")


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {
            "evaluation_times": [],
            "evolution_times": [],
            "diversity_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "throughput": [],
        }
        self.start_time = time.time()
        self._lock = threading.Lock()

    @contextmanager
    def measure_time(self, operation: str):
        """测量操作时间"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            with self._lock:
                if f"{operation}_times" not in self.metrics:
                    self.metrics[f"{operation}_times"] = []
                self.metrics[f"{operation}_times"].append(duration)

    def get_memory_usage(self) -> float:
        """获取当前内存使用率"""
        try:
            return psutil.virtual_memory().percent / 100.0
        except Exception:
            return 0.5  # 默认值

    def get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1) / 100.0
        except Exception:
            return 0.5  # 默认值

    def record_throughput(self, evaluations_per_second: float):
        """记录吞吐量"""
        with self._lock:
            self.metrics["throughput"].append(evaluations_per_second)

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self._lock:
            return {
                "total_runtime": time.time() - self.start_time,
                "avg_evaluation_time": (
                    np.mean(self.metrics["evaluation_times"])
                    if self.metrics["evaluation_times"]
                    else 0
                ),
                "avg_evolution_time": (
                    np.mean(self.metrics["evolution_times"])
                    if self.metrics["evolution_times"]
                    else 0
                ),
                "avg_diversity_time": (
                    np.mean(self.metrics["diversity_times"])
                    if self.metrics["diversity_times"]
                    else 0
                ),
                "peak_memory_usage": (
                    max(self.metrics["memory_usage"])
                    if self.metrics["memory_usage"]
                    else 0
                ),
                "avg_throughput": (
                    np.mean(self.metrics["throughput"])
                    if self.metrics["throughput"]
                    else 0
                ),
                "total_evaluations": len(self.metrics["evaluation_times"]),
            }


def evaluate_individual_worker(args):
    """工作进程中的个体评估函数"""
    try:
        evaluator, params, individual_data, timeout = args

        # 重建Individual对象
        individual = Individual(
            individual_data["genes"],
            individual_data["bounds"],
            individual_data["param_types"],
        )
        individual.fitness = individual_data["fitness"]
        individual.age = individual_data["age"]
        individual.generation = individual_data["generation"]

        # 如果已经有适应度，跳过评估
        if individual.fitness is not None:
            return individual_data["fitness"], None

        # 执行评估
        start_time = time.time()
        fitness = evaluator.evaluate_mesh(params)
        evaluation_time = time.time() - start_time

        return fitness, evaluation_time

    except Exception as e:
        logger.warning(f"工作进程评估失败: {e}")
        return float("inf"), None


class ParallelEvaluationEngine:
    """并行评估引擎 - 核心组件"""

    def __init__(
        self, evaluator, config: ParallelGeneticConfig, param_names: List[str]
    ):
        self.evaluator = evaluator
        self.config = config
        self.param_names = param_names
        self.process_pool = None
        self.performance_monitor = PerformanceMonitor()
        self.evaluation_cache = {} if config.cache_evaluations else None

        self._initialize_parallel_resources()

    def _initialize_parallel_resources(self):
        """初始化并行资源"""
        if self.config.use_multiprocessing and self.config.parallel_evaluation:
            try:
                # 创建进程池
                self.process_pool = ProcessPoolExecutor(
                    max_workers=self.config.n_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
                logger.info(
                    f"并行评估引擎初始化完成 - 工作进程数: {self.config.n_workers}"
                )
            except Exception as e:
                logger.warning(f"进程池初始化失败，回退到串行模式: {e}")
                self.process_pool = None

    def evaluate_population_parallel(self, population: List[Individual]) -> int:
        """并行评估种群"""
        unevaluated = [ind for ind in population if ind.fitness is None]
        if not unevaluated:
            return 0

        logger.info(f"开始并行评估 {len(unevaluated)} 个个体")

        with self.performance_monitor.measure_time("batch_evaluation"):
            if self.process_pool is None or not self.config.parallel_evaluation:
                return self._evaluate_population_serial(unevaluated)
            else:
                return self._evaluate_population_multiprocess(unevaluated)

    def _evaluate_population_serial(self, population: List[Individual]) -> int:
        """串行评估种群（备用方案）"""
        evaluation_count = 0

        for individual in population:
            if individual.fitness is None:
                params = individual.to_params(self.param_names)

                # 检查缓存
                if self.evaluation_cache is not None:
                    cache_key = self._get_cache_key(params)
                    if cache_key in self.evaluation_cache:
                        individual.fitness = self.evaluation_cache[cache_key]
                        evaluation_count += 1
                        continue

                try:
                    individual.fitness = self.evaluator.evaluate_mesh(params)

                    # 更新缓存
                    if self.evaluation_cache is not None:
                        cache_key = self._get_cache_key(params)
                        self.evaluation_cache[cache_key] = individual.fitness

                    evaluation_count += 1
                except Exception as e:
                    logger.warning(f"个体评估失败: {e}")
                    individual.fitness = float("inf")
                    evaluation_count += 1

        return evaluation_count

    def _evaluate_population_multiprocess(self, population: List[Individual]) -> int:
        """多进程评估种群"""
        evaluation_count = 0
        batch_size = self._calculate_optimal_batch_size(len(population))

        # 分批处理
        for batch_start in range(0, len(population), batch_size):
            batch_end = min(batch_start + batch_size, len(population))
            batch = population[batch_start:batch_end]

            # 准备评估任务
            tasks = []
            individuals_to_evaluate = []

            for individual in batch:
                if individual.fitness is None:
                    params = individual.to_params(self.param_names)

                    # 检查缓存
                    if self.evaluation_cache is not None:
                        cache_key = self._get_cache_key(params)
                        if cache_key in self.evaluation_cache:
                            individual.fitness = self.evaluation_cache[cache_key]
                            evaluation_count += 1
                            continue

                    # 准备个体数据（用于进程间传输）
                    individual_data = {
                        "genes": individual.genes.copy(),
                        "bounds": individual.bounds,
                        "param_types": individual.param_types,
                        "fitness": individual.fitness,
                        "age": individual.age,
                        "generation": individual.generation,
                    }

                    task_args = (
                        self.evaluator,
                        params,
                        individual_data,
                        self.config.timeout_seconds,
                    )
                    tasks.append(task_args)
                    individuals_to_evaluate.append(individual)

            if not tasks:
                continue

            # 提交并行任务
            try:
                futures = []
                if self.process_pool is not None:
                    for task_args in tasks:
                        future = self.process_pool.submit(
                            evaluate_individual_worker, task_args
                        )
                        futures.append(future)
                else:
                    # 如果进程池不可用，回退到串行评估
                    for i, individual in enumerate(individuals_to_evaluate):
                        try:
                            params = individual.to_params(self.param_names)
                            individual.fitness = self.evaluator.evaluate_mesh(params)
                            evaluation_count += 1
                        except Exception as eval_e:
                            logger.warning(f"串行评估失败: {eval_e}")
                            individual.fitness = float("inf")
                            evaluation_count += 1
                    continue

                # 收集结果
                for i, future in enumerate(futures):
                    try:
                        fitness, eval_time = future.result(
                            timeout=self.config.timeout_seconds
                        )
                        individuals_to_evaluate[i].fitness = fitness

                        # 更新缓存
                        if self.evaluation_cache is not None and fitness != float(
                            "inf"
                        ):
                            params = individuals_to_evaluate[i].to_params(
                                self.param_names
                            )
                            cache_key = self._get_cache_key(params)
                            self.evaluation_cache[cache_key] = fitness

                        evaluation_count += 1

                        # 记录性能
                        if eval_time is not None:
                            self.performance_monitor.metrics["evaluation_times"].append(
                                eval_time
                            )

                    except Exception as e:
                        logger.warning(f"个体评估任务失败: {e}")
                        individuals_to_evaluate[i].fitness = float("inf")
                        evaluation_count += 1

            except Exception as e:
                logger.error(f"批量评估失败，回退到串行模式: {e}")
                # 回退到串行评估
                for individual in individuals_to_evaluate:
                    if individual.fitness is None:
                        try:
                            params = individual.to_params(self.param_names)
                            individual.fitness = self.evaluator.evaluate_mesh(params)
                            evaluation_count += 1
                        except Exception as eval_e:
                            logger.warning(f"串行评估失败: {eval_e}")
                            individual.fitness = float("inf")
                            evaluation_count += 1

        # 记录吞吐量
        if evaluation_count > 0:
            total_time = sum(
                self.performance_monitor.metrics.get("batch_evaluation_times", [0])
            )
            if total_time > 0:
                throughput = evaluation_count / total_time
                self.performance_monitor.record_throughput(throughput)

        logger.info(f"并行评估完成: {evaluation_count} 个个体")
        return evaluation_count

    def _calculate_optimal_batch_size(self, population_size: int) -> int:
        """计算最优批次大小"""
        if not self.config.auto_batch_size:
            return min(self.config.evaluation_batch_size, population_size)

        # 基于系统资源动态调整
        cpu_cores = self.config.n_workers
        memory_usage = self.performance_monitor.get_memory_usage()
        memory_factor = max(0.1, 1.0 - memory_usage)

        # 计算最优批次大小
        optimal_size = min(
            population_size,
            max(1, int(cpu_cores * 2 * memory_factor)),
            self.config.evaluation_batch_size * 2,
        )

        logger.debug(f"动态批次大小: {optimal_size} (内存使用率: {memory_usage:.2%})")
        return optimal_size

    def _get_cache_key(self, params: Dict[str, float]) -> str:
        """生成缓存键"""
        # 将参数转换为可哈希的字符串
        sorted_params = sorted(params.items())
        return str(hash(tuple(sorted_params)))

    def cleanup(self):
        """清理资源"""
        if self.process_pool:
            try:
                self.process_pool.shutdown(wait=True)
                logger.info("并行评估引擎资源清理完成")
            except Exception as e:
                logger.warning(f"清理并行评估引擎资源时出错: {e}")


class ParallelDiversityEngine:
    """并行多样性计算引擎"""

    def __init__(self, config: ParallelGeneticConfig):
        self.config = config
        self.thread_pool = None

        if config.parallel_diversity:
            self.thread_pool = ThreadPoolExecutor(max_workers=config.n_workers)

    def calculate_diversity_parallel(self, population: List[Individual]) -> float:
        """并行计算种群多样性"""
        if len(population) < 2:
            return 0.0

        if self.config.vectorized_operations:
            return self._calculate_diversity_vectorized(population)
        elif self.config.parallel_diversity and self.thread_pool:
            return self._calculate_diversity_multithread(population)
        else:
            return self._calculate_diversity_serial(population)

    def _calculate_diversity_vectorized(self, population: List[Individual]) -> float:
        """使用NumPy向量化计算多样性"""
        try:
            # 提取基因矩阵
            genes_matrix = np.array([ind.genes for ind in population])

            # 标准化基因（使用第一个个体的边界信息）
            bounds = np.array(population[0].bounds)
            ranges = bounds[:, 1] - bounds[:, 0]
            ranges[ranges == 0] = 1  # 避免除零

            normalized_genes = (genes_matrix - bounds[:, 0]) / ranges

            # 计算距离矩阵
            try:
                from scipy.spatial.distance import pdist

                distances = pdist(normalized_genes, metric="euclidean")
                return float(np.mean(distances))
            except ImportError:
                # 如果scipy不可用，使用numpy计算
                n = len(population)
                total_distance = 0.0
                count = 0

                for i in range(n):
                    for j in range(i + 1, n):
                        diff = normalized_genes[i] - normalized_genes[j]
                        distance = np.sqrt(np.sum(diff**2))
                        total_distance += distance
                        count += 1

                return total_distance / count if count > 0 else 0.0

        except Exception as e:
            logger.warning(f"向量化多样性计算失败，回退到串行模式: {e}")
            return self._calculate_diversity_serial(population)

    def _calculate_diversity_multithread(self, population: List[Individual]) -> float:
        """使用多线程计算多样性"""
        try:
            n = len(population)
            chunk_size = max(1, n // self.config.n_workers)

            # 初始化累计变量
            total_distance = 0.0
            total_pairs = 0

            # 分块计算距离
            futures = []
            for i in range(0, n, chunk_size):
                chunk_end = min(i + chunk_size, n)
                chunk = population[i:chunk_end]
                remaining = population[chunk_end:]

                if self.thread_pool is not None:
                    future = self.thread_pool.submit(
                        self._calculate_chunk_distances, chunk, remaining
                    )
                    futures.append(future)
                else:
                    # 回退到串行计算
                    chunk_distance, chunk_pairs = self._calculate_chunk_distances(
                        chunk, remaining
                    )
                    total_distance += chunk_distance
                    total_pairs += chunk_pairs

            # 收集并行结果
            for future in futures:
                chunk_distance, chunk_pairs = future.result()
                total_distance += chunk_distance
                total_pairs += chunk_pairs

            return total_distance / total_pairs if total_pairs > 0 else 0.0

        except Exception as e:
            logger.warning(f"多线程多样性计算失败，回退到串行模式: {e}")
            return self._calculate_diversity_serial(population)

    def _calculate_diversity_serial(self, population: List[Individual]) -> float:
        """串行计算多样性"""
        n = len(population)
        total_distance = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                distance = population[i].distance_to(population[j])
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _calculate_chunk_distances(
        self, chunk: List[Individual], remaining: List[Individual]
    ) -> Tuple[float, int]:
        """计算块内和块间距离"""
        total_distance = 0.0
        count = 0

        # 块内距离
        for i in range(len(chunk)):
            for j in range(i + 1, len(chunk)):
                distance = chunk[i].distance_to(chunk[j])
                total_distance += distance
                count += 1

        # 块间距离
        for individual in chunk:
            for other in remaining:
                distance = individual.distance_to(other)
                total_distance += distance
                count += 1

        return total_distance, count

    def cleanup(self):
        """清理资源"""
        if self.thread_pool:
            try:
                self.thread_pool.shutdown(wait=True)
                logger.info("并行多样性引擎资源清理完成")
            except Exception as e:
                logger.warning(f"清理并行多样性引擎资源时出错: {e}")


class ParallelGeneticOptimizer(GeneticOptimizer):
    """并行遗传算法优化器 - 完整实现"""

    def __init__(self, param_space, evaluator, config=None, parallel_config=None):
        # 使用并行配置替换基础配置
        if parallel_config is None:
            parallel_config = ParallelGeneticConfig()

        # 初始化基础优化器
        super().__init__(param_space, evaluator, config, parallel_config)

        # 保存并行配置
        self.parallel_config = parallel_config

        # 初始化并行组件
        self.evaluation_engine = ParallelEvaluationEngine(
            evaluator, self.parallel_config, self.param_names
        )
        self.diversity_engine = ParallelDiversityEngine(self.parallel_config)
        self.performance_monitor = PerformanceMonitor()

        # 保持对最佳个体的引用以兼容性
        self.best_individual = None

        logger.info(
            f"并行遗传算法优化器初始化完成 - 工作进程数: {self.parallel_config.n_workers}"
        )

    def optimize(self, n_calls: int, **kwargs) -> OptimizationResult:
        """并行优化主流程"""
        logger.info(
            f"开始并行遗传算法优化: 评估次数={n_calls}, 工作进程={self.parallel_config.n_workers}"
        )

        with self.performance_monitor.measure_time("total_optimization"):
            return self._optimize_parallel(n_calls, **kwargs)

    def _optimize_parallel(self, n_calls: int, **kwargs) -> OptimizationResult:
        """标准并行优化"""
        # 调整参数
        population_size = min(
            self.parallel_config.population_size, max(10, n_calls // 5)
        )
        max_generations = min(
            self.parallel_config.max_generations, n_calls // population_size
        )

        logger.info(
            f"并行优化参数: 种群大小={population_size}, 最大代数={max_generations}"
        )

        start_time = time.time()

        try:
            # 初始化种群
            with self.performance_monitor.measure_time("initialization"):
                population = self.evolution_engine.initialize_population(
                    population_size
                )

            # 并行评估初始种群
            with self.performance_monitor.measure_time("initial_evaluation"):
                total_evaluations = self.evaluation_engine.evaluate_population_parallel(
                    population
                )

            # 检查初始种群
            valid_individuals = [
                ind for ind in population if ind.fitness != float("inf")
            ]
            if not valid_individuals:
                logger.warning("初始种群中没有有效个体")
                return self._generate_parallel_result(
                    0, total_evaluations, time.time() - start_time
                )

            generation = 0
            generation_stats = []

            # 进化循环
            for generation in range(max_generations):
                # 记录统计信息
                stats = self.evolution_engine.record_generation_stats(
                    population, generation
                )
                if stats:
                    generation_stats.append(stats)

                # 检查收敛
                if self.evolution_engine.check_convergence():
                    logger.info(f"在第{generation}代检测到收敛")
                    break

                # 并行进化
                with self.performance_monitor.measure_time("evolution"):
                    new_population = self._evolve_population_parallel(
                        population, generation, max_generations
                    )

                # 并行评估新个体
                with self.performance_monitor.measure_time("evaluation"):
                    new_evaluations = (
                        self.evaluation_engine.evaluate_population_parallel(
                            new_population
                        )
                    )
                    total_evaluations += new_evaluations

                population = new_population
                self.best_individual = self.evolution_engine.update_best_individual(
                    population, self.best_individual
                )

                # 检查评估次数限制
                if total_evaluations >= n_calls:
                    logger.info(f"达到评估次数限制 ({n_calls})")
                    break

                # 性能监控和日志
                if generation % 10 == 0:
                    self._log_performance_stats(generation, total_evaluations)

            execution_time = time.time() - start_time
            result = self._generate_parallel_result(
                generation + 1, total_evaluations, execution_time, generation_stats
            )

            # 记录性能摘要
            performance_summary = self.performance_monitor.get_performance_summary()
            logger.info(
                f"并行优化完成: 最佳适应度={result.best_score:.6f}, "
                f"总代数={generation + 1}, 总评估次数={total_evaluations}"
            )
            logger.info(f"性能摘要: {performance_summary}")

            return result

        except Exception as e:
            logger.error(f"并行优化异常: {e}")
            execution_time = time.time() - start_time
            return self._generate_parallel_result(0, 0, execution_time, [])

        finally:
            # 清理资源
            self._cleanup_parallel_resources()

    def _evolve_population_parallel(
        self, population: List[Individual], generation: int, max_generations: int
    ) -> List[Individual]:
        """并行进化种群"""
        # 排序种群
        population.sort()

        # 计算种群多样性（并行）
        with self.performance_monitor.measure_time("diversity"):
            diversity = self.diversity_engine.calculate_diversity_parallel(population)

        # 保留精英
        elite_size = self.parallel_config.elite_size
        new_population = [individual.copy() for individual in population[:elite_size]]

        # 多样性保持机制
        if self.parallel_config.diversity_preservation and diversity < 0.1:
            mutation_rate = min(0.5, self.parallel_config.mutation_rate * 2)
            logger.debug(f"低多样性检测，增加变异率至 {mutation_rate:.3f}")
        else:
            mutation_rate = self.parallel_config.mutation_rate

        # 并行生成后代
        offspring_needed = len(population) - elite_size

        if self.parallel_config.parallel_evolution:
            offspring = self._generate_offspring_parallel(
                population, offspring_needed, generation, max_generations, mutation_rate
            )
        else:
            offspring = self._generate_offspring_serial(
                population, offspring_needed, generation, max_generations, mutation_rate
            )

        new_population.extend(offspring)

        return new_population[: len(population)]

    def _generate_offspring_parallel(
        self,
        population: List[Individual],
        offspring_needed: int,
        generation: int,
        max_generations: int,
        mutation_rate: float,
    ) -> List[Individual]:
        """并行生成后代"""
        offspring = []
        batch_size = self.parallel_config.evolution_batch_size

        # 使用线程池并行生成后代（因为操作相对轻量）
        with ThreadPoolExecutor(max_workers=self.parallel_config.n_workers) as executor:
            futures = []

            for batch_start in range(0, offspring_needed, batch_size):
                batch_size_actual = min(batch_size, offspring_needed - batch_start)

                future = executor.submit(
                    self._generate_offspring_batch,
                    population,
                    batch_size_actual,
                    generation,
                    max_generations,
                    mutation_rate,
                )
                futures.append(future)

            # 收集结果
            for future in futures:
                batch_offspring = future.result()
                offspring.extend(batch_offspring)

        return offspring[:offspring_needed]

    def _generate_offspring_serial(
        self,
        population: List[Individual],
        offspring_needed: int,
        generation: int,
        max_generations: int,
        mutation_rate: float,
    ) -> List[Individual]:
        """串行生成后代"""
        return self._generate_offspring_batch(
            population, offspring_needed, generation, max_generations, mutation_rate
        )

    def _generate_offspring_batch(
        self,
        population: List[Individual],
        batch_size: int,
        generation: int,
        max_generations: int,
        mutation_rate: float,
    ) -> List[Individual]:
        """生成一批后代"""
        offspring = []

        for _ in range((batch_size + 1) // 2):  # 确保生成足够的后代
            # 选择父母
            parent1 = self.evolution_engine.selection(population)
            parent2 = self.evolution_engine.selection(population)

            # 确保父母不同
            attempts = 0
            while parent1 is parent2 and attempts < 10:
                parent2 = self.evolution_engine.selection(population)
                attempts += 1

            # 交叉
            child1, child2 = parent1.crossover(
                parent2, self.parallel_config.crossover_rate
            )

            # 变异
            child1.mutate(mutation_rate, generation, max_generations)
            child2.mutate(mutation_rate, generation, max_generations)

            # 设置代数
            child1.generation = generation + 1
            child2.generation = generation + 1

            offspring.extend([child1, child2])

        return offspring[:batch_size]

    def _log_performance_stats(self, generation: int, total_evaluations: int):
        """记录性能统计信息"""
        if self.best_individual:
            best_fitness = self.best_individual.fitness
            diversity = self.diversity_engine.calculate_diversity_parallel(
                [self.best_individual] * 2  # 简化的多样性计算
            )

            memory_usage = self.performance_monitor.get_memory_usage()
            cpu_usage = self.performance_monitor.get_cpu_usage()

            logger.info(
                f"第{generation}代: 最佳适应度={best_fitness:.6f}, "
                f"多样性={diversity:.4f}, 评估次数={total_evaluations}, "
                f"内存使用={memory_usage:.1%}, CPU使用={cpu_usage:.1%}"
            )

    def _generate_parallel_result(
        self,
        total_generations: int,
        total_evaluations: int,
        execution_time: float,
        generation_stats: Optional[List[Dict[str, Any]]] = None,
    ) -> OptimizationResult:
        """生成并行优化结果"""
        if generation_stats is None:
            generation_stats = []

        return self.analyzer.generate_optimization_result(
            best_individual=self.best_individual,
            generation_stats=generation_stats,
            evolution_engine=self.evolution_engine,
            total_generations=total_generations,
            total_evaluations=total_evaluations,
            execution_time=execution_time,
        )

    def _cleanup_parallel_resources(self):
        """清理并行资源"""
        try:
            if hasattr(self, "evaluation_engine"):
                self.evaluation_engine.cleanup()

            if hasattr(self, "diversity_engine"):
                self.diversity_engine.cleanup()

            logger.info("并行资源清理完成")
        except Exception as e:
            logger.warning(f"清理并行资源时出错: {e}")
