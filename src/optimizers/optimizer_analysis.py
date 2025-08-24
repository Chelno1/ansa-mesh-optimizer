#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法优化结果分析模块

负责处理优化结果的汇总、分析和统计功能，包括历史记录构建、收敛信息分析等。

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .evolution import EvolutionEngine
from .genetic_config import GeneticConfig
from .individual import Individual
from .optimizer_config import OptimizationResult

logger = logging.getLogger(__name__)


class OptimizerAnalyzer:
    """遗传算法优化结果分析器"""

    def __init__(
        self,
        param_names: List[str],
        bounds: List[Tuple[float, float]],
        genetic_config: GeneticConfig,
    ):
        """
        初始化优化结果分析器

        Args:
            param_names: 参数名称列表
            bounds: 参数边界列表
            genetic_config: 遗传算法配置
        """
        self.param_names = param_names
        self.bounds = bounds
        self.genetic_config = genetic_config

        logger.debug("优化结果分析器初始化完成")

    def generate_optimization_result(
        self,
        best_individual: Optional[Individual],
        generation_stats: List[Dict[str, Any]],
        evolution_engine: EvolutionEngine,
        total_generations: int,
        total_evaluations: int,
        execution_time: float,
    ) -> OptimizationResult:
        """
        生成完整的优化结果

        Args:
            best_individual: 最佳个体
            generation_stats: 代数统计信息
            evolution_engine: 进化引擎
            total_generations: 总代数
            total_evaluations: 总评估次数
            execution_time: 执行时间

        Returns:
            优化结果对象
        """
        if best_individual is None:
            return self._generate_default_result(generation_stats)

        # 构建优化历史记录
        history = self._build_optimization_history(best_individual, generation_stats)

        # 获取多样性指标
        diversity_metrics = evolution_engine.get_diversity_metrics()

        # 构建收敛信息
        convergence_info = self._build_convergence_info(
            evolution_engine,
            diversity_metrics,
            total_generations,
            total_evaluations,
            execution_time,
        )

        return OptimizationResult.from_genetic_result(
            best_params=best_individual.genes.copy(),
            best_value=(
                best_individual.fitness
                if best_individual.fitness is not None
                else float("inf")
            ),
            optimization_history=history,
            parameter_names=self.param_names,
            parameter_ranges=self.bounds,
            generation_stats=generation_stats,
            convergence_info=convergence_info,
        )

    def _generate_default_result(
        self, generation_stats: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """
        生成默认的优化结果（当没有找到有效个体时）

        Args:
            generation_stats: 代数统计信息

        Returns:
            默认优化结果
        """
        logger.error("优化过程中未找到有效的最佳个体")

        # 创建默认的最佳参数（使用参数空间中点）
        default_genes = []
        for low, high in self.bounds:
            default_genes.append((low + high) / 2)

        # 创建默认历史记录
        default_history = [
            {
                "generation": 0,
                "parameters": {
                    self.param_names[i]: default_genes[i]
                    for i in range(len(self.param_names))
                },
                "result": float("inf"),
                "fitness": float("inf"),
                "stats": None,
            }
        ]

        return OptimizationResult.from_genetic_result(
            best_params=default_genes,
            best_value=float("inf"),
            optimization_history=default_history,
            parameter_names=self.param_names,
            parameter_ranges=self.bounds,
            generation_stats=generation_stats,
            convergence_info={
                "converged": False,
                "convergence_generation": -1,
                "final_diversity": 0.0,
                "error": "No valid individuals found during optimization",
            },
        )

    def _build_optimization_history(
        self, best_individual: Individual, generation_stats: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        构建优化历史记录

        Args:
            best_individual: 最佳个体
            generation_stats: 代数统计信息

        Returns:
            优化历史记录列表
        """
        history = []

        if generation_stats:
            for i, stats in enumerate(generation_stats):
                history.append(
                    {
                        "generation": i,
                        "parameters": best_individual.to_params(self.param_names),
                        "result": stats.get("best_fitness", float("inf")),
                        "fitness": stats.get("best_fitness", float("inf")),
                        "stats": stats,
                    }
                )

        return history

    def _build_convergence_info(
        self,
        evolution_engine: EvolutionEngine,
        diversity_metrics: Dict[str, List[float]],
        total_generations: int,
        total_evaluations: int,
        execution_time: float,
    ) -> Dict[str, Any]:
        """
        构建收敛信息

        Args:
            evolution_engine: 进化引擎
            diversity_metrics: 多样性指标
            total_generations: 总代数
            total_evaluations: 总评估次数
            execution_time: 执行时间

        Returns:
            收敛信息字典
        """
        return {
            "converged": evolution_engine.convergence_counter
            >= self.genetic_config.convergence_patience,
            "convergence_generation": total_generations,
            "final_diversity": (
                diversity_metrics["diversity_history"][-1]
                if diversity_metrics["diversity_history"]
                else 0.0
            ),
            "total_generations": total_generations,
            "total_evaluations": total_evaluations,
            "execution_time": execution_time,
            "restart_count": evolution_engine.restart_count,
            "improvement_ratio": evolution_engine.calculate_improvement_ratio(),
        }

    def analyze_generation_performance(
        self, generation_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        分析代数性能

        Args:
            generation_stats: 代数统计信息

        Returns:
            性能分析结果
        """
        if not generation_stats:
            return {
                "total_generations": 0,
                "improvement_trend": "no_data",
                "convergence_rate": 0.0,
                "best_generation": -1,
                "performance_metrics": {},
            }

        best_fitness_values = [
            stats.get("best_fitness", float("inf")) for stats in generation_stats
        ]

        # 找到最佳代数
        best_generation = 0
        best_fitness = float("inf")
        for i, fitness in enumerate(best_fitness_values):
            if fitness < best_fitness:
                best_fitness = fitness
                best_generation = i

        # 计算改进趋势
        improvement_trend = self._calculate_improvement_trend(best_fitness_values)

        # 计算收敛速度
        convergence_rate = self._calculate_convergence_rate(best_fitness_values)

        return {
            "total_generations": len(generation_stats),
            "improvement_trend": improvement_trend,
            "convergence_rate": convergence_rate,
            "best_generation": best_generation,
            "best_fitness": best_fitness,
            "performance_metrics": {
                "initial_fitness": (
                    best_fitness_values[0] if best_fitness_values else float("inf")
                ),
                "final_fitness": (
                    best_fitness_values[-1] if best_fitness_values else float("inf")
                ),
                "total_improvement": (
                    best_fitness_values[0] - best_fitness_values[-1]
                    if len(best_fitness_values) > 1
                    else 0.0
                ),
                "average_fitness": (
                    sum(best_fitness_values) / len(best_fitness_values)
                    if best_fitness_values
                    else float("inf")
                ),
            },
        }

    def _calculate_improvement_trend(self, fitness_values: List[float]) -> str:
        """
        计算改进趋势

        Args:
            fitness_values: 适应度值列表

        Returns:
            改进趋势字符串
        """
        if len(fitness_values) < 2:
            return "insufficient_data"

        improvements = 0
        total_comparisons = 0

        for i in range(1, len(fitness_values)):
            if fitness_values[i] != float("inf") and fitness_values[i - 1] != float(
                "inf"
            ):
                total_comparisons += 1
                if fitness_values[i] < fitness_values[i - 1]:
                    improvements += 1

        if total_comparisons == 0:
            return "no_valid_data"

        improvement_ratio = improvements / total_comparisons

        if improvement_ratio > 0.6:
            return "improving"
        elif improvement_ratio > 0.3:
            return "slowly_improving"
        elif improvement_ratio > 0.1:
            return "stagnating"
        else:
            return "declining"

    def _calculate_convergence_rate(self, fitness_values: List[float]) -> float:
        """
        计算收敛速度

        Args:
            fitness_values: 适应度值列表

        Returns:
            收敛速度（0-1之间的值）
        """
        if len(fitness_values) < 10:
            return 0.0

        # 计算最近10代的方差作为收敛指标
        recent_values = fitness_values[-10:]
        valid_values = [v for v in recent_values if v != float("inf")]

        if len(valid_values) < 5:
            return 0.0

        # 计算变异系数
        mean_val = sum(valid_values) / len(valid_values)
        if mean_val == 0:
            return 1.0

        variance = sum((v - mean_val) ** 2 for v in valid_values) / len(valid_values)
        std_dev = variance**0.5
        coefficient_of_variation = std_dev / abs(mean_val)

        # 转换为收敛率（变异系数越小，收敛率越高）
        convergence_rate = max(0.0, min(1.0, 1.0 - coefficient_of_variation))

        return convergence_rate

    def generate_summary_report(
        self, result: OptimizationResult, generation_stats: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成优化总结报告

        Args:
            result: 优化结果
            generation_stats: 代数统计信息

        Returns:
            总结报告字典
        """
        performance_analysis = self.analyze_generation_performance(generation_stats)

        convergence_info = result.convergence_info or {}

        return {
            "optimization_summary": {
                "best_score": result.best_score,
                "total_generations": convergence_info.get("total_generations", 0),
                "total_evaluations": convergence_info.get("total_evaluations", 0),
                "execution_time": convergence_info.get("execution_time", 0.0),
                "converged": convergence_info.get("converged", False),
                "restart_count": convergence_info.get("restart_count", 0),
            },
            "performance_analysis": performance_analysis,
            "best_parameters": dict(zip(self.param_names, result.best_params)),
            "algorithm_efficiency": {
                "evaluations_per_second": (
                    convergence_info.get("total_evaluations", 0)
                    / max(convergence_info.get("execution_time", 1), 0.001)
                ),
                "improvement_ratio": convergence_info.get("improvement_ratio", 0.0),
                "final_diversity": convergence_info.get("final_diversity", 0.0),
            },
        }
