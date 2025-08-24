#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法配置类

提供遗传算法的配置参数和验证逻辑
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "population_size": self.population_size,
            "elite_size": self.elite_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "tournament_size": self.tournament_size,
            "max_generations": self.max_generations,
            "convergence_threshold": self.convergence_threshold,
            "convergence_patience": self.convergence_patience,
            "adaptive_mutation": self.adaptive_mutation,
            "diversity_preservation": self.diversity_preservation,
            "niching_enabled": self.niching_enabled,
            "restart_enabled": self.restart_enabled,
            "restart_generations": self.restart_generations,
            "max_history_size": self.max_history_size,
            "save_full_history": self.save_full_history,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GeneticConfig":
        """从字典创建配置"""
        # 过滤掉不存在的字段
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)


def create_fast_genetic_config() -> GeneticConfig:
    """创建快速遗传算法配置"""
    return GeneticConfig(
        population_size=20,
        max_generations=50,
        elite_size=2,
        convergence_patience=5,
        restart_generations=10,
        max_history_size=25,
    )


def create_thorough_genetic_config() -> GeneticConfig:
    """创建彻底遗传算法配置"""
    return GeneticConfig(
        population_size=100,
        max_generations=200,
        elite_size=10,
        convergence_patience=20,
        restart_generations=40,
        max_history_size=100,
        convergence_threshold=1e-8,
    )


def create_adaptive_genetic_config() -> GeneticConfig:
    """创建自适应遗传算法配置"""
    return GeneticConfig(
        population_size=50,
        max_generations=100,
        elite_size=5,
        adaptive_mutation=True,
        diversity_preservation=True,
        restart_enabled=True,
        niching_enabled=True,
    )
