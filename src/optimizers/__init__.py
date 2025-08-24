"""
优化器策略模块

提供各种优化算法的策略实现，包括模块化的遗传算法组件
"""

from .evolution import (
    EvolutionEngine,
    gaussian_mutation,
    tournament_selection,
    uniform_crossover,
)

# 新的模块化组件
from .evolution_loop import EvolutionLoop

# 新的模块化遗传算法组件
from .genetic_config import (
    GeneticConfig,
    create_adaptive_genetic_config,
    create_fast_genetic_config,
    create_thorough_genetic_config,
)
from .genetic_optimizer import GeneticOptimizer, create_genetic_optimizer
from .genetic_visualization import (
    GeneticVisualizer,
    create_genetic_visualizer,
    export_evolution_data,
    plot_evolution_history,
)
from .individual import (
    Individual,
    calculate_population_diversity,
    create_individual,
    create_lhs_population,
    create_population,
)
from .optimizer_analysis import OptimizerAnalyzer
from .optimizer_config import (
    OptimizationResult,
    OptimizerConfig,
    create_default_config,
    create_fast_config,
    create_parallel_config,
    create_thorough_config,
)
from .optimizer_strategies import (
    BayesianOptimizerStrategy,
    ForestOptimizerStrategy,
    GeneticOptimizerStrategy,
    OptimizerFactory,
    OptimizerStrategy,
    ParallelOptimizerStrategy,
    RandomOptimizerStrategy,
)

__all__ = [
    # 原有的优化器策略
    "OptimizerStrategy",
    "BayesianOptimizerStrategy",
    "RandomOptimizerStrategy",
    "ForestOptimizerStrategy",
    "GeneticOptimizerStrategy",
    "ParallelOptimizerStrategy",
    "OptimizerFactory",
    "OptimizerConfig",
    "OptimizationResult",
    "create_default_config",
    "create_fast_config",
    "create_thorough_config",
    "create_parallel_config",
    # 遗传算法模块化组件
    "GeneticConfig",
    "create_fast_genetic_config",
    "create_thorough_genetic_config",
    "create_adaptive_genetic_config",
    "Individual",
    "create_individual",
    "create_population",
    "create_lhs_population",
    "calculate_population_diversity",
    "EvolutionEngine",
    "tournament_selection",
    "uniform_crossover",
    "gaussian_mutation",
    "GeneticVisualizer",
    "plot_evolution_history",
    "export_evolution_data",
    "create_genetic_visualizer",
    # 遗传算法优化器
    "GeneticOptimizer",
    "create_genetic_optimizer",
    # 新的模块化组件
    "EvolutionLoop",
    "OptimizerAnalyzer",
]
