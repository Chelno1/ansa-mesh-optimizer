#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遗传算法可视化模块

提供遗传算法优化过程的可视化功能，包括进化曲线、多样性变化等
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 安全导入matplotlib和显示配置
try:
    import matplotlib.pyplot as plt
    import numpy as np

    from ..utils.display_config import DisplayConfig

    MATPLOTLIB_AVAILABLE = True

    # 创建默认显示配置
    _display_config = DisplayConfig()

    def safe_show():
        """安全的显示函数"""
        _display_config.safe_show()

    def safe_close():
        """安全的关闭函数"""
        _display_config.safe_close()

except ImportError:
    MATPLOTLIB_AVAILABLE = False

    def safe_show():
        """安全的显示函数 - 备用版本"""
        pass

    def safe_close():
        """安全的关闭函数 - 备用版本"""
        pass

    logger.warning("matplotlib不可用，无法生成图表")

# 尝试导入字体装饰器模块
try:
    from ..utils.font_decorator import plotting_ready, with_chinese_font

    DECORATOR_AVAILABLE = True
except ImportError:
    logger.warning("字体装饰器模块未找到")
    DECORATOR_AVAILABLE = False

    # 创建空装饰器作为备用
    def with_chinese_font(func):
        return func

    def plotting_ready(backend="Agg", save_original=True):
        def decorator(func):
            return func

        return decorator


class GeneticVisualizer:
    """遗传算法可视化器"""

    def __init__(
        self,
        best_fitness_history: List[float],
        diversity_history: List[float],
        generation_stats: List[Dict[str, Any]],
        genetic_config: Optional[Dict] = None,
    ):
        """
        初始化可视化器

        Args:
            best_fitness_history: 最佳适应度历史
            diversity_history: 多样性历史
            generation_stats: 代数统计信息
            genetic_config: 遗传算法配置
        """
        self.best_fitness_history = best_fitness_history
        self.diversity_history = diversity_history
        self.generation_stats = generation_stats
        self.genetic_config = genetic_config or {}

    @plotting_ready(backend="TkAgg", save_original=True)
    def plot_evolution(
        self, save_path: Optional[str] = None, show_diversity: bool = True
    ) -> None:
        """绘制进化过程 - 使用增强装饰器"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib不可用，无法绘制进化图表")
            return

        try:
            fig_size = (15, 10) if show_diversity else (12, 8)
            fig, axes = plt.subplots(2, 2, figsize=fig_size)

            # 最佳适应度变化
            axes[0, 0].plot(
                self.best_fitness_history, "b-", linewidth=2, label="最佳适应度"
            )
            axes[0, 0].set_xlabel("代数")
            axes[0, 0].set_ylabel("适应度")
            axes[0, 0].set_title("最佳适应度进化曲线")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            # 添加重启点标记
            restart_generations = self.genetic_config.get("restart_generations", 20)
            restart_points = []
            for i, gen_stat in enumerate(self.generation_stats):
                if i > 0 and gen_stat["generation"] % restart_generations == 0:
                    restart_points.append(i)

            for point in restart_points:
                if point < len(self.best_fitness_history):
                    axes[0, 0].axvline(
                        x=point, color="red", linestyle="--", alpha=0.7, label="重启点"
                    )

            # 种群多样性变化
            if show_diversity and self.diversity_history:
                axes[0, 1].plot(
                    self.diversity_history, "g-", linewidth=2, label="种群多样性"
                )
                axes[0, 1].set_xlabel("代数")
                axes[0, 1].set_ylabel("多样性")
                axes[0, 1].set_title("种群多样性变化")
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()

            # 适应度统计（最后一代）
            if self.generation_stats:
                final_stats = self.generation_stats[-1]

                # 模拟最终种群的适应度分布
                mean_fitness = final_stats["mean_fitness"]
                std_fitness = final_stats["std_fitness"]

                # 生成模拟分布数据
                simulated_fitness = np.random.normal(mean_fitness, std_fitness, 100)
                simulated_fitness = np.clip(
                    simulated_fitness,
                    final_stats["best_fitness"],
                    final_stats["worst_fitness"],
                )

                axes[1, 0].hist(
                    simulated_fitness,
                    bins=20,
                    alpha=0.7,
                    color="skyblue",
                    edgecolor="black",
                )
                axes[1, 0].axvline(
                    x=final_stats["best_fitness"],
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f'最佳值: {final_stats["best_fitness"]:.4f}',
                )
                axes[1, 0].axvline(
                    x=mean_fitness,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"平均值: {mean_fitness:.4f}",
                )
                axes[1, 0].set_xlabel("适应度")
                axes[1, 0].set_ylabel("频次")
                axes[1, 0].set_title("最终种群适应度分布（模拟）")
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

                axes[1, 1].plot(
                    improvement_rates, "purple", linewidth=2, label="改进率"
                )
                axes[1, 1].set_xlabel("代数")
                axes[1, 1].set_ylabel("改进率")
                axes[1, 1].set_title(f"滚动改进率 (窗口大小: {window_size})")
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"进化图表已保存: {save_path}")

            if safe_show:
                safe_show()
            else:
                plt.show()

        except Exception as e:
            logger.warning(f"绘制进化图表失败: {e}")

    @plotting_ready(backend="TkAgg", save_original=True)
    def plot_convergence_analysis(self, save_path: Optional[str] = None) -> None:
        """绘制收敛性分析图"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib不可用，无法绘制收敛分析图表")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 适应度改进曲线
            if len(self.best_fitness_history) > 1:
                improvements = []
                for i in range(1, len(self.best_fitness_history)):
                    prev_fitness = self.best_fitness_history[i - 1]
                    curr_fitness = self.best_fitness_history[i]
                    improvement = prev_fitness - curr_fitness
                    improvements.append(improvement)

                axes[0, 0].plot(improvements, "b-", linewidth=2, label="代际改进")
                axes[0, 0].set_xlabel("代数")
                axes[0, 0].set_ylabel("适应度改进")
                axes[0, 0].set_title("代际适应度改进")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()

            # 收敛速度分析
            if len(self.best_fitness_history) > 5:
                convergence_rates = []
                window = 5
                for i in range(window, len(self.best_fitness_history)):
                    recent_variance = np.var(self.best_fitness_history[i - window : i])
                    convergence_rates.append(recent_variance)

                axes[0, 1].plot(convergence_rates, "r-", linewidth=2, label="收敛速度")
                axes[0, 1].set_xlabel("代数")
                axes[0, 1].set_ylabel("方差")
                axes[0, 1].set_title(f"收敛速度分析 (窗口大小: {window})")
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()

            # 多样性与适应度关系
            if self.diversity_history and len(self.diversity_history) == len(
                self.best_fitness_history
            ):
                axes[1, 0].scatter(
                    self.diversity_history,
                    self.best_fitness_history,
                    alpha=0.6,
                    color="green",
                    s=30,
                )
                axes[1, 0].set_xlabel("种群多样性")
                axes[1, 0].set_ylabel("最佳适应度")
                axes[1, 0].set_title("多样性与适应度关系")
                axes[1, 0].grid(True, alpha=0.3)

            # 统计信息汇总
            if self.generation_stats:
                generations = [stat["generation"] for stat in self.generation_stats]
                mean_fitness = [stat["mean_fitness"] for stat in self.generation_stats]
                std_fitness = [stat["std_fitness"] for stat in self.generation_stats]

                axes[1, 1].plot(
                    generations, mean_fitness, "b-", linewidth=2, label="平均适应度"
                )
                axes[1, 1].fill_between(
                    generations,
                    [m - s for m, s in zip(mean_fitness, std_fitness)],
                    [m + s for m, s in zip(mean_fitness, std_fitness)],
                    alpha=0.3,
                    color="blue",
                )
                axes[1, 1].set_xlabel("代数")
                axes[1, 1].set_ylabel("适应度")
                axes[1, 1].set_title("种群适应度统计")
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"收敛分析图表已保存: {save_path}")

            if safe_show:
                safe_show()
            else:
                plt.show()

        except Exception as e:
            logger.warning(f"绘制收敛分析图表失败: {e}")


def plot_evolution_history(
    best_fitness_history: List[float],
    diversity_history: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    绘制进化历史的简化版本

    Args:
        best_fitness_history: 最佳适应度历史
        diversity_history: 多样性历史（可选）
        save_path: 保存路径（可选）
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib不可用，无法绘制进化历史图表")
        return

    try:
        fig, axes = plt.subplots(1, 2 if diversity_history else 1, figsize=(12, 5))
        if not diversity_history:
            axes = [axes]

        # 最佳适应度变化
        axes[0].plot(best_fitness_history, "b-", linewidth=2, label="最佳适应度")
        axes[0].set_xlabel("代数")
        axes[0].set_ylabel("适应度")
        axes[0].set_title("最佳适应度进化曲线")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 种群多样性变化（如果提供）
        if diversity_history:
            axes[1].plot(diversity_history, "g-", linewidth=2, label="种群多样性")
            axes[1].set_xlabel("代数")
            axes[1].set_ylabel("多样性")
            axes[1].set_title("种群多样性变化")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"进化历史图表已保存: {save_path}")

        if safe_show:
            safe_show()
        else:
            plt.show()

    except Exception as e:
        logger.warning(f"绘制进化历史图表失败: {e}")


def export_evolution_data(
    best_fitness_history: List[float],
    diversity_history: List[float],
    generation_stats: List[Dict[str, Any]],
    genetic_config: Dict[str, Any],
    best_individual_info: Dict[str, Any],
    metadata: Dict[str, Any],
    filename: str,
) -> None:
    """
    导出进化数据 - 增强版本

    Args:
        best_fitness_history: 最佳适应度历史
        diversity_history: 多样性历史
        generation_stats: 代数统计信息
        genetic_config: 遗传算法配置
        best_individual_info: 最佳个体信息
        metadata: 元数据
        filename: 导出文件名
    """
    try:
        export_data = {
            "config": genetic_config,
            "results": {
                "best_fitness_history": best_fitness_history,
                "diversity_history": diversity_history,
                "generation_stats": generation_stats,
            },
            "best_individual": best_individual_info,
            "metadata": {**metadata, "export_timestamp": datetime.now().isoformat()},
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"进化数据已导出: {filename}")

    except Exception as e:
        logger.error(f"导出进化数据失败: {e}")


def create_genetic_visualizer(
    best_fitness_history: List[float],
    diversity_history: List[float],
    generation_stats: List[Dict[str, Any]],
    genetic_config: Optional[Dict] = None,
) -> GeneticVisualizer:
    """
    创建遗传算法可视化器的工厂函数

    Args:
        best_fitness_history: 最佳适应度历史
        diversity_history: 多样性历史
        generation_stats: 代数统计信息
        genetic_config: 遗传算法配置

    Returns:
        配置好的可视化器
    """
    return GeneticVisualizer(
        best_fitness_history=best_fitness_history,
        diversity_history=diversity_history,
        generation_stats=generation_stats,
        genetic_config=genetic_config,
    )
