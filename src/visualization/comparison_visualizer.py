#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化器比较可视化模块

从compare_optimizers_improved.py中提取的可视化功能
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# 配置日志
logger = logging.getLogger(__name__)

# 安全导入字体配置模块
try:
    from src.utils.font_decorator import with_chinese_font
    DECORATOR_AVAILABLE = True
except ImportError:
    logger.warning("字体装饰器模块未找到")
    DECORATOR_AVAILABLE = False
    
    def with_chinese_font(func):
        return func

# 安全导入分析库和显示配置
ANALYSIS_LIBS_AVAILABLE = False
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.utils.display_config import safe_show, safe_close, configure_matplotlib_for_display
    ANALYSIS_LIBS_AVAILABLE = True
    # 配置matplotlib显示设置
    configure_matplotlib_for_display()
    logger.info("分析库加载成功")
except ImportError as e:
    logger.warning(f"分析库未完全安装: {e}")
    # 创建占位符，避免导入错误
    class MockPandas:
        def DataFrame(self, *args, **kwargs):
            return None
    pd = MockPandas()
    
    # 创建安全显示函数的备用版本
    def safe_show():
        pass
    
    def safe_close():
        pass


class ComparisonVisualizer:
    """优化器比较可视化器"""
    
    def __init__(self, results_dir: Path):
        """
        初始化可视化器
        
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_visualizations(self, 
                                  results: Dict[str, List[Dict[str, Any]]],
                                  optimizers: List[str],
                                  execution_times: Dict[str, List[float]]) -> None:
        """
        生成所有可视化图表
        
        Args:
            results: 优化结果字典
            optimizers: 优化器列表
            execution_times: 执行时间字典
        """
        if not ANALYSIS_LIBS_AVAILABLE:
            logger.warning("可视化库不可用，跳过图表生成")
            return
        
        try:
            # 设置样式
            plt.style.use('default')
            if 'sns' in globals():
                sns.set_palette("husl")
            
            # 1. 性能比较图
            self._plot_performance_comparison(results, optimizers)
            
            # 2. 执行时间比较图
            self._plot_execution_time_comparison(execution_times, optimizers)
            
            # 3. 箱线图
            self._plot_box_plots(results, optimizers)
            
            # 4. 散点图矩阵
            self._plot_scatter_matrix(results, optimizers)
            
            # 5. 收敛性分析
            self._plot_convergence_analysis(results, optimizers)
            
            logger.info("可视化图表已生成")
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
            logger.debug(traceback.format_exc())
    
    @with_chinese_font
    def _plot_performance_comparison(self, 
                                   results: Dict[str, List[Dict[str, Any]]],
                                   optimizers: List[str]) -> None:
        """绘制性能比较图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        optimizer_names = []
        mean_values = []
        std_values = []
        
        for optimizer in optimizers:
            runs = results.get(optimizer, [])
            successful_runs = [r for r in runs if not r.get('failed', False)]
            
            if successful_runs:
                values = [r['best_value'] for r in successful_runs]
                optimizer_names.append(optimizer)
                mean_values.append(np.mean(values))
                std_values.append(np.std(values))
        
        if not optimizer_names:
            logger.warning("没有数据可用于性能比较图")
            safe_close()
            return
        
        # 柱状图（平均值 + 误差棒）
        x_pos = np.arange(len(optimizer_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(optimizer_names)))
        bars = ax1.bar(x_pos, mean_values, yerr=std_values, capsize=5, 
                      alpha=0.8, color=colors)
        ax1.set_xlabel('优化器')
        ax1.set_ylabel('目标值')
        ax1.set_title('优化器性能比较（平均值 ± 标准差）')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(optimizer_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (mean_val, std_val) in enumerate(zip(mean_values, std_values)):
            ax1.text(i, mean_val + std_val + max(mean_values) * 0.01, 
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 散点图（所有运行结果）
        for i, optimizer in enumerate(optimizer_names):
            successful_runs = [r for r in results[optimizer] if not r.get('failed', False)]
            values = [r['best_value'] for r in successful_runs]
            x_scatter = [i + np.random.uniform(-0.2, 0.2) for _ in values]  # 添加抖动
            ax2.scatter(x_scatter, values, alpha=0.6, s=50, 
                       label=optimizer, color=colors[i])
        
        ax2.set_xlabel('优化器')
        ax2.set_ylabel('目标值')
        ax2.set_title('所有运行结果分布')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(optimizer_names, rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        safe_close()
    
    @with_chinese_font
    def _plot_execution_time_comparison(self, 
                                      execution_times: Dict[str, List[float]],
                                      optimizers: List[str]) -> None:
        """绘制执行时间比较图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 准备数据
        optimizer_names = []
        mean_times = []
        std_times = []
        
        for optimizer in optimizers:
            times = execution_times.get(optimizer, [])
            if times:
                optimizer_names.append(optimizer)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))
        
        if not optimizer_names:
            logger.warning("没有数据可用于执行时间比较图")
            safe_close()
            return
        
        # 柱状图
        x_pos = np.arange(len(optimizer_names))
        colors = plt.cm.viridis(np.linspace(0, 1, len(optimizer_names)))
        bars = ax.bar(x_pos, mean_times, yerr=std_times, capsize=5, 
                     alpha=0.8, color=colors)
        
        ax.set_xlabel('优化器')
        ax.set_ylabel('执行时间 (秒)')
        ax.set_title('优化器执行时间比较')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizer_names, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (mean_time, std_time) in enumerate(zip(mean_times, std_times)):
            ax.text(i, mean_time + std_time + max(mean_times) * 0.02, 
                   f'{mean_time:.1f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        safe_close()
    
    @with_chinese_font
    def _plot_box_plots(self, 
                       results: Dict[str, List[Dict[str, Any]]],
                       optimizers: List[str]) -> None:
        """绘制箱线图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        performance_data = []
        time_data = []
        labels = []
        
        for optimizer in optimizers:
            runs = results.get(optimizer, [])
            successful_runs = [r for r in runs if not r.get('failed', False)]
            
            if successful_runs:
                values = [r['best_value'] for r in successful_runs]
                times = [r.get('execution_time', 0) for r in successful_runs]
                
                performance_data.append(values)
                time_data.append(times)
                labels.append(optimizer)
        
        if not performance_data:
            logger.warning("没有数据可用于箱线图")
            safe_close()
            return
        
        # 性能箱线图
        bp1 = ax1.boxplot(performance_data, labels=labels, patch_artist=True)
        ax1.set_xlabel('优化器')
        ax1.set_ylabel('目标值')
        ax1.set_title('优化器性能分布（箱线图）')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 执行时间箱线图
        bp2 = ax2.boxplot(time_data, labels=labels, patch_artist=True)
        ax2.set_xlabel('优化器')
        ax2.set_ylabel('执行时间 (秒)')
        ax2.set_title('优化器执行时间分布（箱线图）')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'box_plots.png', dpi=300, bbox_inches='tight')
        safe_close()
    
    @with_chinese_font
    def _plot_scatter_matrix(self, 
                           results: Dict[str, List[Dict[str, Any]]],
                           optimizers: List[str]) -> None:
        """绘制散点图矩阵"""
        try:
            # 准备数据用于散点图矩阵
            summary_data = []
            
            for optimizer in optimizers:
                runs = results.get(optimizer, [])
                successful_runs = [r for r in runs if not r.get('failed', False)]
                
                if successful_runs:
                    values = [r['best_value'] for r in successful_runs]
                    times = [r.get('execution_time', 0) for r in successful_runs]
                    
                    summary_entry = {
                        'optimizer': optimizer,
                        'mean_best_value': np.mean(values),
                        'std_best_value': np.std(values),
                        'mean_execution_time': np.mean(times),
                        'efficiency_score': np.mean(values) / np.mean(times) if np.mean(times) > 0 else float('inf'),
                        'robustness_score': 1 / (1 + np.std(values)) if np.std(values) > 0 else 1
                    }
                    summary_data.append(summary_entry)
            
            if len(summary_data) < 2:
                logger.warning("数据不足，无法生成散点图矩阵")
                return
            
            # 转换为DataFrame
            comparison_summary = pd.DataFrame(summary_data)
            
            # 选择数值列
            numeric_cols = ['mean_best_value', 'std_best_value', 'mean_execution_time', 
                          'efficiency_score', 'robustness_score']
            
            available_cols = [col for col in numeric_cols if col in comparison_summary.columns]
            
            if len(available_cols) < 2:
                logger.warning("数据不足，无法生成散点图矩阵")
                return
            
            data_for_plot = comparison_summary[available_cols]
            
            fig, axes = plt.subplots(len(available_cols), len(available_cols), 
                                   figsize=(12, 12))
            
            for i, col1 in enumerate(available_cols):
                for j, col2 in enumerate(available_cols):
                    ax = axes[i, j]
                    
                    if i == j:
                        # 对角线：直方图
                        ax.hist(data_for_plot[col1], bins=10, alpha=0.7)
                        ax.set_title(col1)
                    else:
                        # 非对角线：散点图
                        ax.scatter(data_for_plot[col2], data_for_plot[col1], alpha=0.7)
                        
                        # 添加优化器标签
                        for idx, row in comparison_summary.iterrows():
                            ax.annotate(row['optimizer'], 
                                      (row[col2], row[col1]), 
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.7)
                    
                    if i == len(available_cols) - 1:
                        ax.set_xlabel(col2)
                    if j == 0:
                        ax.set_ylabel(col1)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'scatter_matrix.png', dpi=300, bbox_inches='tight')
            safe_close()
            
        except Exception as e:
            logger.warning(f"生成散点图矩阵失败: {e}")
    
    @with_chinese_font
    def _plot_convergence_analysis(self, 
                                 results: Dict[str, List[Dict[str, Any]]],
                                 optimizers: List[str]) -> None:
        """绘制收敛性分析图"""
        # 这是一个简化的收敛分析，基于最终结果
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for optimizer in optimizers:
            runs = results.get(optimizer, [])
            successful_runs = [r for r in runs if not r.get('failed', False)]
            
            if successful_runs:
                values = [r['best_value'] for r in successful_runs]
                # 假设所有运行都有相同的迭代次数
                n_calls = successful_runs[0].get('n_calls', 30) if successful_runs else 30
                iterations = [n_calls] * len(values)
                
                # 添加一些随机噪声来显示分布
                iterations_jittered = [it + np.random.normal(0, n_calls * 0.01) for it in iterations]
                
                ax.scatter(iterations_jittered, values, alpha=0.6, label=optimizer, s=50)
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最终目标值')
        ax.set_title('优化器收敛性分析')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        safe_close()
