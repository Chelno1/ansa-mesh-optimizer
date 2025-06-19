#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的优化器比较工具

作者: Chel
创建日期: 2025-06-19
版本: 1.1.0
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import warnings

# 第三方库导入（带错误处理）
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    ANALYSIS_LIBS_AVAILABLE = True
except ImportError:
    ANALYSIS_LIBS_AVAILABLE = False
    warnings.warn("分析库（pandas, matplotlib, seaborn）未完全安装，某些功能可能不可用")

# 本地模块导入
from ansa_mesh_optimizer_improved import MeshOptimizer, optimize_mesh_parameters
from config import config_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationComparison:
    """优化器比较分析类"""
    
    def __init__(self, 
                 optimizers: List[str] = None,
                 n_calls: int = 30,
                 evaluator_type: str = 'mock',
                 n_runs: int = 1,
                 use_cache: bool = False):
        """
        初始化优化器比较
        
        Args:
            optimizers: 要比较的优化器列表
            n_calls: 每个优化器的迭代次数
            evaluator_type: 评估器类型
            n_runs: 每个优化器的运行次数（用于统计分析）
            use_cache: 是否使用缓存
        """
        self.optimizers = optimizers or ['bayesian', 'random', 'forest', 'genetic']
        self.n_calls = n_calls
        self.evaluator_type = evaluator_type
        self.n_runs = n_runs
        self.use_cache = use_cache
        
        # 结果存储
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.comparison_summary: Optional[pd.DataFrame] = None
        
        # 创建结果目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f"comparison_results_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"比较结果将保存到: {self.results_dir}")
    
    def run_comparison(self, 
                      run_sensitivity_analysis: bool = True,
                      parallel_execution: bool = False,
                      **kwargs) -> Dict[str, Any]:
        """
        运行优化器比较
        
        Args:
            run_sensitivity_analysis: 是否运行敏感性分析
            parallel_execution: 是否并行执行（实验性功能）
            **kwargs: 其他优化器参数
            
        Returns:
            比较结果字典
        """
        logger.info(f"开始比较 {len(self.optimizers)} 个优化器")
        logger.info(f"每个优化器运行 {self.n_runs} 次，每次 {self.n_calls} 次迭代")
        
        total_start_time = time.time()
        
        for optimizer in self.optimizers:
            self.results[optimizer] = []
            
            logger.info(f"\n{'='*50}")
            logger.info(f"测试优化器: {optimizer}")
            logger.info(f"{'='*50}")
            
            for run_idx in range(self.n_runs):
                logger.info(f"运行 {run_idx + 1}/{self.n_runs}")
                
                try:
                    # 为每次运行设置不同的随机种子
                    run_config = config_manager.optimization_config
                    run_config.random_state = run_config.random_state + run_idx
                    
                    start_time = time.time()
                    
                    # 创建独立的优化器实例
                    mesh_optimizer = MeshOptimizer(
                        config=run_config,
                        evaluator_type=self.evaluator_type,
                        use_cache=self.use_cache
                    )
                    
                    # 执行优化
                    result = mesh_optimizer.optimize(
                        optimizer=optimizer,
                        n_calls=self.n_calls,
                        **kwargs
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # 运行敏感性分析（如果启用且是第一次运行）
                    sensitivity_results = None
                    if run_sensitivity_analysis and run_idx == 0:
                        try:
                            sensitivity_results = mesh_optimizer.sensitivity_analysis(
                                result['best_params'],
                                n_trials=5,
                                noise_level=0.1
                            )
                        except Exception as e:
                            logger.warning(f"敏感性分析失败: {e}")
                    
                    # 记录结果
                    run_result = {
                        'run_index': run_idx,
                        'optimizer': optimizer,
                        'optimizer_name': result['optimizer_name'],
                        'best_value': result['best_value'],
                        'best_params': result['best_params'],
                        'execution_time': execution_time,
                        'n_calls': self.n_calls,
                        'evaluator_type': self.evaluator_type,
                        'sensitivity_results': sensitivity_results,
                        'cache_stats': mesh_optimizer.cache.get_stats() if mesh_optimizer.cache else None,
                        'early_stopping_info': mesh_optimizer.early_stopping.get_best_result() if mesh_optimizer.early_stopping else None
                    }
                    
                    self.results[optimizer].append(run_result)
                    
                    logger.info(f"运行完成: 最佳值={result['best_value']:.6f}, "
                              f"时间={execution_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"优化器 {optimizer} 运行 {run_idx + 1} 失败: {e}")
                    
                    # 记录失败结果
                    error_result = {
                        'run_index': run_idx,
                        'optimizer': optimizer,
                        'error': str(e),
                        'best_value': float('inf'),
                        'execution_time': 0,
                        'failed': True
                    }
                    
                    self.results[optimizer].append(error_result)
        
        total_execution_time = time.time() - total_start_time
        
        # 生成比较摘要
        self._generate_comparison_summary()
        
        # 保存结果
        self._save_results()
        
        # 生成可视化报告
        if ANALYSIS_LIBS_AVAILABLE:
            self._generate_visualizations()
            self._generate_statistical_analysis()
        
        logger.info(f"\n比较完成！总执行时间: {total_execution_time:.2f}秒")
        logger.info(f"详细结果保存在: {self.results_dir}")
        
        return {
            'results': self.results,
            'summary': self.comparison_summary.to_dict() if self.comparison_summary is not None else None,
            'results_dir': str(self.results_dir),
            'total_execution_time': total_execution_time
        }
    
    def _generate_comparison_summary(self) -> None:
        """生成比较摘要"""
        if not ANALYSIS_LIBS_AVAILABLE:
            logger.warning("pandas不可用，跳过摘要生成")
            return
        
        summary_data = []
        
        for optimizer, runs in self.results.items():
            # 过滤成功的运行
            successful_runs = [r for r in runs if not r.get('failed', False)]
            
            if not successful_runs:
                logger.warning(f"优化器 {optimizer} 没有成功的运行")
                continue
            
            # 计算统计指标
            best_values = [r['best_value'] for r in successful_runs]
            execution_times = [r['execution_time'] for r in successful_runs]
            
            summary_data.append({
                'optimizer': optimizer,
                'successful_runs': len(successful_runs),
                'failed_runs': len(runs) - len(successful_runs),
                'mean_best_value': np.mean(best_values),
                'std_best_value': np.std(best_values),
                'min_best_value': np.min(best_values),
                'max_best_value': np.max(best_values),
                'median_best_value': np.median(best_values),
                'mean_execution_time': np.mean(execution_times),
                'std_execution_time': np.std(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times),
                'efficiency_score': np.mean(best_values) / np.mean(execution_times)  # 简单的效率指标
            })
        
        self.comparison_summary = pd.DataFrame(summary_data)
        
        # 排序（按平均最佳值）
        self.comparison_summary = self.comparison_summary.sort_values('mean_best_value')
        
        logger.info("\n比较摘要:")
        logger.info(self.comparison_summary.to_string(index=False))
    
    def _save_results(self) -> None:
        """保存结果到文件"""
        try:
            # 保存原始结果（JSON格式）
            import json
            
            # 准备可序列化的数据
            serializable_results = {}
            for optimizer, runs in self.results.items():
                serializable_results[optimizer] = []
                for run in runs:
                    # 移除不可序列化的对象
                    clean_run = {k: v for k, v in run.items() 
                               if k not in ['sensitivity_results', 'cache_stats', 'early_stopping_info']}
                    serializable_results[optimizer].append(clean_run)
            
            with open(self.results_dir / 'raw_results.json', 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # 保存摘要（CSV格式）
            if self.comparison_summary is not None:
                self.comparison_summary.to_csv(
                    self.results_dir / 'comparison_summary.csv', 
                    index=False
                )
            
            # 保存详细的文本报告
            self._save_text_report()
            
            logger.info("结果已保存到文件")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _save_text_report(self) -> None:
        """保存详细的文本报告"""
        report_file = self.results_dir / 'detailed_report.txt'
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("优化器比较详细报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"比较配置:\n")
                f.write(f"  优化器: {self.optimizers}\n")
                f.write(f"  迭代次数: {self.n_calls}\n")
                f.write(f"  运行次数: {self.n_runs}\n")
                f.write(f"  评估器类型: {self.evaluator_type}\n")
                f.write(f"  使用缓存: {self.use_cache}\n\n")
                
                # 摘要统计
                if self.comparison_summary is not None:
                    f.write("摘要统计:\n")
                    f.write(self.comparison_summary.to_string(index=False))
                    f.write("\n\n")
                
                # 详细结果
                for optimizer, runs in self.results.items():
                    f.write(f"{optimizer} 详细结果:\n")
                    f.write("-" * 40 + "\n")
                    
                    for i, run in enumerate(runs):
                        f.write(f"  运行 {i+1}:\n")
                        if run.get('failed', False):
                            f.write(f"    状态: 失败\n")
                            f.write(f"    错误: {run.get('error', 'Unknown')}\n")
                        else:
                            f.write(f"    最佳值: {run['best_value']:.6f}\n")
                            f.write(f"    执行时间: {run['execution_time']:.2f}s\n")
                            f.write(f"    最佳参数: {run['best_params']}\n")
                        f.write("\n")
                    
                    f.write("\n")
            
            logger.info(f"详细报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"保存详细报告失败: {e}")
    
    def _generate_visualizations(self) -> None:
        """生成可视化图表"""
        if not ANALYSIS_LIBS_AVAILABLE:
            logger.warning("可视化库不可用，跳过图表生成")
            return
        
        try:
            # 设置样式
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. 性能比较图
            self._plot_performance_comparison()
            
            # 2. 执行时间比较图
            self._plot_execution_time_comparison()
            
            # 3. 收敛性分析图
            self._plot_convergence_analysis()
            
            # 4. 参数分布图
            self._plot_parameter_distributions()
            
            # 5. 箱线图
            self._plot_box_plots()
            
            logger.info("可视化图表已生成")
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
    
    def _plot_performance_comparison(self) -> None:
        """绘制性能比较图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        optimizers = []
        mean_values = []
        std_values = []
        
        for optimizer, runs in self.results.items():
            successful_runs = [r for r in runs if not r.get('failed', False)]
            if successful_runs:
                values = [r['best_value'] for r in successful_runs]
                optimizers.append(optimizer)
                mean_values.append(np.mean(values))
                std_values.append(np.std(values))
        
        # 柱状图（平均值 + 误差棒）
        x_pos = np.arange(len(optimizers))
        bars = ax1.bar(x_pos, mean_values, yerr=std_values, capsize=5, 
                      alpha=0.8, color=sns.color_palette("husl", len(optimizers)))
        ax1.set_xlabel('优化器')
        ax1.set_ylabel('目标值')
        ax1.set_title('优化器性能比较（平均值 ± 标准差）')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(optimizers, rotation=45)
        
        # 添加数值标签
        for i, (mean_val, std_val) in enumerate(zip(mean_values, std_values)):
            ax1.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}', 
                    ha='center', va='bottom')
        
        # 散点图（所有运行结果）
        for i, optimizer in enumerate(optimizers):
            successful_runs = [r for r in self.results[optimizer] if not r.get('failed', False)]
            values = [r['best_value'] for r in successful_runs]
            x_scatter = [i] * len(values)
            ax2.scatter(x_scatter, values, alpha=0.6, s=50, 
                       label=optimizer, color=bars[i].get_facecolor())
        
        ax2.set_xlabel('优化器')
        ax2.set_ylabel('目标值')
        ax2.set_title('所有运行结果分布')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(optimizers, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_time_comparison(self) -> None:
        """绘制执行时间比较图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 准备数据
        optimizers = []
        mean_times = []
        std_times = []
        
        for optimizer, runs in self.results.items():
            successful_runs = [r for r in runs if not r.get('failed', False)]
            if successful_runs:
                times = [r['execution_time'] for r in successful_runs]
                optimizers.append(optimizer)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))
        
        # 柱状图
        x_pos = np.arange(len(optimizers))
        bars = ax.bar(x_pos, mean_times, yerr=std_times, capsize=5, 
                     alpha=0.8, color=sns.color_palette("viridis", len(optimizers)))
        
        ax.set_xlabel('优化器')
        ax.set_ylabel('执行时间 (秒)')
        ax.set_title('优化器执行时间比较')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(optimizers, rotation=45)
        
        # 添加数值标签
        for i, (mean_time, std_time) in enumerate(zip(mean_times, std_times)):
            ax.text(i, mean_time + std_time + 0.1, f'{mean_time:.1f}s', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_analysis(self) -> None:
        """绘制收敛性分析图"""
        # 这里需要从优化器历史中提取收敛数据
        # 简化版本：显示最终值与迭代次数的关系
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for optimizer, runs in self.results.items():
            successful_runs = [r for r in runs if not r.get('failed', False)]
            if successful_runs:
                values = [r['best_value'] for r in successful_runs]
                iterations = [self.n_calls] * len(values)
                
                # 添加一些随机噪声来显示分布
                iterations_jittered = [it + np.random.normal(0, 0.5) for it in iterations]
                
                ax.scatter(iterations_jittered, values, alpha=0.6, label=optimizer, s=50)
        
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最终目标值')
        ax.set_title('优化器收敛性分析')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_distributions(self) -> None:
        """绘制参数分布图"""
        # 收集所有成功运行的最佳参数
        all_params = {}
        
        for optimizer, runs in self.results.items():
            successful_runs = [r for r in runs if not r.get('failed', False)]
            for run in successful_runs:
                if 'best_params' in run:
                    for param_name, param_value in run['best_params'].items():
                        if param_name not in all_params:
                            all_params[param_name] = {}
                        if optimizer not in all_params[param_name]:
                            all_params[param_name][optimizer] = []
                        all_params[param_name][optimizer].append(param_value)
        
        if not all_params:
            logger.warning("没有参数数据可用于分布图")
            return
        
        # 创建子图
        n_params = len(all_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (param_name, optimizer_data) in enumerate(all_params.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # 绘制每个优化器的参数分布
            for optimizer, values in optimizer_data.items():
                ax.hist(values, alpha=0.6, label=optimizer, bins=10)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('频次')
            ax.set_title(f'{param_name} 分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_box_plots(self) -> None:
        """绘制箱线图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
        performance_data = []
        time_data = []
        labels = []
        
        for optimizer, runs in self.results.items():
            successful_runs = [r for r in runs if not r.get('failed', False)]
            if successful_runs:
                values = [r['best_value'] for r in successful_runs]
                times = [r['execution_time'] for r in successful_runs]
                
                performance_data.append(values)
                time_data.append(times)
                labels.append(optimizer)
        
        # 性能箱线图
        bp1 = ax1.boxplot(performance_data, labels=labels, patch_artist=True)
        ax1.set_xlabel('优化器')
        ax1.set_ylabel('目标值')
        ax1.set_title('优化器性能分布（箱线图）')
        ax1.tick_params(axis='x', rotation=45)
        
        # 执行时间箱线图
        bp2 = ax2.boxplot(time_data, labels=labels, patch_artist=True)
        ax2.set_xlabel('优化器')
        ax2.set_ylabel('执行时间 (秒)')
        ax2.set_title('优化器执行时间分布（箱线图）')
        ax2.tick_params(axis='x', rotation=45)
        
        # 设置颜色
        colors = sns.color_palette("Set3", len(labels))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'box_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_statistical_analysis(self) -> None:
        """生成统计分析报告"""
        if not ANALYSIS_LIBS_AVAILABLE:
            return
        
        try:
            from scipy import stats
            
            # 准备数据进行统计检验
            performance_data = {}
            time_data = {}
            
            for optimizer, runs in self.results.items():
                successful_runs = [r for r in runs if not r.get('failed', False)]
                if successful_runs and len(successful_runs) > 1:
                    performance_data[optimizer] = [r['best_value'] for r in successful_runs]
                    time_data[optimizer] = [r['execution_time'] for r in successful_runs]
            
            if len(performance_data) < 2:
                logger.warning("统计分析需要至少2个优化器的多次运行数据")
                return
            
            # 生成统计报告
            stats_report = self.results_dir / 'statistical_analysis.txt'
            
            with open(stats_report, 'w', encoding='utf-8') as f:
                f.write("统计分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 描述性统计
                f.write("描述性统计:\n")
                f.write("-" * 30 + "\n")
                
                for optimizer, values in performance_data.items():
                    f.write(f"\n{optimizer} (性能):\n")
                    f.write(f"  样本数: {len(values)}\n")
                    f.write(f"  均值: {np.mean(values):.6f}\n")
                    f.write(f"  标准差: {np.std(values):.6f}\n")
                    f.write(f"  中位数: {np.median(values):.6f}\n")
                    f.write(f"  最小值: {np.min(values):.6f}\n")
                    f.write(f"  最大值: {np.max(values):.6f}\n")
                
                # 正态性检验
                f.write("\n\n正态性检验 (Shapiro-Wilk):\n")
                f.write("-" * 40 + "\n")
                
                for optimizer, values in performance_data.items():
                    if len(values) >= 3:
                        stat, p_value = stats.shapiro(values)
                        f.write(f"{optimizer}: 统计量={stat:.4f}, p值={p_value:.4f}")
                        f.write(f" ({'正态分布' if p_value > 0.05 else '非正态分布'})\n")
                
                # 两两比较（t检验或Mann-Whitney U检验）
                f.write("\n\n两两比较:\n")
                f.write("-" * 30 + "\n")
                
                optimizers_list = list(performance_data.keys())
                for i in range(len(optimizers_list)):
                    for j in range(i + 1, len(optimizers_list)):
                        opt1, opt2 = optimizers_list[i], optimizers_list[j]
                        values1, values2 = performance_data[opt1], performance_data[opt2]
                        
                        # 选择合适的检验方法
                        if len(values1) >= 3 and len(values2) >= 3:
                            # 检查正态性
                            _, p1 = stats.shapiro(values1)
                            _, p2 = stats.shapiro(values2)
                            
                            if p1 > 0.05 and p2 > 0.05:
                                # 使用t检验
                                stat, p_value = stats.ttest_ind(values1, values2)
                                test_name = "t检验"
                            else:
                                # 使用Mann-Whitney U检验
                                stat, p_value = stats.mannwhitneyu(values1, values2)
                                test_name = "Mann-Whitney U检验"
                            
                            f.write(f"\n{opt1} vs {opt2} ({test_name}):\n")
                            f.write(f"  统计量: {stat:.4f}\n")
                            f.write(f"  p值: {p_value:.4f}\n")
                            f.write(f"  结果: {'显著差异' if p_value < 0.05 else '无显著差异'}\n")
                
                # 方差分析（如果有多个组）
                if len(performance_data) > 2:
                    f.write("\n\n单因素方差分析 (ANOVA):\n")
                    f.write("-" * 35 + "\n")
                    
                    values_list = list(performance_data.values())
                    try:
                        f_stat, p_value = stats.f_oneway(*values_list)
                        f.write(f"F统计量: {f_stat:.4f}\n")
                        f.write(f"p值: {p_value:.4f}\n")
                        f.write(f"结果: {'组间存在显著差异' if p_value < 0.05 else '组间无显著差异'}\n")
                    except Exception as e:
                        f.write(f"ANOVA分析失败: {e}\n")
                
                # 效应量计算
                f.write("\n\n效应量分析:\n")
                f.write("-" * 25 + "\n")
                
                # Cohen's d（两两比较）
                for i in range(len(optimizers_list)):
                    for j in range(i + 1, len(optimizers_list)):
                        opt1, opt2 = optimizers_list[i], optimizers_list[j]
                        values1, values2 = performance_data[opt1], performance_data[opt2]
                        
                        if len(values1) >= 2 and len(values2) >= 2:
                            # 计算Cohen's d
                            pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                                (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                               (len(values1) + len(values2) - 2))
                            
                            if pooled_std > 0:
                                cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
                                
                                # 解释效应量大小
                                if abs(cohens_d) < 0.2:
                                    effect_size = "小"
                                elif abs(cohens_d) < 0.5:
                                    effect_size = "中等"
                                elif abs(cohens_d) < 0.8:
                                    effect_size = "大"
                                else:
                                    effect_size = "很大"
                                
                                f.write(f"{opt1} vs {opt2}: Cohen's d = {cohens_d:.4f} ({effect_size}效应)\n")
            
            logger.info(f"统计分析报告已保存: {stats_report}")
            
        except ImportError:
            logger.warning("scipy不可用，跳过统计分析")
        except Exception as e:
            logger.error(f"统计分析失败: {e}")
    
    def get_best_optimizer(self) -> Tuple[str, Dict[str, Any]]:
        """
        获取最佳优化器
        
        Returns:
            (最佳优化器名称, 详细信息)
        """
        if self.comparison_summary is None or self.comparison_summary.empty:
            raise ValueError("没有比较结果可用")
        
        # 根据平均最佳值选择最佳优化器
        best_row = self.comparison_summary.iloc[0]
        
        best_info = {
            'optimizer': best_row['optimizer'],
            'mean_best_value': best_row['mean_best_value'],
            'std_best_value': best_row['std_best_value'],
            'mean_execution_time': best_row['mean_execution_time'],
            'successful_runs': best_row['successful_runs'],
            'efficiency_score': best_row['efficiency_score']
        }
        
        return best_row['optimizer'], best_info
    
    def export_results(self, export_format: str = 'excel') -> str:
        """
        导出结果到指定格式
        
        Args:
            export_format: 导出格式 ('excel', 'csv', 'json')
            
        Returns:
            导出文件路径
        """
        if not ANALYSIS_LIBS_AVAILABLE and export_format == 'excel':
            logger.warning("pandas不可用，改用CSV格式导出")
            export_format = 'csv'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format.lower() == 'excel':
            export_file = self.results_dir / f'comparison_results_{timestamp}.xlsx'
            
            with pd.ExcelWriter(export_file) as writer:
                # 摘要表
                if self.comparison_summary is not None:
                    self.comparison_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # 详细结果表
                all_results = []
                for optimizer, runs in self.results.items():
                    for run in runs:
                        if not run.get('failed', False):
                            result_row = {
                                'optimizer': optimizer,
                                'run_index': run['run_index'],
                                'best_value': run['best_value'],
                                'execution_time': run['execution_time']
                            }
                            # 添加最佳参数
                            if 'best_params' in run:
                                result_row.update(run['best_params'])
                            all_results.append(result_row)
                
                if all_results:
                    detailed_df = pd.DataFrame(all_results)
                    detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        elif export_format.lower() == 'csv':
            export_file = self.results_dir / f'comparison_results_{timestamp}.csv'
            
            if self.comparison_summary is not None:
                self.comparison_summary.to_csv(export_file, index=False)
        
        elif export_format.lower() == 'json':
            export_file = self.results_dir / f'comparison_results_{timestamp}.json'
            
            export_data = {
                'summary': self.comparison_summary.to_dict() if self.comparison_summary is not None else None,
                'detailed_results': self.results,
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'n_calls': self.n_calls,
                    'n_runs': self.n_runs,
                    'evaluator_type': self.evaluator_type
                }
            }
            
            import json
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
        
        logger.info(f"结果已导出: {export_file}")
        return str(export_file)

def compare_optimizers(
    optimizers: List[str] = ['bayesian', 'random', 'forest', 'genetic'],
    n_calls: int = 30,
    n_runs: int = 3,
    evaluator_type: str = 'mock',
    run_sensitivity_analysis: bool = True,
    generate_report: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷的优化器比较函数
    
    Args:
        optimizers: 要比较的优化器列表
        n_calls: 每个优化器的迭代次数
        n_runs: 每个优化器的运行次数
        evaluator_type: 评估器类型
        run_sensitivity_analysis: 是否运行敏感性分析
        generate_report: 是否生成详细报告
        **kwargs: 其他参数
        
    Returns:
        比较结果字典
    """
    logger.info("开始优化器性能比较")
    
    # 创建比较器
    comparison = OptimizationComparison(
        optimizers=optimizers,
        n_calls=n_calls,
        evaluator_type=evaluator_type,
        n_runs=n_runs,
        use_cache=False  # 比较时不使用缓存确保公平性
    )
    
    # 运行比较
    results = comparison.run_comparison(
        run_sensitivity_analysis=run_sensitivity_analysis,
        **kwargs
    )
    
    # 获取最佳优化器
    try:
        best_optimizer, best_info = comparison.get_best_optimizer()
        results['best_optimizer'] = best_optimizer
        results['best_optimizer_info'] = best_info
        
        logger.info(f"最佳优化器: {best_optimizer}")
        logger.info(f"平均最佳值: {best_info['mean_best_value']:.6f}")
        
    except Exception as e:
        logger.warning(f"无法确定最佳优化器: {e}")
    
    # 导出结果
    if generate_report:
        try:
            export_file = comparison.export_results('excel')
            results['export_file'] = export_file
        except Exception as e:
            logger.warning(f"导出结果失败: {e}")
    
    return results

# 向后兼容函数
def compare_optimizers_legacy(*args, **kwargs):
    """向后兼容的比较函数"""
    warnings.warn("compare_optimizers_legacy已弃用，请使用compare_optimizers", 
                  DeprecationWarning, stacklevel=2)
    return compare_optimizers(*args, **kwargs)

if __name__ == "__main__":
    # 示例用法
    logger.info("优化器比较工具示例")
    
    # 快速比较（使用模拟评估器）
    results = compare_optimizers(
        optimizers=['random', 'genetic'],
        n_calls=15,
        n_runs=2,
        evaluator_type='mock'
    )
    
    print("\n比较结果摘要:")
    if 'summary' in results and results['summary']:
        summary_df = pd.DataFrame(results['summary'])
        print(summary_df.to_string(index=False))
    
    if 'best_optimizer' in results:
        print(f"\n推荐的最佳优化器: {results['best_optimizer']}")
    
    print(f"\n详细结果保存在: {results['results_dir']}")