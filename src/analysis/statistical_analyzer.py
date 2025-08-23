#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计分析模块

从compare_optimizers_improved.py中提取的统计分析功能
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入统计库
SCIPY_AVAILABLE = False
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
    logger.info("scipy统计库加载成功")
except ImportError:
    logger.warning("scipy不可用，将跳过高级统计分析")


class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, results_dir: Path):
        """
        初始化统计分析器
        
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_statistical_analysis(self, 
                                    results: Dict[str, List[Dict[str, Any]]],
                                    optimizers: List[str]) -> None:
        """
        生成统计分析报告
        
        Args:
            results: 优化结果字典
            optimizers: 优化器列表
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy不可用，跳过统计分析")
            return
        
        try:
            # 准备数据进行统计检验
            performance_data = {}
            time_data = {}
            
            for optimizer in optimizers:
                runs = results.get(optimizer, [])
                successful_runs = [r for r in runs if not r.get('failed', False)]
                
                if len(successful_runs) > 1:
                    performance_data[optimizer] = [r['best_value'] for r in successful_runs]
                    time_data[optimizer] = [r.get('execution_time', 0) for r in successful_runs]
            
            if len(performance_data) < 2:
                logger.warning("统计分析需要至少2个优化器的多次运行数据")
                return
            
            # 生成统计报告
            self._write_statistical_report(performance_data, time_data)
            
        except Exception as e:
            logger.error(f"统计分析失败: {e}")
    
    def _write_statistical_report(self, 
                                performance_data: Dict[str, List[float]], 
                                time_data: Dict[str, List[float]]) -> None:
        """写入统计分析报告"""
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
                    try:
                        stat, p_value = stats.shapiro(values)
                        f.write(f"{optimizer}: 统计量={stat:.4f}, p值={p_value:.4f}")
                        f.write(f" ({'正态分布' if p_value > 0.05 else '非正态分布'})\n")
                    except Exception as e:
                        f.write(f"{optimizer}: 检验失败 ({e})\n")
            
            # 两两比较
            f.write("\n\n两两比较:\n")
            f.write("-" * 30 + "\n")
            
            optimizers_list = list(performance_data.keys())
            for i in range(len(optimizers_list)):
                for j in range(i + 1, len(optimizers_list)):
                    opt1, opt2 = optimizers_list[i], optimizers_list[j]
                    values1, values2 = performance_data[opt1], performance_data[opt2]
                    
                    if len(values1) >= 3 and len(values2) >= 3:
                        try:
                            # 使用Mann-Whitney U检验（非参数）
                            stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                            
                            f.write(f"\n{opt1} vs {opt2} (Mann-Whitney U检验):\n")
                            f.write(f"  统计量: {stat:.4f}\n")
                            f.write(f"  p值: {p_value:.4f}\n")
                            f.write(f"  结果: {'显著差异' if p_value < 0.05 else '无显著差异'}\n")
                            
                        except Exception as e:
                            f.write(f"\n{opt1} vs {opt2}: 检验失败 ({e})\n")
            
            # 方差分析（如果有多个组）
            if len(performance_data) > 2:
                f.write("\n\nKruskal-Wallis检验 (非参数方差分析):\n")
                f.write("-" * 45 + "\n")
                
                try:
                    values_list = list(performance_data.values())
                    h_stat, p_value = stats.kruskal(*values_list)
                    f.write(f"H统计量: {h_stat:.4f}\n")
                    f.write(f"p值: {p_value:.4f}\n")
                    f.write(f"结果: {'组间存在显著差异' if p_value < 0.05 else '组间无显著差异'}\n")
                except Exception as e:
                    f.write(f"Kruskal-Wallis检验失败: {e}\n")
        
        logger.info(f"统计分析报告已保存: {stats_report}")
    
    def calculate_effect_sizes(self, 
                             performance_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        计算效应量
        
        Args:
            performance_data: 性能数据
            
        Returns:
            效应量结果
        """
        effect_sizes = {}
        optimizers_list = list(performance_data.keys())
        
        for i in range(len(optimizers_list)):
            for j in range(i + 1, len(optimizers_list)):
                opt1, opt2 = optimizers_list[i], optimizers_list[j]
                values1, values2 = performance_data[opt1], performance_data[opt2]
                
                if len(values1) >= 3 and len(values2) >= 3:
                    # Cohen's d
                    mean1, mean2 = np.mean(values1), np.mean(values2)
                    std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
                    pooled_std = np.sqrt(((len(values1) - 1) * std1**2 + 
                                        (len(values2) - 1) * std2**2) / 
                                       (len(values1) + len(values2) - 2))
                    
                    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    effect_sizes[f"{opt1}_vs_{opt2}"] = {
                        'cohens_d': cohens_d,
                        'interpretation': self._interpret_cohens_d(abs(cohens_d))
                    }
        
        return effect_sizes
    
    def _interpret_cohens_d(self, d: float) -> str:
        """解释Cohen's d效应量"""
        if d < 0.2:
            return "小效应"
        elif d < 0.5:
            return "中等效应"
        elif d < 0.8:
            return "大效应"
        else:
            return "非常大效应"
    
    def generate_confidence_intervals(self, 
                                    performance_data: Dict[str, List[float]],
                                    confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        生成置信区间
        
        Args:
            performance_data: 性能数据
            confidence_level: 置信水平
            
        Returns:
            置信区间结果
        """
        confidence_intervals = {}
        alpha = 1 - confidence_level
        
        for optimizer, values in performance_data.items():
            if len(values) >= 3:
                mean = np.mean(values)
                sem = stats.sem(values)  # 标准误差
                
                if SCIPY_AVAILABLE:
                    # 使用t分布
                    t_critical = stats.t.ppf(1 - alpha/2, len(values) - 1)
                    margin_error = t_critical * sem
                else:
                    # 使用正态分布近似
                    z_critical = 1.96  # 95%置信区间
                    margin_error = z_critical * sem
                
                confidence_intervals[optimizer] = {
                    'mean': mean,
                    'lower_bound': mean - margin_error,
                    'upper_bound': mean + margin_error,
                    'margin_error': margin_error
                }
        
        return confidence_intervals
    
    def perform_power_analysis(self, 
                             performance_data: Dict[str, List[float]],
                             alpha: float = 0.05) -> Dict[str, float]:
        """
        执行功效分析
        
        Args:
            performance_data: 性能数据
            alpha: 显著性水平
            
        Returns:
            功效分析结果
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy不可用，跳过功效分析")
            return {}
        
        power_results = {}
        optimizers_list = list(performance_data.keys())
        
        for i in range(len(optimizers_list)):
            for j in range(i + 1, len(optimizers_list)):
                opt1, opt2 = optimizers_list[i], optimizers_list[j]
                values1, values2 = performance_data[opt1], performance_data[opt2]
                
                if len(values1) >= 3 and len(values2) >= 3:
                    try:
                        # 计算效应量
                        mean1, mean2 = np.mean(values1), np.mean(values2)
                        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
                        pooled_std = np.sqrt(((len(values1) - 1) * std1**2 + 
                                            (len(values2) - 1) * std2**2) / 
                                           (len(values1) + len(values2) - 2))
                        
                        effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        
                        # 简化的功效计算（基于t检验）
                        n1, n2 = len(values1), len(values2)
                        df = n1 + n2 - 2
                        
                        # 非中心参数
                        ncp = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
                        
                        # 临界值
                        t_critical = stats.t.ppf(1 - alpha/2, df)
                        
                        # 功效（简化计算）
                        power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
                        
                        power_results[f"{opt1}_vs_{opt2}"] = power
                        
                    except Exception as e:
                        logger.warning(f"功效分析失败 {opt1} vs {opt2}: {e}")
        
        return power_results
