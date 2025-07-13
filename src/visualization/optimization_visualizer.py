#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化过程可视化模块

从ansa_mesh_optimizer_refactored.py中提取的可视化功能
"""

import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 安全导入字体配置模块
try:
    from utils.font_decorator import with_chinese_font
    DECORATOR_AVAILABLE = True
except ImportError:
    logger.warning("字体装饰器模块未找到")
    DECORATOR_AVAILABLE = False
    
    def with_chinese_font(func):
        return func

# 安全导入可视化库
VISUALIZATION_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    from utils.display_config import configure_matplotlib_for_display, safe_show, safe_close
    configure_matplotlib_for_display()
    VISUALIZATION_AVAILABLE = True
    logger.info("可视化库加载成功")
except ImportError as e:
    logger.warning(f"可视化库未安装: {e}")
    
    # 创建安全显示函数的备用版本
    def safe_show():
        pass
    
    def safe_close():
        pass


class OptimizationVisualizer:
    """优化过程可视化器"""
    
    def __init__(self, report_dir: Optional[Path] = None):
        """
        初始化可视化器
        
        Args:
            report_dir: 报告保存目录
        """
        if report_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = Path(f"optimization_reports/{timestamp}_visualization")
        
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_optimization_plots(self, 
                                  result: Dict[str, Any],
                                  optimization_history: List[Dict[str, Any]],
                                  early_stopping=None) -> None:
        """
        生成优化相关的所有图表
        
        Args:
            result: 优化结果
            optimization_history: 优化历史
            early_stopping: 早停对象
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化库不可用，跳过图表生成")
            return
        
        try:
            # 收敛图
            if 'skopt_result' in result:
                self._plot_convergence(result)
                
                # 参数重要性图（如果数据足够）
                if result.get('n_calls', 0) >= 20:
                    self._plot_parameter_importance(result)
            
            # 优化历史图
            if optimization_history:
                self._plot_optimization_history(optimization_history)
            
            # 早停历史图
            if early_stopping and hasattr(early_stopping, 'plot_history'):
                try:
                    early_stopping.plot_history(str(self.report_dir / "early_stopping_history.png"))
                except Exception as e:
                    logger.warning(f"无法生成早停历史图: {e}")
            
            logger.info(f"优化图表已保存到: {self.report_dir}")
            
        except Exception as e:
            logger.warning(f"生成优化图表失败: {e}")
    
    def _plot_convergence(self, result: Dict[str, Any]) -> None:
        """绘制收敛图"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # 尝试导入scikit-optimize的绘图函数
            try:
                from skopt.plots import plot_convergence
                
                plt.figure(figsize=(10, 6))
                # Handle both dictionary and object result formats
                skopt_result = result.get('skopt_result') if isinstance(result, dict) else getattr(result, 'skopt_result', None)
                optimizer_name = result.get('optimizer_name', 'Unknown') if isinstance(result, dict) else getattr(result, 'optimizer_name', 'Unknown')
                if skopt_result:
                    plot_convergence(skopt_result)
                    plt.title(f"Convergence - {optimizer_name}")
                plt.savefig(self.report_dir / "convergence.png", dpi=300, bbox_inches='tight')
                safe_close()
                
            except ImportError:
                logger.warning("scikit-optimize不可用，跳过收敛图生成")
                
        except Exception as e:
            logger.warning(f"生成收敛图失败: {e}")
    
    def _plot_parameter_importance(self, result: Dict[str, Any]) -> None:
        """绘制参数重要性图"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # 尝试导入scikit-optimize的绘图函数
            try:
                from skopt.plots import plot_objective
                
                plt.figure(figsize=(12, 8))
                # Handle both dictionary and object result formats
                skopt_result = result.get('skopt_result') if isinstance(result, dict) else getattr(result, 'skopt_result', None)
                if skopt_result:
                    plot_objective(skopt_result)
                plt.savefig(self.report_dir / "parameter_importance.png", dpi=300, bbox_inches='tight')
                safe_close()
                
            except ImportError:
                logger.warning("scikit-optimize不可用，跳过参数重要性图生成")
            except Exception as e:
                logger.warning(f"无法生成参数重要性图: {e}")
                
        except Exception as e:
            logger.warning(f"生成参数重要性图失败: {e}")
    
    @with_chinese_font
    def _plot_optimization_history(self, optimization_history: List[Dict[str, Any]]) -> None:
        """绘制优化历史"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            results = [entry['result'] for entry in optimization_history]
            iterations = list(range(1, len(results) + 1))
            
            plt.figure(figsize=(12, 6))
            
            # 子图1: 目标值变化
            plt.subplot(1, 2, 1)
            plt.plot(iterations, results, 'b-', alpha=0.7, label='Objective Value')
            
            # 计算最佳值序列
            best_so_far = []
            current_best = float('inf')
            for result in results:
                if result < current_best:
                    current_best = result
                best_so_far.append(current_best)
            
            plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best So Far')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('Optimization Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图2: 改进分布
            plt.subplot(1, 2, 2)
            improvements = []
            for i in range(1, len(best_so_far)):
                if best_so_far[i] < best_so_far[i-1]:
                    improvements.append(i)
            
            if improvements:
                plt.scatter(improvements, [best_so_far[i] for i in improvements], 
                           c='red', s=50, alpha=0.7, label='Improvements')
                plt.plot(iterations, best_so_far, 'b-', alpha=0.5, label='Best Value')
                plt.xlabel('Iteration')
                plt.ylabel('Best Value')
                plt.title('Improvement Points')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.report_dir / "optimization_history.png", dpi=300, bbox_inches='tight')
            safe_close()
            
        except Exception as e:
            logger.warning(f"绘制优化历史失败: {e}")
    
    @with_chinese_font
    def plot_sensitivity_analysis(self,
                                sensitivity_results: Dict[str, List],
                                best_params: Dict[str, float],
                                save_path: Optional[str] = None) -> None:
        """
        绘制敏感性分析图表
        
        Args:
            sensitivity_results: 敏感性分析结果
            best_params: 最佳参数
            save_path: 保存路径（可选）
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化库不可用，跳过敏感性分析图表生成")
            return
        
        try:
            n_params = len(sensitivity_results)
            if n_params == 0:
                return
                
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            # 正确处理axes的类型
            if n_params == 1:
                axes_list = [axes]
            elif n_rows == 1:
                axes_list = list(axes) if n_cols > 1 else [axes]
            elif n_cols == 1:
                axes_list = list(axes) if n_rows > 1 else [axes]
            else:
                axes_list = axes.flatten()
            
            for i, (param_name, results) in enumerate(sensitivity_results.items()):
                ax = axes_list[i]
                
                test_values, objectives = zip(*results)
                ax.plot(test_values, objectives, 'o-', linewidth=2, markersize=6)
                ax.axvline(x=best_params[param_name], color='r', linestyle='--', 
                          linewidth=2, label='Best Value')
                ax.set_title(f'敏感性: {param_name}')
                ax.set_xlabel(param_name)
                ax.set_ylabel('目标值')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # 隐藏多余的子图
            for i in range(n_params, len(axes_list)):
                axes_list[i].set_visible(False)
            
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                filename = Path(save_path)
            else:
                filename = self.report_dir / "sensitivity_analysis.png"
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            safe_close()
            
            logger.info(f"敏感性分析图表已保存: {filename}")
            
        except Exception as e:
            logger.warning(f"生成敏感性分析图表失败: {e}")
    
    def plot_parameter_evolution(self, 
                               optimization_history: List[Dict[str, Any]],
                               param_names: List[str]) -> None:
        """
        绘制参数演化图
        
        Args:
            optimization_history: 优化历史
            param_names: 参数名称列表
        """
        if not VISUALIZATION_AVAILABLE or not optimization_history:
            return
        
        try:
            n_params = len(param_names)
            if n_params == 0:
                return
            
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            # 正确处理axes的类型
            if n_params == 1:
                axes_list = [axes]
            elif n_rows == 1:
                axes_list = list(axes) if n_cols > 1 else [axes]
            elif n_cols == 1:
                axes_list = list(axes) if n_rows > 1 else [axes]
            else:
                axes_list = axes.flatten()
            
            iterations = list(range(1, len(optimization_history) + 1))
            
            for i, param_name in enumerate(param_names):
                ax = axes_list[i]
                
                # 提取参数值
                param_values = []
                for entry in optimization_history:
                    params = entry.get('params', {})
                    param_values.append(params.get(param_name, 0))
                
                ax.plot(iterations, param_values, 'o-', alpha=0.7)
                ax.set_title(f'参数演化: {param_name}')
                ax.set_xlabel('迭代次数')
                ax.set_ylabel(param_name)
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(n_params, len(axes_list)):
                axes_list[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.report_dir / "parameter_evolution.png", dpi=300, bbox_inches='tight')
            safe_close()
            
            logger.info(f"参数演化图已保存: {self.report_dir / 'parameter_evolution.png'}")
            
        except Exception as e:
            logger.warning(f"生成参数演化图失败: {e}")