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
                                  result,
                                  optimization_history=None,
                                  early_stopping=None) -> None:
        """
        生成优化相关的所有图表 - 兼容多种数据格式
        
        Args:
            result: 优化结果，可以是字典或OptimizationResult对象
            optimization_history: 优化历史（可选）
            early_stopping: 早停对象（可选）
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化库不可用，跳过图表生成")
            return
        
        try:
            # 确保matplotlib配置正确
            from utils.display_config import configure_matplotlib_for_display
            configure_matplotlib_for_display()
            
            plots_generated = 0
            
            # 处理不同的数据格式
            if isinstance(result, dict):
                # 字典格式的结果
                # 收敛图
                if 'skopt_result' in result:
                    try:
                        self._plot_convergence(result)
                        plots_generated += 1
                    except Exception as e:
                        logger.warning(f"收敛图生成失败: {e}")
                    
                    # 参数重要性图（如果数据足够）
                    if result.get('n_calls', 0) >= 20:
                        try:
                            self._plot_parameter_importance(result)
                            plots_generated += 1
                        except Exception as e:
                            logger.warning(f"参数重要性图生成失败: {e}")
                
                # 优化历史图
                if optimization_history:
                    try:
                        self._plot_optimization_history(optimization_history)
                        plots_generated += 1
                    except Exception as e:
                        logger.warning(f"优化历史图生成失败: {e}")
            
            else:
                # OptimizationResult对象或其他格式
                # 使用新的可视化方法
                try:
                    self.plot_optimization_history(result)
                    plots_generated += 1
                except Exception as e:
                    logger.warning(f"优化历史图生成失败: {e}")
                
                try:
                    self.plot_parameter_evolution(result)
                    plots_generated += 1
                except Exception as e:
                    logger.warning(f"参数演化图生成失败: {e}")
                
                try:
                    self.plot_parameter_distribution(result)
                    plots_generated += 1
                except Exception as e:
                    logger.warning(f"参数分布图生成失败: {e}")
            
            # 早停历史图
            if early_stopping and hasattr(early_stopping, 'plot_history'):
                try:
                    early_stopping.plot_history(str(self.report_dir / "early_stopping_history.png"))
                    plots_generated += 1
                except Exception as e:
                    logger.warning(f"无法生成早停历史图: {e}")
            
            if plots_generated > 0:
                logger.info(f"已生成 {plots_generated} 个优化图表，保存到: {self.report_dir}")
            else:
                logger.warning("没有生成任何图表")
            
        except Exception as e:
            logger.warning(f"生成优化图表失败: {e}")
    
    def _plot_convergence(self, result: Dict[str, Any]) -> None:
        """绘制收敛图"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # Handle both dictionary and object result formats
            skopt_result = result.get('skopt_result') if isinstance(result, dict) else getattr(result, 'skopt_result', None)
            optimizer_name = result.get('optimizer_name', 'Unknown') if isinstance(result, dict) else getattr(result, 'optimizer_name', 'Unknown')
            
            if skopt_result is None:
                logger.debug("没有skopt_result，跳过收敛图生成")
                return
            
            # 尝试导入scikit-optimize的绘图函数
            try:
                from skopt.plots import plot_convergence
                
                plt.figure(figsize=(10, 6))
                
                # 验证skopt_result是否有效且来自真实的scikit-optimize
                if (hasattr(skopt_result, 'func_vals') and
                    hasattr(skopt_result, 'x_iters') and
                    hasattr(skopt_result, '__class__') and
                    'skopt' in str(skopt_result.__class__.__module__)):
                    
                    plot_convergence(skopt_result)
                    plt.title(f"Convergence - {optimizer_name}")
                    plt.savefig(self.report_dir / "convergence.png", dpi=300, bbox_inches='tight')
                    logger.info(f"收敛图已保存: {self.report_dir / 'convergence.png'}")
                else:
                    # 创建自定义收敛图
                    self._plot_custom_convergence(result)
                    
                safe_close()
                
            except ImportError:
                logger.warning("scikit-optimize不可用，使用自定义收敛图")
                self._plot_custom_convergence(result)
                
        except Exception as e:
            logger.warning(f"生成收敛图失败: {e}")
            # 尝试生成自定义收敛图作为备选
            try:
                self._plot_custom_convergence(result)
            except Exception as e2:
                logger.warning(f"生成自定义收敛图也失败: {e2}")
    
    def _plot_custom_convergence(self, result: Dict[str, Any]) -> None:
        """绘制自定义收敛图（当scikit-optimize不可用时）"""
        try:
            optimizer_name = result.get('optimizer_name', 'Unknown') if isinstance(result, dict) else getattr(result, 'optimizer_name', 'Unknown')
            
            # 尝试从skopt_result获取数据
            skopt_result = result.get('skopt_result') if isinstance(result, dict) else getattr(result, 'skopt_result', None)
            
            if skopt_result and hasattr(skopt_result, 'func_vals'):
                func_vals = skopt_result.func_vals
            else:
                # 如果没有skopt_result，尝试从其他地方获取数据
                logger.warning("无法获取收敛数据，跳过收敛图生成")
                return
            
            plt.figure(figsize=(10, 6))
            
            # 计算最佳值序列
            best_so_far = []
            current_best = float('inf')
            for val in func_vals:
                if val < current_best:
                    current_best = val
                best_so_far.append(current_best)
            
            iterations = list(range(1, len(func_vals) + 1))
            
            # 绘制收敛曲线
            plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best Value So Far')
            plt.scatter(iterations, func_vals, alpha=0.6, s=30, label='Function Evaluations')
            
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title(f"Convergence - {optimizer_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(self.report_dir / "convergence.png", dpi=300, bbox_inches='tight')
            logger.info(f"自定义收敛图已保存: {self.report_dir / 'convergence.png'}")
            
        except Exception as e:
            logger.warning(f"生成自定义收敛图失败: {e}")
    
    def _plot_parameter_importance(self, result: Dict[str, Any]) -> None:
        """绘制参数重要性图"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # Handle both dictionary and object result formats
            skopt_result = result.get('skopt_result') if isinstance(result, dict) else getattr(result, 'skopt_result', None)
            
            if skopt_result is None:
                logger.debug("没有skopt_result，跳过参数重要性图生成")
                return
            
            # 尝试导入scikit-optimize的绘图函数
            try:
                from skopt.plots import plot_objective
                
                # 验证skopt_result是否有效且有space信息
                if (hasattr(skopt_result, 'space') and
                    skopt_result.space is not None and
                    hasattr(skopt_result.space, 'n_dims') and
                    skopt_result.space.n_dims > 0 and
                    hasattr(skopt_result, '__class__') and
                    'skopt' in str(skopt_result.__class__.__module__)):
                    
                    plt.figure(figsize=(12, 8))
                    plot_objective(skopt_result)
                    plt.savefig(self.report_dir / "parameter_importance.png", dpi=300, bbox_inches='tight')
                    logger.info(f"参数重要性图已保存: {self.report_dir / 'parameter_importance.png'}")
                    safe_close()
                else:
                    logger.debug("skopt_result缺少space信息，跳过参数重要性图生成")
                
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
    
    def plot_parameter_evolution_legacy(self,
                               optimization_history: List[Dict[str, Any]],
                               param_names: List[str]) -> None:
        """
        绘制参数演化图 - 传统版本
        
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
    
    @with_chinese_font
    def plot_optimization_history(self, data, save_path=None):
        """
        绘制优化历史曲线 - 兼容遗传算法数据格式
        
        Args:
            data: 优化数据，可以是OptimizationResult对象或包含history的字典
            save_path: 保存路径
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化库不可用，跳过优化历史图表生成")
            return
        
        try:
            # 处理不同的数据格式
            if hasattr(data, 'generation_stats') and data.generation_stats:
                # OptimizationResult对象，使用generation_stats
                history = data.generation_stats
                scores = [stat.get('best_fitness', 0) for stat in history]
                generations = [stat.get('generation', i) for i, stat in enumerate(history)]
            elif hasattr(data, 'history') and data.history:
                # OptimizationResult对象，使用history
                history = data.history
                scores = [score for _, score in history]
                generations = list(range(1, len(scores) + 1))
            elif isinstance(data, dict) and 'history' in data:
                # 字典格式
                history = data['history']
                scores = []
                generations = []
                for i, record in enumerate(history):
                    if isinstance(record, dict):
                        score = record.get('best_score') or record.get('score') or record.get('result')
                        if score is not None:
                            scores.append(score)
                            generations.append(i + 1)
            else:
                logger.error("无法识别的数据格式")
                return
            
            if not scores:
                logger.warning("没有有效的得分数据")
                return
            
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 绘制得分曲线
            plt.plot(generations, scores, 'b-', linewidth=2, label='优化得分', marker='o', markersize=4)
            
            # 添加最佳得分线
            best_score = min(scores) if scores else 0
            plt.axhline(y=best_score, color='r', linestyle='--', alpha=0.7,
                       label=f'最佳得分: {best_score:.4f}')
            
            plt.xlabel('迭代次数')
            plt.ylabel('得分')
            plt.title('优化历史曲线')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 设置坐标轴范围
            if len(scores) > 1:
                plt.xlim(0, max(generations) + 1)
                score_range = max(scores) - min(scores)
                if score_range > 0:
                    plt.ylim(min(scores) - score_range * 0.1, max(scores) + score_range * 0.1)
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"优化历史图表已保存至: {save_path}")
            else:
                plt.savefig(self.report_dir / "optimization_history.png", dpi=300, bbox_inches='tight')
                logger.info(f"优化历史图表已保存至: {self.report_dir / 'optimization_history.png'}")
            
            safe_close()
            
        except Exception as e:
            logger.error(f"绘制优化历史失败: {e}")
            # 创建错误图表
            try:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, f'图表生成失败\n错误: {str(e)}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=14, color='red')
                plt.title('优化历史曲线 (错误)')
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                safe_close()
            except:
                pass
    
    @with_chinese_font
    def plot_parameter_evolution(self, data, save_path=None):
        """
        绘制参数演化图 - 兼容遗传算法数据格式
        
        Args:
            data: 优化数据
            save_path: 保存路径
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化库不可用，跳过参数演化图表生成")
            return
        
        try:
            # 提取参数数据
            params_data = {}
            param_names = []
            
            if hasattr(data, 'generation_stats') and data.generation_stats:
                # 从generation_stats提取参数（如果有的话）
                # 这种情况下我们需要从best_individual获取参数
                if hasattr(data, 'parameter_names'):
                    param_names = data.parameter_names
                    # 模拟参数演化（因为generation_stats可能不包含详细参数）
                    for name in param_names:
                        params_data[name] = []
                    
                    # 如果有最佳参数，至少显示最终值
                    if hasattr(data, 'best_params') and isinstance(data.best_params, dict):
                        for name, value in data.best_params.items():
                            if name in params_data:
                                # 为每一代创建一个值（简化处理）
                                params_data[name] = [value] * len(data.generation_stats)
            
            elif hasattr(data, 'history') and data.history:
                # 从history提取参数
                if hasattr(data, 'parameter_names'):
                    param_names = data.parameter_names
                    for name in param_names:
                        params_data[name] = []
                    
                    for params_list, _ in data.history:
                        for i, value in enumerate(params_list):
                            if i < len(param_names):
                                params_data[param_names[i]].append(value)
            
            elif isinstance(data, dict) and 'history' in data:
                # 字典格式
                history = data['history']
                for record in history:
                    if isinstance(record, dict) and 'params' in record:
                        params = record['params']
                        if isinstance(params, dict):
                            for param, value in params.items():
                                if param not in params_data:
                                    params_data[param] = []
                                params_data[param].append(value)
            
            if not params_data:
                logger.warning("没有参数演化数据")
                return
            
            # 过滤掉空的参数
            params_data = {k: v for k, v in params_data.items() if v}
            
            if not params_data:
                logger.warning("所有参数数据都为空")
                return
            
            # 创建子图
            n_params = len(params_data)
            fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
            if n_params == 1:
                axes = [axes]
            
            for i, (param, values) in enumerate(params_data.items()):
                generations = list(range(1, len(values) + 1))
                
                axes[i].plot(generations, values, 'g-', linewidth=2, marker='o', markersize=4)
                axes[i].set_title(f'参数演化: {param}')
                axes[i].set_xlabel('迭代次数')
                axes[i].set_ylabel('参数值')
                axes[i].grid(True, alpha=0.3)
                
                # 设置坐标轴范围
                if len(values) > 1:
                    axes[i].set_xlim(0, max(generations) + 1)
                    value_range = max(values) - min(values)
                    if value_range > 0:
                        axes[i].set_ylim(min(values) - value_range * 0.1,
                                       max(values) + value_range * 0.1)
            
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"参数演化图表已保存至: {save_path}")
            else:
                plt.savefig(self.report_dir / "parameter_evolution.png", dpi=300, bbox_inches='tight')
                logger.info(f"参数演化图表已保存至: {self.report_dir / 'parameter_evolution.png'}")
            
            safe_close()
            
        except Exception as e:
            logger.error(f"绘制参数演化失败: {e}")
            # 创建错误图表
            try:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, f'参数演化图生成失败\n错误: {str(e)}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=14, color='red')
                plt.title('参数演化图 (错误)')
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                safe_close()
            except:
                pass
    
    @with_chinese_font
    def plot_parameter_distribution(self, data, save_path=None):
        """
        绘制参数分布图
        
        Args:
            data: 优化数据
            save_path: 保存路径
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("可视化库不可用，跳过参数分布图表生成")
            return
        
        try:
            # 收集所有参数值
            all_params = {}
            
            if hasattr(data, 'history') and data.history:
                # 从history提取参数
                if hasattr(data, 'parameter_names'):
                    param_names = data.parameter_names
                    for name in param_names:
                        all_params[name] = []
                    
                    for params_list, _ in data.history:
                        for i, value in enumerate(params_list):
                            if i < len(param_names):
                                all_params[param_names[i]].append(value)
            
            elif isinstance(data, dict) and 'history' in data:
                # 字典格式
                history = data['history']
                for record in history:
                    if isinstance(record, dict) and 'params' in record:
                        params = record['params']
                        if isinstance(params, dict):
                            for param, value in params.items():
                                if isinstance(value, (int, float)):
                                    if param not in all_params:
                                        all_params[param] = []
                                    all_params[param].append(value)
            
            if not all_params:
                logger.warning("没有有效的参数数据用于分布图")
                return
            
            # 创建子图
            n_params = len(all_params)
            fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
            if n_params == 1:
                axes = [axes]
            
            for i, (param, values) in enumerate(all_params.items()):
                if values:
                    import numpy as np
                    axes[i].hist(values, bins=min(20, len(set(values))), alpha=0.7,
                               color='skyblue', edgecolor='black')
                    axes[i].set_title(f'参数分布: {param}')
                    axes[i].set_xlabel('参数值')
                    axes[i].set_ylabel('频率')
                    axes[i].grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7,
                                  label=f'均值: {mean_val:.4f}')
                    axes[i].legend()
                    
                    # 在图上显示统计信息
                    axes[i].text(0.02, 0.98, f'均值: {mean_val:.4f}\n标准差: {std_val:.4f}\n样本数: {len(values)}',
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"参数分布图已保存至: {save_path}")
            else:
                plt.savefig(self.report_dir / "parameter_distribution.png", dpi=300, bbox_inches='tight')
                logger.info(f"参数分布图已保存至: {self.report_dir / 'parameter_distribution.png'}")
            
            safe_close()
            
        except Exception as e:
            logger.error(f"绘制参数分布图失败: {e}")
            # 创建错误图表
            try:
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, f'参数分布图生成失败\n错误: {str(e)}',
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=14, color='red')
                plt.title('参数分布图 (错误)')
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                safe_close()
            except:
                pass
    
    def generate_complete_visualization_report(self, data, output_dir=None):
        """
        生成完整的可视化报告 - 兼容多种数据格式
        
        Args:
            data: 优化数据
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = self.report_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            plots_generated = 0
            
            # 生成优化历史图
            try:
                history_path = output_dir / "optimization_history.png"
                self.plot_optimization_history(data, save_path=str(history_path))
                if history_path.exists():
                    plots_generated += 1
            except Exception as e:
                logger.warning(f"生成优化历史图失败: {e}")
            
            # 生成参数演化图
            try:
                evolution_path = output_dir / "parameter_evolution.png"
                self.plot_parameter_evolution(data, save_path=str(evolution_path))
                if evolution_path.exists():
                    plots_generated += 1
            except Exception as e:
                logger.warning(f"生成参数演化图失败: {e}")
            
            # 生成参数分布图
            try:
                distribution_path = output_dir / "parameter_distribution.png"
                self.plot_parameter_distribution(data, save_path=str(distribution_path))
                if distribution_path.exists():
                    plots_generated += 1
            except Exception as e:
                logger.warning(f"生成参数分布图失败: {e}")
            
            # 生成现有的优化图表（如果数据支持）
            try:
                if (not isinstance(data, dict) and hasattr(data, 'generation_stats')) or (isinstance(data, dict) and 'history' in data):
                    # 转换数据格式以兼容现有方法
                    if not isinstance(data, dict) and hasattr(data, 'generation_stats') and data.generation_stats:
                        optimization_history = []
                        for stat in data.generation_stats:
                            optimization_history.append({
                                'result': stat.get('best_fitness', 0),
                                'params': {},  # 参数信息可能不在generation_stats中
                                'timestamp': '',
                                'evaluation_count': stat.get('generation', 0)
                            })
                        
                        self.generate_optimization_plots(data, optimization_history)
                        plots_generated += 1
            except Exception as e:
                logger.warning(f"生成现有优化图表失败: {e}")
            
            logger.info(f"可视化报告生成完成，共生成 {plots_generated} 个图表，保存至: {output_dir}")
            return plots_generated > 0
            
        except Exception as e:
            logger.error(f"生成完整可视化报告失败: {e}")
            return False