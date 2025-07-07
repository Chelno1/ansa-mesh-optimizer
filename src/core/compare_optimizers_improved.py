#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的优化器比较工具 - 重构版本

作者: Chel
创建日期: 2025-06-19
版本: 1.3.0
更新日期: 2025-07-07
重构: 模块化架构，分离可视化、统计分析和报告生成
"""

import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# 配置日志
logger = logging.getLogger(__name__)

# 安全导入分析库
ANALYSIS_LIBS_AVAILABLE = False
try:
    import pandas as pd
    import numpy as np
    ANALYSIS_LIBS_AVAILABLE = True
    logger.info("分析库加载成功")
except ImportError as e:
    logger.warning(f"分析库未完全安装: {e}")
    # 创建占位符，避免导入错误
    class MockPandas:
        def DataFrame(self, *args, **kwargs):
            return None
    pd = MockPandas()

# 尝试导入统计库
SCIPY_AVAILABLE = False
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy不可用，将跳过高级统计分析")

# 本地模块导入
try:
    from core.ansa_mesh_optimizer_improved import MeshOptimizer, optimize_mesh_parameters
    from config.config import config_manager
    from utils.utils import performance_monitor, format_execution_time, calculate_statistics
    
    # 导入新的模块化组件
    from visualization.comparison_visualizer import ComparisonVisualizer
    from analysis.statistical_analyzer import StatisticalAnalyzer
    from reports.comparison_reporter import ComparisonReporter
except ImportError as e:
    logger.error(f"本地模块导入失败: {e}")
    raise

class OptimizationComparison:
    """优化器比较分析类 - 增强版本"""
    
    def __init__(self,
                 optimizers: Optional[List[str]] = None,
                 n_calls: int = 30,
                 evaluator_type: str = 'mock',
                 n_runs: int = 1,
                 use_cache: bool = False,
                 parallel_execution: bool = False,
                 max_workers: Optional[int] = None):
        """
        初始化优化器比较
        
        Args:
            optimizers: 要比较的优化器列表
            n_calls: 每个优化器的迭代次数
            evaluator_type: 评估器类型
            n_runs: 每个优化器的运行次数（用于统计分析）
            use_cache: 是否使用缓存
            parallel_execution: 是否并行执行
            max_workers: 最大并行工作线程数
        """
        self.optimizers = optimizers or ['bayesian', 'random', 'forest', 'genetic']
        self.n_calls = n_calls
        self.evaluator_type = evaluator_type
        self.n_runs = n_runs
        self.use_cache = use_cache
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers or min(mp.cpu_count(), len(self.optimizers))
        
        # 结果存储
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.comparison_summary: Optional[Any] = None  # 可能是DataFrame或dict
        self.execution_times: Dict[str, List[float]] = {}
        self.failed_runs: Dict[str, int] = {}
        
        # 创建结果目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(f"comparison_results_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.total_start_time = None
        self.comparison_metadata = {
            'start_time': None,
            'end_time': None,
            'total_optimizations': len(self.optimizers) * self.n_runs,
            'parallel_execution': self.parallel_execution,
            'evaluator_type': self.evaluator_type
        }
        
        logger.info(f"比较结果将保存到: {self.results_dir}")
        logger.info(f"并行执行: {self.parallel_execution}, 最大工作线程: {self.max_workers}")
    
    def run_comparison(self, 
                      run_sensitivity_analysis: bool = True,
                      save_individual_reports: bool = False,
                      **kwargs) -> Dict[str, Any]:
        """
        运行优化器比较
        
        Args:
            run_sensitivity_analysis: 是否运行敏感性分析
            save_individual_reports: 是否保存每个优化器的单独报告
            **kwargs: 其他优化器参数
            
        Returns:
            比较结果字典
        """
        logger.info(f"开始比较 {len(self.optimizers)} 个优化器")
        logger.info(f"每个优化器运行 {self.n_runs} 次，每次 {self.n_calls} 次迭代")
        
        self.total_start_time = time.time()
        self.comparison_metadata['start_time'] = datetime.now().isoformat()
        
        try:
            # 验证优化器可用性
            available_optimizers = self._check_optimizers_availability()
            if not available_optimizers:
                raise RuntimeError("没有可用的优化器")
            
            # 根据可用性更新优化器列表
            if set(available_optimizers) != set(self.optimizers):
                logger.warning(f"部分优化器不可用，使用: {available_optimizers}")
                self.optimizers = available_optimizers
            
            # 执行比较
            if self.parallel_execution and len(self.optimizers) > 1:
                self._run_parallel_comparison(run_sensitivity_analysis, save_individual_reports, **kwargs)
            else:
                self._run_sequential_comparison(run_sensitivity_analysis, save_individual_reports, **kwargs)
            
            # 生成比较摘要
            self._generate_comparison_summary()
            
            # 保存结果
            reporter = ComparisonReporter(self.results_dir)
            reporter.save_all_results(
                self.results,
                self.comparison_summary,
                self.comparison_metadata,
                self.optimizers,
                self.failed_runs
            )
            
            # 生成可视化报告
            if ANALYSIS_LIBS_AVAILABLE:
                try:
                    visualizer = ComparisonVisualizer(self.results_dir)
                    visualizer.generate_all_visualizations(self.results, self.optimizers, self.execution_times)
                    
                    if SCIPY_AVAILABLE:
                        analyzer = StatisticalAnalyzer(self.results_dir)
                        analyzer.generate_statistical_analysis(self.results, self.optimizers)
                except Exception as e:
                    logger.warning(f"生成可视化报告失败: {e}")
            
            total_execution_time = time.time() - self.total_start_time
            self.comparison_metadata['end_time'] = datetime.now().isoformat()
            self.comparison_metadata['total_execution_time'] = total_execution_time
            
            logger.info(f"比较完成！总执行时间: {format_execution_time(total_execution_time)}")
            logger.info(f"详细结果保存在: {self.results_dir}")
            
            return self._build_final_result()
            
        except Exception as e:
            logger.error(f"比较过程中发生错误: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def _check_optimizers_availability(self) -> List[str]:
        """检查优化器可用性"""
        try:
            from core.ansa_mesh_optimizer_improved import check_dependencies
            deps = check_dependencies()
            
            available_optimizers = []
            
            for optimizer in self.optimizers:
                if optimizer in ['bayesian', 'random', 'forest']:
                    if deps['skopt_available']:
                        available_optimizers.append(optimizer)
                    else:
                        logger.warning(f"优化器 {optimizer} 不可用: 需要 scikit-optimize")
                elif optimizer in ['genetic', 'parallel']:
                    available_optimizers.append(optimizer)
                else:
                    logger.warning(f"未知优化器: {optimizer}")
            
            return available_optimizers
            
        except Exception as e:
            logger.error(f"检查优化器可用性失败: {e}")
            # 返回保守的可用列表
            return [opt for opt in self.optimizers if opt in ['genetic', 'parallel']]
    
    def _run_parallel_comparison(self, run_sensitivity_analysis: bool, 
                               save_individual_reports: bool, **kwargs) -> None:
        """并行执行比较"""
        logger.info(f"使用 {self.max_workers} 个线程并行执行比较")
        
        # 创建任务列表
        tasks = []
        for optimizer in self.optimizers:
            for run_idx in range(self.n_runs):
                tasks.append((optimizer, run_idx))
        
        # 初始化结果存储
        for optimizer in self.optimizers:
            self.results[optimizer] = []
            self.execution_times[optimizer] = []
            self.failed_runs[optimizer] = 0
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_task = {
                executor.submit(self._run_single_optimization, optimizer, run_idx, kwargs): (optimizer, run_idx)
                for optimizer, run_idx in tasks
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_task):
                optimizer, run_idx = future_to_task[future]
                
                try:
                    result = future.result()
                    self.results[optimizer].append(result)
                    self.execution_times[optimizer].append(result.get('execution_time', 0))
                    
                    completed += 1
                    progress = (completed / len(tasks)) * 100
                    logger.info(f"并行执行进度: {progress:.1f}% ({completed}/{len(tasks)})")
                    
                except Exception as e:
                    logger.error(f"优化器 {optimizer} 运行 {run_idx + 1} 失败: {e}")
                    self.failed_runs[optimizer] += 1
                    
                    # 记录失败结果
                    error_result = {
                        'optimizer': optimizer,
                        'run_index': run_idx,
                        'error': str(e),
                        'best_value': float('inf'),
                        'execution_time': 0,
                        'failed': True
                    }
                    self.results[optimizer].append(error_result)
    
    def _run_sequential_comparison(self, run_sensitivity_analysis: bool, 
                                 save_individual_reports: bool, **kwargs) -> None:
        """顺序执行比较"""
        for optimizer in self.optimizers:
            self.results[optimizer] = []
            self.execution_times[optimizer] = []
            self.failed_runs[optimizer] = 0
            
            logger.info(f"\n{'='*50}")
            logger.info(f"测试优化器: {optimizer}")
            logger.info(f"{'='*50}")
            
            for run_idx in range(self.n_runs):
                logger.info(f"运行 {run_idx + 1}/{self.n_runs}")
                
                try:
                    result = self._run_single_optimization(optimizer, run_idx, kwargs)
                    self.results[optimizer].append(result)
                    self.execution_times[optimizer].append(result.get('execution_time', 0))
                    
                    logger.info(f"运行完成: 最佳值={result['best_value']:.6f}, "
                              f"时间={format_execution_time(result['execution_time'])}")
                    
                except Exception as e:
                    logger.error(f"优化器 {optimizer} 运行 {run_idx + 1} 失败: {e}")
                    self.failed_runs[optimizer] += 1
                    
                    # 记录失败结果
                    error_result = {
                        'optimizer': optimizer,
                        'run_index': run_idx,
                        'error': str(e),
                        'best_value': float('inf'),
                        'execution_time': 0,
                        'failed': True
                    }
                    self.results[optimizer].append(error_result)
    
    def _run_single_optimization(self, optimizer: str, run_idx: int, 
                               extra_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """运行单次优化"""
        # 为每次运行设置不同的随机种子
        run_config = config_manager.optimization_config
        original_seed = run_config.random_state
        run_config.random_state = original_seed + run_idx * 1000 + hash(optimizer) % 1000
        
        try:
            with performance_monitor(f"{optimizer} 优化 (运行 {run_idx + 1})"):
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
                    **extra_kwargs
                )
                
                # 添加运行特定信息
                result.update({
                    'run_index': run_idx,
                    'optimizer': optimizer,
                    'evaluator_type': self.evaluator_type,
                    'random_seed': run_config.random_state
                })
                
                return result
                
        finally:
            # 恢复原始随机种子
            run_config.random_state = original_seed
    
    def _generate_comparison_summary(self) -> None:
        """生成比较摘要"""
        summary_data = []
        
        for optimizer in self.optimizers:
            runs = self.results.get(optimizer, [])
            
            # 过滤成功的运行
            successful_runs = [r for r in runs if not r.get('failed', False)]
            failed_count = len(runs) - len(successful_runs)
            
            if not successful_runs:
                logger.warning(f"优化器 {optimizer} 没有成功的运行")
                continue
            
            # 计算统计指标
            best_values = [r['best_value'] for r in successful_runs]
            execution_times = [r.get('execution_time', 0) for r in successful_runs]
            
            # 基础统计
            stats = calculate_statistics(best_values)
            time_stats = calculate_statistics(execution_times)
            
            summary_entry = {
                'optimizer': optimizer,
                'successful_runs': len(successful_runs),
                'failed_runs': failed_count,
                'success_rate': len(successful_runs) / len(runs) if runs else 0,
                
                # 性能统计
                'mean_best_value': stats['mean'],
                'std_best_value': stats['std'],
                'min_best_value': stats['min'],
                'max_best_value': stats['max'],
                'median_best_value': stats['median'],
                'q25_best_value': stats['q25'],
                'q75_best_value': stats['q75'],
                
                # 时间统计
                'mean_execution_time': time_stats['mean'],
                'std_execution_time': time_stats['std'],
                'min_execution_time': time_stats['min'],
                'max_execution_time': time_stats['max'],
                
                # 效率指标
                'efficiency_score': stats['mean'] / time_stats['mean'] if time_stats['mean'] > 0 else float('inf'),
                'robustness_score': 1 / (1 + stats['std']) if stats['std'] > 0 else 1,
                
                # 原始数据
                'best_values': best_values,
                'execution_times': execution_times
            }
            
            summary_data.append(summary_entry)
        
        # 创建摘要
        if ANALYSIS_LIBS_AVAILABLE:
            try:
                self.comparison_summary = pd.DataFrame(summary_data)
                # 排序（按平均最佳值）
                if hasattr(self.comparison_summary, 'empty') and not self.comparison_summary.empty:
                    self.comparison_summary = self.comparison_summary.sort_values('mean_best_value')
            except Exception as e:
                logger.warning(f"创建pandas DataFrame失败: {e}")
                self.comparison_summary = summary_data
        else:
            self.comparison_summary = summary_data
        
        # 排序（如果是列表）
        if isinstance(self.comparison_summary, list):
            self.comparison_summary.sort(key=lambda x: x['mean_best_value'])
        
        logger.info(f"\n比较摘要已生成，包含 {len(summary_data)} 个优化器的结果")
    
    def _save_results(self) -> None:
        """保存结果到文件"""
        try:
            # 保存原始结果（JSON格式）
            self._save_raw_results()
            
            # 保存摘要
            self._save_summary()
            
            # 保存元数据
            self._save_metadata()
            
            # 保存详细的文本报告
            self._save_text_report()
            
            logger.info("所有结果已保存到文件")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _save_raw_results(self) -> None:
        """保存原始结果"""
        import json
        
        # 准备可序列化的数据
        serializable_results = {}
        for optimizer, runs in self.results.items():
            serializable_results[optimizer] = []
            for run in runs:
                # 移除不可序列化的对象
                clean_run = {}
                for key, value in run.items():
                    if key in ['skopt_result', 'genetic_result']:
                        continue  # 跳过复杂对象
                    try:
                        json.dumps(value)  # 测试是否可序列化
                        clean_run[key] = value
                    except (TypeError, ValueError):
                        clean_run[key] = str(value)
                
                serializable_results[optimizer].append(clean_run)
        
        with open(self.results_dir / 'raw_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def _save_summary(self) -> None:
        """保存摘要数据"""
        if ANALYSIS_LIBS_AVAILABLE and hasattr(self.comparison_summary, 'to_csv'):
            try:
                # 保存为CSV
                csv_file = self.results_dir / 'comparison_summary.csv'
                self.comparison_summary.to_csv(csv_file, index=False)
                
                # 保存为Excel（如果可能）
                try:
                    excel_file = self.results_dir / 'comparison_summary.xlsx'
                    self.comparison_summary.to_excel(excel_file, index=False)
                except Exception:
                    pass  # Excel保存可能失败，忽略
                    
            except Exception as e:
                logger.warning(f"保存CSV摘要失败: {e}")
        
        # 保存为JSON
        if isinstance(self.comparison_summary, list):
            summary_data = self.comparison_summary
        else:
            try:
                if self.comparison_summary is not None and hasattr(self.comparison_summary, 'to_dict'):
                    summary_data = self.comparison_summary.to_dict('records')
                else:
                    summary_data = [{'error': 'Summary conversion failed'}]
            except:
                summary_data = [{'error': 'Summary conversion failed'}]
        
        import json
        with open(self.results_dir / 'comparison_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def _save_metadata(self) -> None:
        """保存元数据"""
        import json
        
        metadata = {
            'comparison_metadata': self.comparison_metadata,
            'configuration': {
                'optimizers': self.optimizers,
                'n_calls': self.n_calls,
                'n_runs': self.n_runs,
                'evaluator_type': self.evaluator_type,
                'use_cache': self.use_cache,
                'parallel_execution': self.parallel_execution,
                'max_workers': self.max_workers
            },
            'statistics': {
                'total_optimizations': sum(len(runs) for runs in self.results.values()),
                'successful_optimizations': sum(len([r for r in runs if not r.get('failed', False)]) 
                                              for runs in self.results.values()),
                'failed_optimizations': sum(self.failed_runs.values()),
                'optimizers_tested': len(self.optimizers)
            },
            'environment': {
                'analysis_libs_available': ANALYSIS_LIBS_AVAILABLE,
                'scipy_available': SCIPY_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(self.results_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _save_text_report(self) -> None:
        """保存详细的文本报告"""
        report_file = self.results_dir / 'detailed_report.txt'
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("优化器比较详细报告\n")
                f.write("=" * 60 + "\n\n")
                
                # 比较配置
                f.write(f"比较配置:\n")
                f.write(f"  优化器: {', '.join(self.optimizers)}\n")
                f.write(f"  迭代次数: {self.n_calls}\n")
                f.write(f"  运行次数: {self.n_runs}\n")
                f.write(f"  评估器类型: {self.evaluator_type}\n")
                f.write(f"  使用缓存: {self.use_cache}\n")
                f.write(f"  并行执行: {self.parallel_execution}\n")
                f.write(f"  总执行时间: {format_execution_time(self.comparison_metadata.get('total_execution_time', 0))}\n\n")
                
                # 摘要统计
                if isinstance(self.comparison_summary, list):
                    summary_data = self.comparison_summary
                else:
                    try:
                        summary_data = self.comparison_summary.to_dict('records')
                    except:
                        summary_data = []
                
                if summary_data:
                    f.write("摘要统计:\n")
                    f.write("-" * 40 + "\n")
                    
                    for entry in summary_data:
                        optimizer = entry.get('optimizer', 'Unknown')
                        f.write(f"\n{optimizer}:\n")
                        f.write(f"  成功运行: {entry.get('successful_runs', 0)}/{entry.get('successful_runs', 0) + entry.get('failed_runs', 0)}\n")
                        f.write(f"  平均最佳值: {entry.get('mean_best_value', 0):.6f} ± {entry.get('std_best_value', 0):.6f}\n")
                        f.write(f"  最佳值范围: [{entry.get('min_best_value', 0):.6f}, {entry.get('max_best_value', 0):.6f}]\n")
                        f.write(f"  平均执行时间: {format_execution_time(entry.get('mean_execution_time', 0))}\n")
                        f.write(f"  效率分数: {entry.get('efficiency_score', 0):.6f}\n")
                        f.write(f"  鲁棒性分数: {entry.get('robustness_score', 0):.6f}\n")
                    
                    f.write("\n")
                
                # 详细结果
                f.write("详细运行结果:\n")
                f.write("-" * 40 + "\n")
                
                for optimizer, runs in self.results.items():
                    f.write(f"\n{optimizer}:\n")
                    
                    for i, run in enumerate(runs):
                        f.write(f"  运行 {i+1}:\n")
                        if run.get('failed', False):
                            f.write(f"    状态: 失败\n")
                            f.write(f"    错误: {run.get('error', 'Unknown')}\n")
                        else:
                            f.write(f"    最佳值: {run.get('best_value', 0):.6f}\n")
                            f.write(f"    执行时间: {format_execution_time(run.get('execution_time', 0))}\n")
                            if 'best_params' in run:
                                f.write(f"    最佳参数: {run['best_params']}\n")
                        f.write("\n")
            
            logger.info(f"详细报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"保存详细报告失败: {e}")
    
    
    def _build_final_result(self) -> Dict[str, Any]:
        """构建最终结果"""
        result = {
            'results': self.results,
            'execution_times': self.execution_times,
            'failed_runs': self.failed_runs,
            'results_dir': str(self.results_dir),
            'metadata': self.comparison_metadata
        }
        
        # 添加摘要
        if isinstance(self.comparison_summary, list):
            result['summary'] = self.comparison_summary
        else:
            try:
                result['summary'] = self.comparison_summary.to_dict('records')
            except:
                result['summary'] = None
        
        # 确定最佳优化器
        try:
            best_optimizer, best_info = self.get_best_optimizer()
            result['best_optimizer'] = best_optimizer
            result['best_optimizer_info'] = best_info
        except Exception as e:
            logger.warning(f"无法确定最佳优化器: {e}")
        
        return result
    
    def get_best_optimizer(self) -> Tuple[str, Dict[str, Any]]:
        """
        获取最佳优化器
        
        Returns:
            (最佳优化器名称, 详细信息)
        """
        if isinstance(self.comparison_summary, list):
            if not self.comparison_summary:
                raise ValueError("没有比较结果可用")
            best_entry = self.comparison_summary[0]  # 已经排序
        else:
            if self.comparison_summary is None or self.comparison_summary.empty:
                raise ValueError("没有比较结果可用")
            best_entry = self.comparison_summary.iloc[0].to_dict()
        
        best_info = {
            'optimizer': best_entry['optimizer'],
            'mean_best_value': best_entry['mean_best_value'],
            'std_best_value': best_entry['std_best_value'],
            'mean_execution_time': best_entry['mean_execution_time'],
            'successful_runs': best_entry['successful_runs'],
            'success_rate': best_entry['success_rate'],
            'efficiency_score': best_entry.get('efficiency_score', 0),
            'robustness_score': best_entry.get('robustness_score', 0)
        }
        
        return best_entry['optimizer'], best_info
    
    def export_results(self, export_format: str = 'excel') -> str:
        """
        导出结果到指定格式
        
        Args:
            export_format: 导出格式 ('excel', 'csv', 'json')
            
        Returns:
            导出文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format.lower() == 'excel' and ANALYSIS_LIBS_AVAILABLE:
            export_file = self.results_dir / f'comparison_results_{timestamp}.xlsx'
            
            try:
                with pd.ExcelWriter(export_file) as writer:
                    # 摘要表
                    if hasattr(self.comparison_summary, 'to_excel'):
                        self.comparison_summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # 详细结果表
                    all_results = []
                    for optimizer, runs in self.results.items():
                        for run in runs:
                            if not run.get('failed', False):
                                result_row = {
                                    'optimizer': optimizer,
                                    'run_index': run.get('run_index', 0),
                                    'best_value': run.get('best_value', 0),
                                    'execution_time': run.get('execution_time', 0)
                                }
                                # 添加最佳参数
                                if 'best_params' in run:
                                    for param_name, param_value in run['best_params'].items():
                                        result_row[f'param_{param_name}'] = param_value
                                all_results.append(result_row)
                    
                    if all_results:
                        detailed_df = pd.DataFrame(all_results)
                        detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                        
            except Exception as e:
                logger.error(f"Excel导出失败: {e}")
                export_format = 'csv'  # 降级到CSV
        
        if export_format.lower() == 'csv':
            export_file = self.results_dir / f'comparison_results_{timestamp}.csv'
            
            if hasattr(self.comparison_summary, 'to_csv'):
                self.comparison_summary.to_csv(export_file, index=False)
            else:
                # 手动创建CSV
                import csv
                with open(export_file, 'w', newline='', encoding='utf-8') as f:
                    if isinstance(self.comparison_summary, list) and self.comparison_summary:
                        writer = csv.DictWriter(f, fieldnames=self.comparison_summary[0].keys())
                        writer.writeheader()
                        writer.writerows(self.comparison_summary)
        
        elif export_format.lower() == 'json':
            export_file = self.results_dir / f'comparison_results_{timestamp}.json'
            
            export_data = self._build_final_result()
            
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
    parallel_execution: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷的优化器比较函数 - 增强版本
    
    Args:
        optimizers: 要比较的优化器列表
        n_calls: 每个优化器的迭代次数
        n_runs: 每个优化器的运行次数
        evaluator_type: 评估器类型
        run_sensitivity_analysis: 是否运行敏感性分析
        generate_report: 是否生成详细报告
        parallel_execution: 是否并行执行
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
        use_cache=False,  # 比较时不使用缓存确保公平性
        parallel_execution=parallel_execution
    )
    
    # 运行比较
    results = comparison.run_comparison(
        run_sensitivity_analysis=run_sensitivity_analysis,
        save_individual_reports=generate_report,
        **kwargs
    )
    
    # 导出结果
    if generate_report:
        try:
            if ANALYSIS_LIBS_AVAILABLE:
                export_file = comparison.export_results('excel')
            else:
                export_file = comparison.export_results('json')
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
    
    try:
        # 快速比较（使用模拟评估器）
        results = compare_optimizers(
            optimizers=['random', 'genetic'],
            n_calls=10,
            n_runs=2,
            evaluator_type='mock',
            parallel_execution=False
        )
        
        print("\n比较结果摘要:")
        if 'summary' in results and results['summary']:
            for entry in results['summary']:
                optimizer = entry['optimizer']
                mean_val = entry['mean_best_value']
                std_val = entry['std_best_value']
                exec_time = entry['mean_execution_time']
                print(f"{optimizer:12}: {mean_val:.6f} ± {std_val:.6f} ({format_execution_time(exec_time)})")
        
        if 'best_optimizer' in results:
            print(f"\n推荐的最佳优化器: {results['best_optimizer']}")
        
        print(f"\n详细结果保存在: {results['results_dir']}")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        traceback.print_exc()