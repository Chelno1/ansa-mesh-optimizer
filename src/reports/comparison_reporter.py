#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化器比较报告生成模块

从compare_optimizers_improved.py中提取的报告生成功能
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 安全导入pandas
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("pandas不可用，部分报告功能将受限")

# 导入工具函数
try:
    from src.utils import format_execution_time
except ImportError:
    def format_execution_time(seconds):
        """备用时间格式化函数"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class ComparisonReporter:
    """优化器比较报告生成器"""
    
    def __init__(self, results_dir: Path):
        """
        初始化报告生成器
        
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_all_results(self, 
                        results: Dict[str, List[Dict[str, Any]]],
                        comparison_summary: Any,
                        comparison_metadata: Dict[str, Any],
                        optimizers: List[str],
                        failed_runs: Dict[str, int]) -> None:
        """
        保存所有结果到文件
        
        Args:
            results: 原始结果
            comparison_summary: 比较摘要
            comparison_metadata: 比较元数据
            optimizers: 优化器列表
            failed_runs: 失败运行统计
        """
        try:
            # 保存原始结果（JSON格式）
            self._save_raw_results(results)
            
            # 保存摘要
            self._save_summary(comparison_summary)
            
            # 保存元数据
            self._save_metadata(comparison_metadata, optimizers, failed_runs)
            
            # 保存详细的文本报告
            self._save_text_report(results, comparison_summary, comparison_metadata, optimizers)
            
            logger.info("所有结果已保存到文件")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _save_raw_results(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """保存原始结果"""
        # 准备可序列化的数据
        serializable_results = {}
        for optimizer, runs in results.items():
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
    
    def _save_summary(self, comparison_summary: Any) -> None:
        """保存摘要数据"""
        if PANDAS_AVAILABLE and hasattr(comparison_summary, 'to_csv'):
            try:
                # 保存为CSV
                csv_file = self.results_dir / 'comparison_summary.csv'
                comparison_summary.to_csv(csv_file, index=False)
                
                # 保存为Excel（如果可能）
                try:
                    excel_file = self.results_dir / 'comparison_summary.xlsx'
                    comparison_summary.to_excel(excel_file, index=False)
                except Exception:
                    pass  # Excel保存可能失败，忽略
                    
            except Exception as e:
                logger.warning(f"保存CSV摘要失败: {e}")
        
        # 保存为JSON
        if isinstance(comparison_summary, list):
            summary_data = comparison_summary
        else:
            try:
                if comparison_summary is not None and hasattr(comparison_summary, 'to_dict'):
                    summary_data = comparison_summary.to_dict('records')
                else:
                    summary_data = [{'error': 'Summary conversion failed'}]
            except:
                summary_data = [{'error': 'Summary conversion failed'}]
        
        with open(self.results_dir / 'comparison_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def _save_metadata(self, 
                      comparison_metadata: Dict[str, Any],
                      optimizers: List[str],
                      failed_runs: Dict[str, int]) -> None:
        """保存元数据"""
        metadata = {
            'comparison_metadata': comparison_metadata,
            'configuration': {
                'optimizers': optimizers,
                'total_optimizations': sum(len(runs) for runs in failed_runs.values()) if failed_runs else 0,
                'failed_optimizations': sum(failed_runs.values()) if failed_runs else 0,
                'optimizers_tested': len(optimizers)
            },
            'environment': {
                'pandas_available': PANDAS_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(self.results_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _save_text_report(self, 
                         results: Dict[str, List[Dict[str, Any]]],
                         comparison_summary: Any,
                         comparison_metadata: Dict[str, Any],
                         optimizers: List[str]) -> None:
        """保存详细的文本报告"""
        report_file = self.results_dir / 'detailed_report.txt'
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("优化器比较详细报告\n")
                f.write("=" * 60 + "\n\n")
                
                # 比较配置
                f.write(f"比较配置:\n")
                f.write(f"  优化器: {', '.join(optimizers)}\n")
                f.write(f"  总执行时间: {format_execution_time(comparison_metadata.get('total_execution_time', 0))}\n\n")
                
                # 摘要统计
                if isinstance(comparison_summary, list):
                    summary_data = comparison_summary
                else:
                    try:
                        summary_data = comparison_summary.to_dict('records') if hasattr(comparison_summary, 'to_dict') else []
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
                
                for optimizer, runs in results.items():
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
    
    def export_results(self, 
                      results: Dict[str, List[Dict[str, Any]]],
                      comparison_summary: Any,
                      export_format: str = 'excel') -> str:
        """
        导出结果到指定格式
        
        Args:
            results: 原始结果
            comparison_summary: 比较摘要
            export_format: 导出格式 ('excel', 'csv', 'json')
            
        Returns:
            导出文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format.lower() == 'excel' and PANDAS_AVAILABLE:
            export_file = self.results_dir / f'comparison_results_{timestamp}.xlsx'
            
            try:
                with pd.ExcelWriter(export_file) as writer:
                    # 摘要表
                    if hasattr(comparison_summary, 'to_excel'):
                        comparison_summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # 详细结果表
                    all_results = []
                    for optimizer, runs in results.items():
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
            
            if hasattr(comparison_summary, 'to_csv'):
                comparison_summary.to_csv(export_file, index=False)
            else:
                # 手动创建CSV
                import csv
                with open(export_file, 'w', newline='', encoding='utf-8') as f:
                    if isinstance(comparison_summary, list) and comparison_summary:
                        writer = csv.DictWriter(f, fieldnames=comparison_summary[0].keys())
                        writer.writeheader()
                        writer.writerows(comparison_summary)
        
        elif export_format.lower() == 'json':
            export_file = self.results_dir / f'comparison_results_{timestamp}.json'
            
            export_data = {
                'results': results,
                'summary': comparison_summary if isinstance(comparison_summary, list) else 
                          comparison_summary.to_dict('records') if hasattr(comparison_summary, 'to_dict') else None
            }
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
        
        logger.info(f"结果已导出: {export_file}")
        return str(export_file)
    
    def generate_summary_statistics(self, 
                                  results: Dict[str, List[Dict[str, Any]]],
                                  optimizers: List[str]) -> List[Dict[str, Any]]:
        """
        生成摘要统计信息
        
        Args:
            results: 原始结果
            optimizers: 优化器列表
            
        Returns:
            摘要统计列表
        """
        try:
            from src.utils import calculate_statistics
        except ImportError:
            # 备用统计计算函数
            def calculate_statistics(values):
                import numpy as np
                return {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        summary_data = []
        
        for optimizer in optimizers:
            runs = results.get(optimizer, [])
            
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
        
        # 排序（按平均最佳值）
        summary_data.sort(key=lambda x: x['mean_best_value'])
        
        return summary_data
