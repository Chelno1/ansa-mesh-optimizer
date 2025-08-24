#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化过程报告生成模块

从ansa_mesh_optimizer_refactored.py中提取的报告生成功能
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 导入工具函数
try:
    from ..utils import format_execution_time
except ImportError:

    def format_execution_time(seconds):
        """备用时间格式化函数"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class OptimizationReporter:
    """优化过程报告生成器"""

    def __init__(self, report_dir: Optional[Path] = None):
        """
        初始化报告生成器

        Args:
            report_dir: 报告保存目录
        """
        if report_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path(f"optimization_reports/{timestamp}_optimization")

        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_optimization_report(
        self,
        result: Dict[str, Any],
        optimization_history: List[Dict[str, Any]],
        config: Any,
        param_space: Any,
        cache_stats: Optional[Dict[str, Any]] = None,
        early_stopping_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成完整的优化报告

        Args:
            result: 优化结果
            optimization_history: 优化历史
            config: 优化配置
            param_space: 参数空间
            cache_stats: 缓存统计（可选）
            early_stopping_info: 早停信息（可选）

        Returns:
            报告目录路径
        """
        try:
            # 生成文本报告
            report_file = self.report_dir / "optimization_report.txt"
            self._write_text_report(
                report_file,
                result,
                optimization_history,
                config,
                cache_stats,
                early_stopping_info,
            )

            # 生成数据文件
            self._save_optimization_data(
                result, optimization_history, config, param_space
            )

            logger.info(f"详细报告已保存到: {self.report_dir}")
            return str(self.report_dir)

        except Exception as e:
            logger.error(f"生成优化报告失败: {e}")
            raise

    def _write_text_report(
        self,
        report_file: Path,
        result: Dict[str, Any],
        optimization_history: List[Dict[str, Any]],
        config: Any,
        cache_stats: Optional[Dict[str, Any]] = None,
        early_stopping_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """写入文本报告"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"优化报告 - {result.get('optimizer_name', 'Unknown')}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write(f"优化器: {result.get('optimizer', 'Unknown')}\n")
            f.write(f"迭代次数: {result.get('n_calls', 'N/A')}\n")
            f.write(f"总评估次数: {result.get('total_evaluations', 'N/A')}\n")
            f.write(f"最佳目标值: {result.get('best_value', 0):.6f}\n\n")

            f.write("最佳参数:\n")
            best_params = result.get("best_params", {})
            for name, value in best_params.items():
                f.write(f"  {name}: {value}\n")
            f.write("\n")

            # 收敛信息
            # Handle both dictionary and object result formats
            if isinstance(result, dict) and "convergence_info" in result:
                conv_info = result["convergence_info"]
            elif hasattr(result, "convergence_info"):
                conv_info = getattr(result, "convergence_info", None)
            else:
                conv_info = None

            if conv_info:
                f.write("收敛信息:\n")
                f.write(f"  最佳迭代: {conv_info.get('best_iteration', 'N/A')}\n")
                f.write(f"  改进比例: {conv_info.get('improvement_ratio', 0.0):.2%}\n")
                f.write("\n")

            # 缓存统计
            if cache_stats:
                f.write("缓存统计:\n")
                for key, value in cache_stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # 早停信息
            if early_stopping_info:
                f.write("早停信息:\n")
                for key, value in early_stopping_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # 优化历史摘要
            if optimization_history:
                f.write("优化历史摘要:\n")
                f.write(f"  总评估次数: {len(optimization_history)}\n")

                results = [entry["result"] for entry in optimization_history]
                f.write(f"  最佳值: {min(results):.6f}\n")
                f.write(f"  最差值: {max(results):.6f}\n")
                f.write(f"  平均值: {sum(results)/len(results):.6f}\n")

                # 计算改进次数
                improvements = sum(
                    1 for i in range(1, len(results)) if results[i] < results[i - 1]
                )
                f.write(f"  改进次数: {improvements}\n")
                f.write(f"  改进率: {improvements/(len(results)-1)*100:.1f}%\n")

    def _save_optimization_data(
        self,
        result: Dict[str, Any],
        optimization_history: List[Dict[str, Any]],
        config: Any,
        param_space: Any,
    ) -> None:
        """保存优化数据"""
        try:
            # 保存参数历史
            history_file = self.report_dir / "optimization_history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(optimization_history, f, indent=2, ensure_ascii=False)

            # 保存最佳参数
            best_params_file = self.report_dir / "best_parameters.json"
            with open(best_params_file, "w", encoding="utf-8") as f:
                json.dump(
                    result.get("best_params", {}), f, indent=2, ensure_ascii=False
                )

            # 保存配置信息
            config_file = self.report_dir / "optimization_config.json"
            config_data = {
                "optimizer": result.get("optimizer", "Unknown"),
                "n_calls": result.get("n_calls", 0),
                "config": self._serialize_config(config),
                "parameter_space": self._serialize_param_space(param_space),
            }

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"保存优化数据失败: {e}")

    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        """序列化配置对象"""
        try:
            if hasattr(config, "__dict__"):
                config_dict = {}
                for key, value in config.__dict__.items():
                    try:
                        json.dumps(value)  # 测试是否可序列化
                        config_dict[key] = value
                    except (TypeError, ValueError):
                        config_dict[key] = str(value)
                return config_dict
            else:
                return {"config": str(config)}
        except Exception as e:
            logger.warning(f"配置序列化失败: {e}")
            return {"error": "Config serialization failed"}

    def _serialize_param_space(self, param_space: Any) -> Dict[str, Any]:
        """序列化参数空间对象"""
        try:
            param_space_dict = {}

            if hasattr(param_space, "get_param_names"):
                param_space_dict["param_names"] = param_space.get_param_names()

            if hasattr(param_space, "get_bounds"):
                param_space_dict["bounds"] = param_space.get_bounds()

            if hasattr(param_space, "get_param_types"):
                param_types = param_space.get_param_types()
                # 转换类型为字符串
                param_space_dict["param_types"] = [
                    t.__name__ if hasattr(t, "__name__") else str(t)
                    for t in param_types
                ]

            return param_space_dict

        except Exception as e:
            logger.warning(f"参数空间序列化失败: {e}")
            return {"error": "Parameter space serialization failed"}

    def save_best_params(
        self,
        best_params: Dict[str, Any],
        best_value: float,
        optimizer_name: str,
        total_evaluations: int,
        filename: Optional[str] = None,
    ) -> str:
        """
        保存最佳参数到文件

        Args:
            best_params: 最佳参数
            best_value: 最佳值
            optimizer_name: 优化器名称
            total_evaluations: 总评估次数
            filename: 保存文件名（可选）

        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = str(self.report_dir / "best_parameters.txt")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Best Mesh Parameters - {optimizer_name}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Best Objective Value: {best_value:.6f}\n")
                f.write(f"# Total Evaluations: {total_evaluations}\n\n")

                for key, value in best_params.items():
                    f.write(f"{key} = {value}\n")

            logger.info(f"最佳参数已保存到: {filename}")
            return filename

        except Exception as e:
            logger.error(f"保存最佳参数失败: {e}")
            raise

    def generate_performance_summary(
        self, optimization_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成性能摘要

        Args:
            optimization_history: 优化历史

        Returns:
            性能摘要字典
        """
        if not optimization_history:
            return {"error": "No optimization history available"}

        try:
            results = [entry["result"] for entry in optimization_history]

            # 基础统计
            import numpy as np

            summary = {
                "total_evaluations": len(optimization_history),
                "best_value": float(np.min(results)),
                "worst_value": float(np.max(results)),
                "mean_value": float(np.mean(results)),
                "std_value": float(np.std(results)),
                "median_value": float(np.median(results)),
            }

            # 改进统计
            improvements = []
            best_so_far = float("inf")
            for i, result in enumerate(results):
                if result < best_so_far:
                    best_so_far = result
                    improvements.append(i)

            summary.update(
                {
                    "improvement_count": len(improvements),
                    "improvement_rate": (
                        len(improvements) / len(results) if results else 0
                    ),
                    "improvement_iterations": improvements,
                    "convergence_iteration": (
                        improvements[-1] if improvements else len(results) - 1
                    ),
                }
            )

            # 计算改进比例
            if len(results) > 1:
                initial_value = results[0]
                final_value = min(results)
                if initial_value != 0:
                    improvement_ratio = (initial_value - final_value) / initial_value
                    summary["total_improvement_ratio"] = max(0.0, improvement_ratio)
                else:
                    summary["total_improvement_ratio"] = 0.0
            else:
                summary["total_improvement_ratio"] = 0.0

            return summary

        except Exception as e:
            logger.error(f"生成性能摘要失败: {e}")
            return {"error": f"Performance summary generation failed: {e}"}

    def export_optimization_data(
        self,
        result: Dict[str, Any],
        optimization_history: List[Dict[str, Any]],
        export_format: str = "json",
    ) -> str:
        """
        导出优化数据

        Args:
            result: 优化结果
            optimization_history: 优化历史
            export_format: 导出格式 ('json', 'csv')

        Returns:
            导出文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format.lower() == "json":
            export_file = self.report_dir / f"optimization_data_{timestamp}.json"

            export_data = {
                "result": result,
                "optimization_history": optimization_history,
                "performance_summary": self.generate_performance_summary(
                    optimization_history
                ),
                "export_timestamp": datetime.now().isoformat(),
            }

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        elif export_format.lower() == "csv":
            export_file = self.report_dir / f"optimization_history_{timestamp}.csv"

            try:
                import pandas as pd

                # 转换优化历史为DataFrame
                history_data = []
                for i, entry in enumerate(optimization_history):
                    row = {
                        "iteration": i + 1,
                        "result": entry["result"],
                        "timestamp": entry.get("timestamp", ""),
                        "evaluation_count": entry.get("evaluation_count", i + 1),
                    }

                    # 添加参数
                    params = entry.get("params", {})
                    for param_name, param_value in params.items():
                        row[f"param_{param_name}"] = param_value

                    history_data.append(row)

                df = pd.DataFrame(history_data)
                df.to_csv(export_file, index=False)

            except ImportError:
                logger.warning("pandas不可用，无法导出CSV格式")
                return self.export_optimization_data(
                    result, optimization_history, "json"
                )

        else:
            raise ValueError(f"不支持的导出格式: {export_format}")

        logger.info(f"优化数据已导出: {export_file}")
        return str(export_file)
