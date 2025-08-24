#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa批处理结果解析和报告生成模块

作者: Chel
创建日期: 2025-08-24
版本: 2.0.0
功能: 从 batch_mesh.py 中提取的报告生成逻辑
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 安全导入Ansa模块
ANSA_AVAILABLE = False
try:
    from ansa import base, constants, mesh

    ANSA_AVAILABLE = True
except ImportError:
    pass


class QualityReportGenerator:
    """质量报告生成器"""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir or Path.cwd().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_quality_report(
        self,
        quality_results: Dict[str, Any],
        runner_stats: Dict[str, Any],
        config_info: Dict[str, Any],
        output_file: Optional[Path] = None,
        include_details: bool = True,
    ) -> str:
        """
        生成质量报告 - 增强版本

        Args:
            quality_results: 质量检查结果
            runner_stats: 运行统计信息
            config_info: 配置信息
            output_file: 报告输出文件路径
            include_details: 是否包含详细信息

        Returns:
            报告文件路径
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.output_dir / f"quality_report_{timestamp}.txt"
        else:
            output_file = Path(output_file)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                self._write_report_header(f)
                self._write_overall_statistics(f, quality_results)
                self._write_thresholds_info(f, quality_results)
                self._write_detailed_checks(f, quality_results, include_details)
                self._write_runtime_statistics(f, runner_stats)
                self._write_configuration_info(f, config_info)

            logger.info(f"质量报告已生成: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            return ""

    def _write_report_header(self, f) -> None:
        """写入报告头部"""
        f.write("Ansa批处理网格质量报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Ansa可用: {ANSA_AVAILABLE}\n\n")

    def _write_overall_statistics(
        self, f, quality_results: Dict[str, Any]
    ) -> None:
        """写入总体统计"""
        f.write("总体统计:\n")
        f.write("-" * 20 + "\n")
        f.write(f"总单元数: {quality_results.get('total_elements', 'N/A')}\n")
        f.write(f"不合格单元数: {quality_results.get('bad_elements', 'N/A')}\n")
        f.write(
            f"质量比例: {quality_results.get('quality_ratio', 0.0):.2%}\n\n"
        )

    def _write_thresholds_info(self, f, quality_results: Dict[str, Any]) -> None:
        """写入阈值信息"""
        thresholds = quality_results.get("thresholds", {})
        f.write("质量阈值:\n")
        f.write("-" * 20 + "\n")
        for key, value in thresholds.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

    def _write_detailed_checks(
        self, f, quality_results: Dict[str, Any], include_details: bool
    ) -> None:
        """写入详细检查结果"""
        checks = quality_results.get("checks", {})
        f.write("详细检查结果:\n")
        f.write("-" * 30 + "\n")

        for check_name, check_result in checks.items():
            f.write(f"\n{check_name.upper()} 检查:\n")
            f.write(f"  状态: {check_result.get('status', 'N/A')}\n")
            f.write(f"  阈值: {check_result.get('threshold', 'N/A')}\n")
            f.write(f"  不合格数量: {check_result.get('failed_count', 'N/A')}\n")
            f.write(f"  检查总数: {check_result.get('total_checked', 'N/A')}\n")

            if include_details and "failed_values" in check_result:
                failed_values = check_result["failed_values"]
                if failed_values:
                    f.write(f"  最差值: {check_result.get('worst_value', 'N/A')}\n")
                    f.write(
                        f"  平均不合格值: {check_result.get('avg_failed_value', 'N/A')}\n"
                    )

    def _write_runtime_statistics(self, f, runner_stats: Dict[str, Any]) -> None:
        """写入运行统计"""
        f.write("\n\n运行统计:\n")
        f.write("-" * 20 + "\n")
        for key, value in runner_stats.items():
            if key.endswith("_time") and value:
                f.write(
                    f"{key}: {time.strftime('%H:%M:%S', time.localtime(value))}\n"
                )
            else:
                f.write(f"{key}: {value}\n")

    def _write_configuration_info(self, f, config_info: Dict[str, Any]) -> None:
        """写入配置信息"""
        f.write("\n\n配置信息:\n")
        f.write("-" * 20 + "\n")

        if "retry_attempts" in config_info:
            f.write(f"重试次数: {config_info['retry_attempts']}\n")
        if "timeout" in config_info:
            f.write(f"超时时间: {config_info['timeout']}秒\n")

        # 写入其他配置属性
        for attr_name in [
            "min_element_length",
            "max_element_length",
            "mpar_file",
            "qual_file",
        ]:
            if attr_name in config_info:
                f.write(f"{attr_name}: {config_info[attr_name]}\n")

    def generate_csv_report(
        self,
        quality_results: Dict[str, Any],
        output_file: Optional[Path] = None,
    ) -> str:
        """
        生成CSV格式的质量报告

        Args:
            quality_results: 质量检查结果
            output_file: 输出文件路径

        Returns:
            CSV报告文件路径
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.output_dir / f"quality_report_{timestamp}.csv"
        else:
            output_file = Path(output_file)

        try:
            import csv

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # 写入标题行
                writer.writerow([
                    "检查类型",
                    "状态",
                    "阈值",
                    "不合格数量",
                    "检查总数",
                    "不合格率",
                    "最差值",
                    "平均不合格值",
                ])

                # 写入检查结果
                checks = quality_results.get("checks", {})
                for check_name, check_result in checks.items():
                    failed_count = check_result.get("failed_count", 0)
                    total_checked = check_result.get("total_checked", 0)
                    failure_rate = (
                        failed_count / total_checked if total_checked > 0 else 0.0
                    )

                    writer.writerow([
                        check_name,
                        check_result.get("status", "N/A"),
                        check_result.get("threshold", "N/A"),
                        failed_count,
                        total_checked,
                        f"{failure_rate:.2%}",
                        check_result.get("worst_value", "N/A"),
                        check_result.get("avg_failed_value", "N/A"),
                    ])

            logger.info(f"CSV质量报告已生成: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"生成CSV质量报告失败: {e}")
            return ""

    def generate_json_report(
        self,
        quality_results: Dict[str, Any],
        runner_stats: Dict[str, Any],
        config_info: Dict[str, Any],
        output_file: Optional[Path] = None,
    ) -> str:
        """
        生成JSON格式的质量报告

        Args:
            quality_results: 质量检查结果
            runner_stats: 运行统计信息
            config_info: 配置信息
            output_file: 输出文件路径

        Returns:
            JSON报告文件路径
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.output_dir / f"quality_report_{timestamp}.json"
        else:
            output_file = Path(output_file)

        try:
            import json

            report_data = {
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ansa_available": ANSA_AVAILABLE,
                },
                "quality_results": quality_results,
                "runtime_stats": runner_stats,
                "configuration": config_info,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            logger.info(f"JSON质量报告已生成: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"生成JSON质量报告失败: {e}")
            return ""


class ResultAnalyzer:
    """结果分析器"""

    @staticmethod
    def analyze_quality_trends(
        quality_results_history: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        分析质量趋势

        Args:
            quality_results_history: 历史质量检查结果列表

        Returns:
            趋势分析结果
        """
        if not quality_results_history:
            return {"error": "没有历史数据"}

        try:
            # 提取关键指标
            quality_ratios = []
            bad_element_counts = []
            timestamps = []

            for result in quality_results_history:
                quality_ratios.append(result.get("quality_ratio", 0.0))
                bad_element_counts.append(result.get("bad_elements", 0))
                timestamps.append(result.get("timestamp", 0))

            # 计算趋势
            if len(quality_ratios) > 1:
                quality_trend = "improving" if quality_ratios[-1] > quality_ratios[0] else "degrading"
                avg_quality = sum(quality_ratios) / len(quality_ratios)
                quality_variance = sum((x - avg_quality) ** 2 for x in quality_ratios) / len(quality_ratios)
            else:
                quality_trend = "stable"
                avg_quality = quality_ratios[0] if quality_ratios else 0.0
                quality_variance = 0.0

            return {
                "trend": quality_trend,
                "average_quality_ratio": avg_quality,
                "quality_variance": quality_variance,
                "latest_quality_ratio": quality_ratios[-1] if quality_ratios else 0.0,
                "total_runs": len(quality_results_history),
                "improvement_percentage": (
                    ((quality_ratios[-1] - quality_ratios[0]) / quality_ratios[0] * 100)
                    if len(quality_ratios) > 1 and quality_ratios[0] > 0
                    else 0.0
                ),
            }

        except Exception as e:
            logger.error(f"分析质量趋势失败: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_worst_performing_checks(quality_results: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        获取表现最差的检查项

        Args:
            quality_results: 质量检查结果

        Returns:
            按失败率排序的检查项列表
        """
        checks = quality_results.get("checks", {})
        worst_checks = []

        for check_name, check_result in checks.items():
            failed_count = check_result.get("failed_count", 0)
            total_checked = check_result.get("total_checked", 0)
            
            if total_checked > 0:
                failure_rate = failed_count / total_checked
                worst_checks.append({
                    "check_name": check_name,
                    "failure_rate": failure_rate,
                    "failed_count": failed_count,
                    "total_checked": total_checked,
                    "status": check_result.get("status", "UNKNOWN"),
                })

        # 按失败率降序排序
        worst_checks.sort(key=lambda x: x["failure_rate"], reverse=True)
        return worst_checks

    @staticmethod
    def calculate_quality_score(quality_results: Dict[str, Any]) -> float:
        """
        计算综合质量评分 (0-100)

        Args:
            quality_results: 质量检查结果

        Returns:
            质量评分
        """
        quality_ratio = quality_results.get("quality_ratio", 0.0)
        
        # 基础分数基于质量比例
        base_score = quality_ratio * 100
        
        # 根据不同检查项的失败情况进行调整
        checks = quality_results.get("checks", {})
        penalty = 0
        
        for check_name, check_result in checks.items():
            if check_result.get("status") == "NOK":
                failed_count = check_result.get("failed_count", 0)
                total_checked = check_result.get("total_checked", 1)
                
                # 不同检查项有不同的权重
                weight = {
                    "min_length": 0.8,
                    "max_length": 0.8,
                    "aspect_ratio": 1.0,
                    "skewness": 1.2,
                    "warping": 1.0,
                    "min_angle_quads": 0.9,
                    "max_angle_quads": 0.9,
                    "min_angle_trias": 0.9,
                    "max_angle_trias": 0.9,
                }.get(check_name, 1.0)
                
                failure_rate = failed_count / total_checked
                penalty += failure_rate * weight * 10  # 最大扣分10分每项
        
        # 计算最终分数
        final_score = max(0, base_score - penalty)
        return min(100, final_score)


def generate_quality_report(
    quality_results: Dict[str, Any],
    runner_stats: Dict[str, Any],
    config_info: Dict[str, Any],
    output_file: Optional[Path] = None,
    include_details: bool = True,
) -> str:
    """
    生成质量报告的便利函数

    Args:
        quality_results: 质量检查结果
        runner_stats: 运行统计信息
        config_info: 配置信息
        output_file: 输出文件路径
        include_details: 是否包含详细信息

    Returns:
        报告文件路径
    """
    generator = QualityReportGenerator()
    return generator.generate_quality_report(
        quality_results, runner_stats, config_info, output_file, include_details
    )


def analyze_quality_results(quality_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析质量结果的便利函数

    Args:
        quality_results: 质量检查结果

    Returns:
        分析结果
    """
    worst_checks = ResultAnalyzer.get_worst_performing_checks(quality_results)
    quality_score = ResultAnalyzer.calculate_quality_score(quality_results)
    
    return {
        "quality_score": quality_score,
        "worst_performing_checks": worst_checks,
        "overall_status": "PASS" if quality_score >= 80 else "FAIL",
    }