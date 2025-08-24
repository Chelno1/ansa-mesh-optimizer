#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa批处理报告模块的单元测试

作者: Chel
创建日期: 2025-08-24
版本: 2.0.0
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.batch.report import (
    QualityReportGenerator,
    ResultAnalyzer,
    generate_quality_report,
    analyze_quality_results,
)


class TestQualityReportGenerator(unittest.TestCase):
    """QualityReportGenerator类的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = QualityReportGenerator(self.temp_dir)
        
        # 创建测试数据
        self.quality_results = {
            "timestamp": 1234567890,
            "total_elements": 1000,
            "bad_elements": 50,
            "quality_ratio": 0.95,
            "thresholds": {
                "min_length": 2.0,
                "max_length": 8.0,
                "aspect_ratio": 3.0,
            },
            "checks": {
                "min_length": {
                    "status": "NOK",
                    "threshold": 2.0,
                    "failed_count": 30,
                    "total_checked": 1000,
                    "worst_value": 1.5,
                    "avg_failed_value": 1.8,
                    "failed_values": [1.5, 1.7, 1.9],
                },
                "max_length": {
                    "status": "NOK",
                    "threshold": 8.0,
                    "failed_count": 20,
                    "total_checked": 1000,
                    "worst_value": 10.0,
                    "avg_failed_value": 9.5,
                    "failed_values": [8.5, 9.0, 10.0],
                },
            },
        }
        
        self.runner_stats = {
            "start_time": 1234567880,
            "end_time": 1234567890,
            "retry_count": 1,
            "success": True,
        }
        
        self.config_info = {
            "retry_attempts": 3,
            "timeout": 300,
            "min_element_length": 2.0,
            "max_element_length": 8.0,
        }

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.generator.output_dir, self.temp_dir)
        self.assertTrue(self.temp_dir.exists())

    def test_generate_quality_report_default_file(self):
        """测试生成质量报告 - 默认文件名"""
        report_file = self.generator.generate_quality_report(
            self.quality_results,
            self.runner_stats,
            self.config_info
        )
        
        self.assertTrue(report_file)
        report_path = Path(report_file)
        self.assertTrue(report_path.exists())
        self.assertTrue(report_path.name.startswith("quality_report_"))
        self.assertTrue(report_path.name.endswith(".txt"))

    def test_generate_quality_report_custom_file(self):
        """测试生成质量报告 - 自定义文件名"""
        custom_file = self.temp_dir / "custom_report.txt"
        
        report_file = self.generator.generate_quality_report(
            self.quality_results,
            self.runner_stats,
            self.config_info,
            output_file=custom_file
        )
        
        self.assertEqual(report_file, str(custom_file))
        self.assertTrue(custom_file.exists())

    def test_generate_quality_report_content(self):
        """测试质量报告内容"""
        report_file = self.generator.generate_quality_report(
            self.quality_results,
            self.runner_stats,
            self.config_info
        )
        
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查关键内容
        self.assertIn("Ansa批处理网格质量报告", content)
        self.assertIn("总单元数: 1000", content)
        self.assertIn("不合格单元数: 50", content)
        self.assertIn("质量比例: 95.00%", content)
        self.assertIn("MIN_LENGTH 检查:", content)
        self.assertIn("最差值: 1.5", content)

    def test_generate_quality_report_without_details(self):
        """测试生成质量报告 - 不包含详细信息"""
        report_file = self.generator.generate_quality_report(
            self.quality_results,
            self.runner_stats,
            self.config_info,
            include_details=False
        )
        
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 不应该包含详细的最差值信息
        self.assertNotIn("最差值:", content)
        self.assertNotIn("平均不合格值:", content)

    def test_generate_csv_report(self):
        """测试生成CSV报告"""
        csv_file = self.generator.generate_csv_report(self.quality_results)
        
        self.assertTrue(csv_file)
        csv_path = Path(csv_file)
        self.assertTrue(csv_path.exists())
        self.assertTrue(csv_path.name.endswith(".csv"))
        
        # 检查CSV内容
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        self.assertIn("检查类型,状态,阈值", content)
        self.assertIn("min_length,NOK,2.0", content)

    def test_generate_json_report(self):
        """测试生成JSON报告"""
        json_file = self.generator.generate_json_report(
            self.quality_results,
            self.runner_stats,
            self.config_info
        )
        
        self.assertTrue(json_file)
        json_path = Path(json_file)
        self.assertTrue(json_path.exists())
        self.assertTrue(json_path.name.endswith(".json"))
        
        # 检查JSON内容
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("quality_results", data)
        self.assertIn("runtime_stats", data)
        self.assertIn("configuration", data)

    def test_write_report_sections(self):
        """测试报告各部分的写入"""
        # 这是一个私有方法测试，主要是为了覆盖率
        output_file = self.temp_dir / "test_sections.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            self.generator._write_report_header(f)
            self.generator._write_overall_statistics(f, self.quality_results)
            self.generator._write_thresholds_info(f, self.quality_results)
            self.generator._write_detailed_checks(f, self.quality_results, True)
            self.generator._write_runtime_statistics(f, self.runner_stats)
            self.generator._write_configuration_info(f, self.config_info)
        
        # 验证文件存在且有内容
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)


class TestResultAnalyzer(unittest.TestCase):
    """ResultAnalyzer类的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.quality_results = {
            "quality_ratio": 0.85,
            "bad_elements": 150,
            "total_elements": 1000,
            "checks": {
                "min_length": {
                    "status": "NOK",
                    "failed_count": 80,
                    "total_checked": 1000,
                },
                "max_length": {
                    "status": "NOK",
                    "failed_count": 70,
                    "total_checked": 1000,
                },
                "aspect_ratio": {
                    "status": "OK",
                    "failed_count": 0,
                    "total_checked": 1000,
                },
            },
        }

    def test_analyze_quality_trends_empty_history(self):
        """测试分析空的质量趋势历史"""
        result = ResultAnalyzer.analyze_quality_trends([])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "没有历史数据")

    def test_analyze_quality_trends_single_result(self):
        """测试分析单个质量结果的趋势"""
        history = [{"quality_ratio": 0.9, "bad_elements": 100, "timestamp": 1000}]
        
        result = ResultAnalyzer.analyze_quality_trends(history)
        
        self.assertEqual(result["trend"], "stable")
        self.assertEqual(result["average_quality_ratio"], 0.9)
        self.assertEqual(result["latest_quality_ratio"], 0.9)
        self.assertEqual(result["total_runs"], 1)

    def test_analyze_quality_trends_improving(self):
        """测试分析改善的质量趋势"""
        history = [
            {"quality_ratio": 0.7, "bad_elements": 300, "timestamp": 1000},
            {"quality_ratio": 0.8, "bad_elements": 200, "timestamp": 2000},
            {"quality_ratio": 0.9, "bad_elements": 100, "timestamp": 3000},
        ]
        
        result = ResultAnalyzer.analyze_quality_trends(history)
        
        self.assertEqual(result["trend"], "improving")
        self.assertEqual(result["latest_quality_ratio"], 0.9)
        self.assertAlmostEqual(result["average_quality_ratio"], 0.8, places=1)
        self.assertGreater(result["improvement_percentage"], 0)

    def test_analyze_quality_trends_degrading(self):
        """测试分析恶化的质量趋势"""
        history = [
            {"quality_ratio": 0.9, "bad_elements": 100, "timestamp": 1000},
            {"quality_ratio": 0.8, "bad_elements": 200, "timestamp": 2000},
            {"quality_ratio": 0.7, "bad_elements": 300, "timestamp": 3000},
        ]
        
        result = ResultAnalyzer.analyze_quality_trends(history)
        
        self.assertEqual(result["trend"], "degrading")
        self.assertEqual(result["latest_quality_ratio"], 0.7)
        self.assertLess(result["improvement_percentage"], 0)

    def test_get_worst_performing_checks(self):
        """测试获取表现最差的检查项"""
        worst_checks = ResultAnalyzer.get_worst_performing_checks(self.quality_results)
        
        self.assertEqual(len(worst_checks), 3)
        
        # 应该按失败率降序排序
        self.assertEqual(worst_checks[0]["check_name"], "min_length")
        self.assertEqual(worst_checks[0]["failure_rate"], 0.08)  # 80/1000
        
        self.assertEqual(worst_checks[1]["check_name"], "max_length")
        self.assertEqual(worst_checks[1]["failure_rate"], 0.07)  # 70/1000
        
        self.assertEqual(worst_checks[2]["check_name"], "aspect_ratio")
        self.assertEqual(worst_checks[2]["failure_rate"], 0.0)   # 0/1000

    def test_get_worst_performing_checks_empty(self):
        """测试获取表现最差的检查项 - 空检查"""
        empty_results = {"checks": {}}
        worst_checks = ResultAnalyzer.get_worst_performing_checks(empty_results)
        
        self.assertEqual(len(worst_checks), 0)

    def test_calculate_quality_score_high_quality(self):
        """测试计算高质量评分"""
        high_quality_results = {
            "quality_ratio": 0.95,
            "checks": {
                "min_length": {"status": "OK", "failed_count": 5, "total_checked": 1000},
            },
        }
        
        score = ResultAnalyzer.calculate_quality_score(high_quality_results)
        
        self.assertGreater(score, 90)
        self.assertLessEqual(score, 100)

    def test_calculate_quality_score_low_quality(self):
        """测试计算低质量评分"""
        low_quality_results = {
            "quality_ratio": 0.5,
            "checks": {
                "min_length": {"status": "NOK", "failed_count": 300, "total_checked": 1000},
                "max_length": {"status": "NOK", "failed_count": 200, "total_checked": 1000},
            },
        }
        
        score = ResultAnalyzer.calculate_quality_score(low_quality_results)
        
        self.assertLess(score, 60)
        self.assertGreaterEqual(score, 0)

    def test_calculate_quality_score_with_weights(self):
        """测试带权重的质量评分计算"""
        results_with_skewness = {
            "quality_ratio": 0.9,
            "checks": {
                "skewness": {"status": "NOK", "failed_count": 100, "total_checked": 1000},
            },
        }
        
        score = ResultAnalyzer.calculate_quality_score(results_with_skewness)
        
        # skewness有更高的权重，应该扣更多分
        self.assertLess(score, 90)


class TestReportUtilityFunctions(unittest.TestCase):
    """报告工具函数的单元测试"""

    def setUp(self):
        """测试前的准备工作"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.quality_results = {
            "total_elements": 1000,
            "bad_elements": 50,
            "quality_ratio": 0.95,
        }
        
        self.runner_stats = {"success": True}
        self.config_info = {"timeout": 300}

    def tearDown(self):
        """测试后的清理工作"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.batch.report.QualityReportGenerator')
    def test_generate_quality_report_function(self, mock_generator_class):
        """测试generate_quality_report便利函数"""
        mock_generator = mock_generator_class.return_value
        mock_generator.generate_quality_report.return_value = "test_report.txt"
        
        result = generate_quality_report(
            self.quality_results,
            self.runner_stats,
            self.config_info
        )
        
        mock_generator_class.assert_called_once()
        mock_generator.generate_quality_report.assert_called_once_with(
            self.quality_results,
            self.runner_stats,
            self.config_info,
            None,
            True
        )
        self.assertEqual(result, "test_report.txt")

    def test_analyze_quality_results_function(self):
        """测试analyze_quality_results便利函数"""
        quality_results = {
            "quality_ratio": 0.85,
            "checks": {
                "min_length": {
                    "status": "NOK",
                    "failed_count": 100,
                    "total_checked": 1000,
                },
            },
        }
        
        result = analyze_quality_results(quality_results)
        
        self.assertIn("quality_score", result)
        self.assertIn("worst_performing_checks", result)
        self.assertIn("overall_status", result)
        
        # 质量评分应该在合理范围内
        self.assertGreaterEqual(result["quality_score"], 0)
        self.assertLessEqual(result["quality_score"], 100)
        
        # 状态应该是PASS或FAIL
        self.assertIn(result["overall_status"], ["PASS", "FAIL"])


if __name__ == "__main__":
    unittest.main()