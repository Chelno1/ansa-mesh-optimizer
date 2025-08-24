#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试运行器 - 执行所有测试并生成报告

作者: Chel
创建日期: 2025-07-07
"""

import logging
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict

# 项目根目录（用于路径引用）
project_root = Path(__file__).parent.parent


def setup_test_logging():
    """设置测试日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("tests/test_results.log"),
            logging.StreamHandler(),
        ],
    )


def discover_and_run_tests() -> Dict[str, Any]:
    """发现并运行所有测试"""
    logger = logging.getLogger(__name__)

    # 测试发现
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests", pattern="test_*.py")

    # 运行测试
    logger.info("开始运行测试套件...")
    start_time = time.time()

    # 使用详细的测试运行器
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)

    result = runner.run(test_suite)

    end_time = time.time()
    execution_time = end_time - start_time

    # 收集结果
    test_results = {
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, "skipped") else 0,
        "success_rate": (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
            if result.testsRun > 0
            else 0
        ),
        "execution_time": execution_time,
        "failure_details": result.failures,
        "error_details": result.errors,
    }

    return test_results


def run_specific_test_modules() -> Dict[str, Dict[str, Any]]:
    """运行特定的测试模块"""
    logger = logging.getLogger(__name__)

    test_modules = [
        "tests.unit.test_config",
        "tests.unit.test_mesh_optimizer",
        "tests.unit.test_mesh_evaluator",
        "tests.integration.test_cli_integration",
        "tests.integration.test_batch_mesh_integration",
    ]

    results = {}

    for module_name in test_modules:
        logger.info(f"运行测试模块: {module_name}")

        try:
            # 导入并运行测试模块
            test_loader = unittest.TestLoader()
            test_suite = test_loader.loadTestsFromName(module_name)

            runner = unittest.TextTestRunner(
                verbosity=1, stream=sys.stdout, buffer=True
            )

            start_time = time.time()
            result = runner.run(test_suite)
            end_time = time.time()

            results[module_name] = {
                "total_tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "execution_time": end_time - start_time,
                "success": result.wasSuccessful(),
            }

        except Exception as e:
            logger.error(f"运行测试模块 {module_name} 时出错: {e}")
            results[module_name] = {"error": str(e), "success": False}

    return results


def generate_test_report(
    results: Dict[str, Any], module_results: Dict[str, Dict[str, Any]]
):
    """生成测试报告"""
    logger = logging.getLogger(__name__)

    report_file = Path("tests/test_report.txt")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("ANSA网格优化器测试报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 总体结果
        f.write("总体测试结果:\n")
        f.write("-" * 20 + "\n")
        f.write(f"总测试数: {results['total_tests']}\n")
        f.write(f"失败数: {results['failures']}\n")
        f.write(f"错误数: {results['errors']}\n")
        f.write(f"跳过数: {results['skipped']}\n")
        f.write(f"成功率: {results['success_rate']:.2%}\n")
        f.write(f"执行时间: {results['execution_time']:.2f}秒\n\n")

        # 模块级结果
        f.write("模块级测试结果:\n")
        f.write("-" * 30 + "\n")
        for module, module_result in module_results.items():
            f.write(f"\n{module}:\n")
            if "error" in module_result:
                f.write(f"  错误: {module_result['error']}\n")
            else:
                f.write(f"  测试数: {module_result['total_tests']}\n")
                f.write(f"  失败数: {module_result['failures']}\n")
                f.write(f"  错误数: {module_result['errors']}\n")
                f.write(f"  执行时间: {module_result['execution_time']:.2f}秒\n")
                f.write(f"  状态: {'通过' if module_result['success'] else '失败'}\n")

        # 失败详情
        if results["failure_details"]:
            f.write("\n\n失败详情:\n")
            f.write("-" * 20 + "\n")
            for i, (test, traceback) in enumerate(results["failure_details"], 1):
                f.write(f"\n{i}. {test}:\n")
                f.write(f"{traceback}\n")

        # 错误详情
        if results["error_details"]:
            f.write("\n\n错误详情:\n")
            f.write("-" * 20 + "\n")
            for i, (test, traceback) in enumerate(results["error_details"], 1):
                f.write(f"\n{i}. {test}:\n")
                f.write(f"{traceback}\n")

    logger.info(f"测试报告已生成: {report_file}")


def main():
    """主函数"""
    setup_test_logging()
    logger = logging.getLogger(__name__)

    logger.info("开始ANSA网格优化器测试")

    try:
        # 运行所有测试
        logger.info("运行完整测试套件...")
        overall_results = discover_and_run_tests()

        # 运行模块级测试
        logger.info("运行模块级测试...")
        module_results = run_specific_test_modules()

        # 生成报告
        generate_test_report(overall_results, module_results)

        # 输出摘要
        logger.info("测试完成!")
        logger.info(f"总测试数: {overall_results['total_tests']}")
        logger.info(f"成功率: {overall_results['success_rate']:.2%}")
        logger.info(f"执行时间: {overall_results['execution_time']:.2f}秒")

        # 返回适当的退出码
        if overall_results["failures"] > 0 or overall_results["errors"] > 0:
            logger.warning("存在测试失败或错误")
            return 1
        else:
            logger.info("所有测试通过")
            return 0

    except Exception as e:
        logger.error(f"测试运行过程中发生异常: {e}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
