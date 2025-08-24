#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的Ansa批处理网格脚本 - 模块化重构版本

作者: Chel
创建日期: 2025-06-19
版本: 2.0.0
更新日期: 2025-08-24
重构: 模块化拆分，提高可维护性和可测试性
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# 设置脚本目录
script_dir = Path(__file__).parent.resolve()
cwd_dir = Path.cwd().resolve()

# 独立的日志配置 - 不依赖其他模块
def setup_independent_logging(log_level=logging.INFO, log_dir=None):
    """
    为batch_mesh.py设置独立的日志配置
    
    Args:
        log_level: 日志级别
        log_dir: 日志目录
    """
    if log_dir is None:
        log_dir = cwd_dir
    
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    log_file = log_dir / "batch_mesh.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

# 配置独立日志
setup_independent_logging(log_level=logging.INFO, log_dir=cwd_dir)
logger = logging.getLogger(__name__)

# 导入新的模块化组件
try:
    from .batch import (
        AnsaBatchConfig,
        AnsaBatchMeshRunner,
        QualityReportGenerator,
        create_default_config,
        load_config_from_file,
        analyze_quality_results,
    )
except ImportError:
    # 如果作为独立脚本运行，尝试直接导入
    sys.path.insert(0, str(script_dir))
    from batch import (
        AnsaBatchConfig,
        AnsaBatchMeshRunner,
        QualityReportGenerator,
        create_default_config,
        load_config_from_file,
        analyze_quality_results,
    )

# 为了向后兼容，重新导出类
AnsaBatchConfig = AnsaBatchConfig
AnsaBatchMeshRunner = AnsaBatchMeshRunner


def main() -> int:
    """主函数 - 增强版本"""
    try:
        logger.info("开始Ansa批处理网格操作")

        # 创建批处理运行器
        runner = AnsaBatchMeshRunner(
            script_dir=script_dir,
            cwd_dir=cwd_dir
        )

        # 运行批处理网格
        mesh_success = runner.run_batch_mesh()

        if not mesh_success:
            logger.error("网格生成失败")
            return 1

        # 检查质量
        quality_results = runner.check_element_quality()

        # 输出不合格网格数量（用于优化器读取）
        bad_elements = quality_results.get("bad_elements", 0)
        print(f"bad elements: {bad_elements}")

        # 生成质量报告
        report_generator = QualityReportGenerator(output_dir=cwd_dir)
        runner_stats = runner.get_stats()
        config_info = runner.config.get_config_summary()
        
        report_generator.generate_quality_report(
            quality_results, 
            runner_stats, 
            config_info
        )

        # 保存模型
        save_success = runner.save_model()

        if not save_success:
            logger.warning("模型保存失败，但继续执行")

        # 输出统计信息
        stats = runner.get_stats()
        logger.info(f"运行统计: {stats}")

        # 分析质量结果
        analysis = analyze_quality_results(quality_results)
        logger.info(f"质量分析: 评分={analysis['quality_score']:.1f}, 状态={analysis['overall_status']}")

        logger.info("Ansa批处理网格操作完成")

        # 返回退出码（0表示成功，1表示有不合格网格）
        return 1 if bad_elements > 0 else 0

    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 130
    except Exception as e:
        logger.error(f"批处理操作异常: {e}")
        logger.debug(traceback.format_exc())
        return 2  # 异常退出码


def run_quality_check_only(
    min_len: float = 2.0, max_len: float = 8.0, output_report: bool = True
) -> Dict[str, Any]:
    """
    仅运行质量检查（不生成网格）

    Args:
        min_len: 最小长度阈值
        max_len: 最大长度阈值
        output_report: 是否生成报告

    Returns:
        质量检查结果
    """
    runner = AnsaBatchMeshRunner(
        script_dir=script_dir,
        cwd_dir=cwd_dir
    )

    custom_thresholds = {"min_element_length": min_len, "max_element_length": max_len}

    results = runner.check_element_quality(custom_thresholds)

    if output_report:
        report_generator = QualityReportGenerator(output_dir=cwd_dir)
        runner_stats = runner.get_stats()
        config_info = runner.config.get_config_summary()
        
        report_generator.generate_quality_report(
            results, 
            runner_stats, 
            config_info
        )

    return results


def batch_mesh_with_params(params: Dict[str, float]) -> int:
    """
    使用指定参数运行批处理网格

    Args:
        params: 网格参数字典

    Returns:
        不合格网格单元数量
    """
    try:
        # 创建运行器
        runner = AnsaBatchMeshRunner(
            script_dir=script_dir,
            cwd_dir=cwd_dir
        )

        # 运行网格生成
        mesh_success = runner.run_batch_mesh(params)

        if not mesh_success:
            logger.error("网格生成失败")
            return 99999  # 返回大数值表示失败

        # 检查质量
        quality_results = runner.check_element_quality()
        bad_elements = int(quality_results.get("bad_elements", 99999))

        logger.info(f"网格参数: {params}")
        logger.info(f"不合格网格数量: {bad_elements}")

        return bad_elements

    except Exception as e:
        logger.error(f"批处理网格异常: {e}")
        return 99999


# 向后兼容函数
def check_shell_min_length(min_len: float) -> str:
    """
    检查壳单元最小尺寸（向后兼容）

    Args:
        min_len: 最小长度阈值

    Returns:
        检查状态 ('OK' 或 'NOK')
    """
    runner = AnsaBatchMeshRunner(
        script_dir=script_dir,
        cwd_dir=cwd_dir
    )
    result = runner._check_shell_quality(min_len, "min_length")

    # 输出不合格单元数（与原代码兼容）
    print(f'bad elements: {result["failed_count"]}')

    return str(result["status"])


def check_shell_max_length(max_len: float) -> str:
    """
    检查壳单元最大尺寸（向后兼容）

    Args:
        max_len: 最大长度阈值

    Returns:
        检查状态 ('OK' 或 'NOK')
    """
    runner = AnsaBatchMeshRunner(
        script_dir=script_dir,
        cwd_dir=cwd_dir
    )
    result = runner._check_shell_quality(max_len, "max_length")

    # 输出不合格单元数（与原代码兼容）
    print(f'bad elements: {result["failed_count"]}')

    return str(result["status"])


def run_batch_mesh() -> int:
    """
    运行批处理网格（向后兼容）

    Returns:
        成功返回1，失败返回0
    """
    runner = AnsaBatchMeshRunner(
        script_dir=script_dir,
        cwd_dir=cwd_dir
    )
    success = runner.run_batch_mesh()
    return 1 if success else 0


# 新增的便利函数
def create_batch_runner(
    config_file: Optional[Path] = None,
    custom_config = None
):
    """
    创建批处理运行器的便利函数

    Args:
        config_file: 配置文件路径
        custom_config: 自定义配置

    Returns:
        批处理运行器实例
    """
    if custom_config:
        config = custom_config
    elif config_file and config_file.exists():
        config = load_config_from_file(config_file, cwd_dir)
    else:
        config = create_default_config(cwd_dir)
        if config_file:
            config.load_from_file(config_file)

    return AnsaBatchMeshRunner(
        script_dir=script_dir,
        cwd_dir=cwd_dir,
        config=config
    )


def run_full_batch_analysis(
    params: Optional[Dict[str, float]] = None,
    custom_thresholds: Optional[Dict[str, float]] = None,
    output_formats: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    运行完整的批处理分析，包括网格生成、质量检查和报告生成

    Args:
        params: 网格参数
        custom_thresholds: 自定义质量阈值
        output_formats: 输出格式列表 ['txt', 'csv', 'json']

    Returns:
        完整的分析结果
    """
    if output_formats is None:
        output_formats = ['txt']

    try:
        # 创建运行器
        runner = create_batch_runner()

        # 运行网格生成
        mesh_success = runner.run_batch_mesh(params)
        
        # 检查质量
        quality_results = runner.check_element_quality(custom_thresholds)
        
        # 获取统计信息
        runner_stats = runner.get_stats()
        config_info = runner.config.get_config_summary()
        
        # 生成不同格式的报告
        report_generator = QualityReportGenerator(output_dir=cwd_dir)
        report_files = {}
        
        for format_type in output_formats:
            if format_type == 'txt':
                report_files['txt'] = report_generator.generate_quality_report(
                    quality_results, runner_stats, config_info
                )
            elif format_type == 'csv':
                report_files['csv'] = report_generator.generate_csv_report(quality_results)
            elif format_type == 'json':
                report_files['json'] = report_generator.generate_json_report(
                    quality_results, runner_stats, config_info
                )
        
        # 分析结果
        analysis = analyze_quality_results(quality_results)
        
        # 保存模型
        model_saved = runner.save_model()
        
        return {
            "success": mesh_success,
            "quality_results": quality_results,
            "analysis": analysis,
            "runner_stats": runner_stats,
            "config_info": config_info,
            "report_files": report_files,
            "model_saved": model_saved,
        }
        
    except Exception as e:
        logger.error(f"完整批处理分析异常: {e}")
        return {
            "success": False,
            "error": str(e),
            "quality_results": {},
            "analysis": {},
        }


if __name__ == "__main__":
    # 如果直接运行此脚本，执行主函数
    exit_code = main()
    sys.exit(exit_code)
