#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时文件管理模块
负责处理网格评估器的临时目录和文件管理

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .io_utils import (
    cleanup_temp_files,
    copy_mpar_files_to_temp_dir,
    create_temp_config_in_dir,
    create_timestamped_temp_dir,
    process_parameter_files_in_temp_dir,
)
from .parameter_replacement_strategies import format_mpar_parameter_value

logger = logging.getLogger(__name__)


class TempFileManager:
    """临时文件管理器 - 统一管理临时目录和文件的生命周期"""

    def __init__(self, config_manager, parameter_replacer):
        """
        初始化临时文件管理器
        
        Args:
            config_manager: 配置管理器实例
            parameter_replacer: 参数替换管理器实例
        """
        if config_manager is None:
            raise ValueError("TempFileManager requires a config_manager instance")
        if parameter_replacer is None:
            raise ValueError("TempFileManager requires a parameter_replacer instance")
            
        self.config_manager = config_manager
        self.config = config_manager.ansa_config
        self.parameter_replacer = parameter_replacer
        self.cwd_dir = Path.cwd().resolve()
        
        # 直接使用配置的 criteria_dir 绝对路径
        self.criterion_dir = Path(self.config.criteria_dir)
        
        # 临时文件跟踪
        self.temp_dir: Optional[str] = None
        self.temp_files: List[Optional[str]] = []

    def setup_temp_environment(self, params: Dict[str, float]) -> str:
        """
        设置临时环境
        
        Args:
            params: 清理后的参数字典
            
        Returns:
            临时目录路径
            
        Raises:
            Exception: 设置失败时抛出
        """
        try:
            # 创建带时间戳的临时文件夹
            self.temp_dir = create_timestamped_temp_dir()
            logger.info(f"创建临时目录: {self.temp_dir}")

            # 将*.ansa_mpar文件拷贝到临时文件夹
            copied_mpar_files = copy_mpar_files_to_temp_dir(
                self.temp_dir, self.criterion_dir, self.config.mpar_file_pattern
            )
            logger.debug(f"拷贝MPAR文件: {len(copied_mpar_files)} 个文件")

            # 在临时文件夹中创建临时配置文件
            config_file = create_temp_config_in_dir(
                self.temp_dir, params, format_mpar_parameter_value, str(self.criterion_dir)
            )
            self.temp_files.append(config_file)
            logger.debug(f"创建临时配置文件: {config_file}")

            # 处理mpar参数文件替换（在临时文件夹中）
            process_parameter_files_in_temp_dir(
                self.temp_dir, params, self.parameter_replacer
            )
            logger.debug("完成参数文件替换处理")

            return self.temp_dir
            
        except Exception as e:
            logger.error(f"设置临时环境失败: {e}")
            # 如果设置失败，清理已创建的文件
            self.cleanup()
            raise

    def get_temp_dir(self) -> Optional[str]:
        """获取当前临时目录路径"""
        return self.temp_dir

    def add_temp_file(self, file_path: Optional[str]) -> None:
        """
        添加临时文件到跟踪列表
        
        Args:
            file_path: 临时文件路径
        """
        if file_path is not None:
            self.temp_files.append(file_path)

    def cleanup(self) -> None:
        """清理临时文件（但保留临时目录和JSON文件用于调试）"""
        try:
            # 清理临时文件，但保留JSON文件
            if self.temp_files:
                # 过滤掉JSON文件，只清理其他类型的临时文件
                files_to_cleanup = []
                json_files_kept = []
                
                for file_path in self.temp_files:
                    if file_path and file_path.endswith('.json'):
                        json_files_kept.append(file_path)
                        logger.debug(f"保留JSON文件: {file_path}")
                    else:
                        files_to_cleanup.append(file_path)
                
                if files_to_cleanup:
                    cleanup_temp_files(files_to_cleanup)
                    logger.debug(f"清理了 {len(files_to_cleanup)} 个临时文件")
                
                if json_files_kept:
                    logger.debug(f"保留了 {len(json_files_kept)} 个JSON文件用于调试")
                
                # 只清除已处理的非JSON文件，保留JSON文件在列表中
                self.temp_files = json_files_kept
            
            # 注意：我们不清理临时目录，以便进行调试
            # 这与原始代码行为保持一致
            if self.temp_dir:
                logger.debug(f"保留临时目录用于调试: {self.temp_dir}")
                
        except Exception as e:
            logger.warning(f"清理临时文件时发生错误: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口 - 自动清理"""
        self.cleanup()


def create_temp_file_manager(config_manager, parameter_replacer) -> TempFileManager:
    """
    创建临时文件管理器实例
    
    Args:
        config_manager: 配置管理器实例
        parameter_replacer: 参数替换管理器实例
        
    Returns:
        临时文件管理器实例
    """
    return TempFileManager(config_manager, parameter_replacer)