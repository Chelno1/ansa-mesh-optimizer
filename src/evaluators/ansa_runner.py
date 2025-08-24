#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANSA运行器模块
负责处理ANSA批处理的执行逻辑

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import logging
from pathlib import Path
from typing import List, Optional

from src.evaluators.environment import (
    check_input_files,
    handle_ansa_returncode,
    run_ansa_batch,
)
from src.evaluators.io_utils import parse_ansa_output, simulate_evaluation

logger = logging.getLogger(__name__)


class AnsaRunner:
    """ANSA运行器 - 负责执行ANSA批处理命令"""

    def __init__(self, config_manager):
        """
        初始化ANSA运行器
        
        Args:
            config_manager: 配置管理器实例
        """
        if config_manager is None:
            raise ValueError("AnsaRunner requires a config_manager instance")
            
        self.config_manager = config_manager
        self.config = config_manager.ansa_config
        self.cwd_dir = Path.cwd().resolve()

    def run_ansa_batch(self, temp_dir: str) -> float:
        """
        运行ANSA批处理
        
        Args:
            temp_dir: 临时目录路径
            
        Returns:
            不合格网格单元数量
        """
        try:
            # 构建ANSA命令
            ansa_command = self._build_ansa_command(temp_dir)
            logger.debug(f"ANSA命令: {' '.join(ansa_command)}")

            # 检查输入文件
            if not self._validate_input_files():
                logger.error("输入文件验证失败，使用模拟评估")
                return simulate_evaluation()

            # 执行命令
            result = run_ansa_batch(
                ansa_command, temp_dir, self.config.execution_timeout
            )
            logger.debug(f"ANSA执行结果: 返回码={result.returncode}")

            # 处理返回代码
            is_success, error_msg = handle_ansa_returncode(result)
            if not is_success:
                logger.error(f"ANSA执行失败: {error_msg}")
                return simulate_evaluation()

            # 解析成功的输出
            bad_elements_count = parse_ansa_output(result.stdout)
            logger.info(f"ANSA批处理完成，不合格单元数: {bad_elements_count}")
            return bad_elements_count

        except Exception as e:
            logger.error(f"ANSA批处理执行失败: {e}")
            return simulate_evaluation()

    def _build_ansa_command(self, temp_dir: str) -> List[str]:
        """
        构建ANSA命令
        
        Args:
            temp_dir: 临时目录路径
            
        Returns:
            ANSA命令列表
        """
        return [
            self.config.ansa_executable,
            "-b",
            "-execpy",
            f"load_script: '{self.config.script_dir / self.config.batch_script}'",
            "-i",
            f"{self.cwd_dir / self.config.input_model}",
            "-changedir",
            temp_dir,
        ]

    def _validate_input_files(self) -> bool:
        """
        验证输入文件
        
        Returns:
            验证是否成功
        """
        input_model_path = str(self.cwd_dir / self.config.input_model)
        batch_script_path = str(self.config.script_dir / self.config.batch_script)

        is_valid, error_msg = check_input_files(input_model_path, batch_script_path)
        if not is_valid:
            logger.error(f"输入文件验证失败: {error_msg}")
            return False
            
        return True

    def get_input_model_path(self) -> str:
        """获取输入模型文件路径"""
        return str(self.cwd_dir / self.config.input_model)

    def get_batch_script_path(self) -> str:
        """获取批处理脚本路径"""
        return str(self.config.script_dir / self.config.batch_script)

    def get_ansa_executable(self) -> str:
        """获取ANSA可执行文件路径"""
        return self.config.ansa_executable

    def get_execution_timeout(self) -> Optional[int]:
        """获取执行超时时间"""
        return self.config.execution_timeout


def create_ansa_runner(config_manager) -> AnsaRunner:
    """
    创建ANSA运行器实例
    
    Args:
        config_manager: 配置管理器实例
        
    Returns:
        ANSA运行器实例
    """
    return AnsaRunner(config_manager)