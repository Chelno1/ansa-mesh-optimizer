#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估器环境验证模块

此模块包含从 mesh_evaluator.py 中分离出的环境检查逻辑，
负责验证 Ansa 环境的可用性和配置。

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def validate_ansa_environment(ansa_executable: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    验证Ansa环境
    
    Args:
        ansa_executable: Ansa可执行文件路径
        timeout: 验证超时时间（秒）
        
    Returns:
        (是否验证成功, 错误消息)
    """
    try:
        # 检查Ansa可执行文件
        result = subprocess.run(
            [ansa_executable, '--version'],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            error_msg = "Ansa可执行文件无法运行"
            logger.warning(error_msg)
            return False, error_msg
        
        logger.info("Ansa环境验证成功")
        return True, None
        
    except subprocess.TimeoutExpired:
        error_msg = f"Ansa版本检查超时（{timeout}秒）"
        logger.warning(error_msg)
        return False, error_msg
    except FileNotFoundError:
        error_msg = f"Ansa可执行文件未找到: {ansa_executable}"
        logger.warning(error_msg)
        return False, error_msg
    except RuntimeError as e:
        error_msg = f"Ansa环境验证失败: {e}"
        logger.warning(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Ansa环境验证异常: {e}"
        logger.warning(error_msg)
        return False, error_msg


def check_input_files(input_model_path: str, batch_script_path: str) -> Tuple[bool, str]:
    """
    检查输入文件是否存在
    
    Args:
        input_model_path: 输入模型文件路径
        batch_script_path: 批处理脚本路径
        
    Returns:
        (是否验证成功, 错误消息)
    """
    # 验证输入模型文件存在
    input_model = Path(input_model_path)
    if not input_model.exists():
        error_msg = f"输入模型文件不存在: {input_model_path}"
        logger.error(error_msg)
        return False, error_msg
    
    # 验证批处理脚本存在
    batch_script = Path(batch_script_path)
    if not batch_script.exists():
        error_msg = f"批处理脚本不存在: {batch_script_path}"
        logger.error(error_msg)
        return False, error_msg
    
    logger.info("输入文件验证成功")
    return True, ""


def run_ansa_batch(ansa_command: list, temp_dir: str, timeout: int) -> subprocess.CompletedProcess:
    """
    运行Ansa批处理命令
    
    Args:
        ansa_command: Ansa命令列表
        temp_dir: 工作目录
        timeout: 执行超时时间
        
    Returns:
        subprocess.CompletedProcess对象
        
    Raises:
        subprocess.TimeoutExpired: 命令超时
        FileNotFoundError: 可执行文件未找到
        PermissionError: 权限错误
        Exception: 其他执行错误
    """
    logger.info(f"执行Ansa命令: {' '.join(ansa_command)}")
    
    # 执行命令
    result = subprocess.run(
        ansa_command,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=temp_dir
    )
    
    return result


def handle_ansa_returncode(result: subprocess.CompletedProcess) -> Tuple[bool, Optional[str]]:
    """
    处理Ansa返回代码
    
    Args:
        result: subprocess运行结果
        
    Returns:
        (是否成功, 错误消息)
    """
    if result.returncode == 0:
        return True, None
    elif result.returncode == 1:
        logger.warning("Ansa返回代码1 - 可能有警告但继续执行")
        return True, None  # 返回代码1通常表示有警告但可以继续
    elif result.returncode == 2:
        error_msg = f"Ansa返回代码2 - 致命错误: {result.stderr}"
        logger.error(error_msg)
        return False, error_msg
    else:
        error_msg = f"Ansa执行失败，返回代码: {result.returncode}, 错误输出: {result.stderr}"
        logger.error(error_msg)
        return False, error_msg


class AnsaEnvironmentValidator:
    """Ansa环境验证器类"""
    
    def __init__(self, ansa_config):
        """
        初始化环境验证器
        
        Args:
            ansa_config: Ansa配置对象
        """
        self.config = ansa_config
        self.is_valid = False
        self.error_message = None
        
    def validate(self) -> bool:
        """
        执行完整的环境验证
        
        Returns:
            验证是否成功
        """
        # 验证Ansa可执行文件
        is_valid, error_msg = validate_ansa_environment(
            self.config.ansa_executable,
            timeout=getattr(self.config, 'validation_timeout', 10)
        )
        
        if not is_valid:
            self.is_valid = False
            self.error_message = error_msg
            return False
        
        # 验证输入文件
        input_model_path = getattr(self.config, 'input_model', '')
        batch_script_path = getattr(self.config, 'script_dir', Path()) / getattr(self.config, 'batch_script', '')
        
        is_valid, error_msg = check_input_files(
            str(input_model_path),
            str(batch_script_path)
        )
        
        if not is_valid:
            self.is_valid = False
            self.error_message = error_msg
            return False
        
        self.is_valid = True
        self.error_message = None
        logger.info("Ansa环境验证完成")
        return True
    
    def get_validation_result(self) -> Tuple[bool, Optional[str]]:
        """
        获取验证结果
        
        Returns:
            (是否验证成功, 错误消息)
        """
        return self.is_valid, self.error_message