#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估器 I/O 工具模块

此模块包含临时文件操作、ANSA 输出解析等 I/O 相关功能。
从 utils.py 中分离出来，专注于文件和输出处理。

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import os
import re
import json
import shutil
import tempfile
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def create_timestamped_temp_dir() -> str:
    """
    创建带时间戳的临时文件夹
    
    Returns:
        临时文件夹路径
    """
    try:
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒精度
        
        # 创建临时目录名称
        temp_dir_name = f"ansa_mesh_eval_{timestamp}"
        temp_dir_path = os.path.join(os.getcwd(), temp_dir_name)
        
        # 创建目录
        os.makedirs(temp_dir_path, exist_ok=True)
        
        logger.info(f"创建临时文件夹: {temp_dir_path}")
        return temp_dir_path
        
    except Exception as e:
        logger.error(f"创建临时文件夹失败: {e}")
        # 如果创建失败，回退到当前目录
        return os.getcwd()


def copy_mpar_files_to_temp_dir(temp_dir: str, criterion_dir: Path, mpar_file_pattern: str) -> str:
    """
    将*.ansa_mpar文件拷贝到临时文件夹
    
    Args:
        temp_dir: 临时文件夹路径
        criterion_dir: criterion目录路径
        mpar_file_pattern: mpar文件匹配模式
        
    Returns:
        拷贝后的文件路径
    """
    try:
        # 查找mpar文件
        mpar_files = list(Path(criterion_dir).glob(mpar_file_pattern))
        
        if not mpar_files:
            logger.warning("未找到mpar文件，跳过文件拷贝")
            return ""
        
        # 只取第一个mpar文件
        mpar_file = mpar_files[0]
        
        # 构建目标文件路径
        dest_file = os.path.join(temp_dir, mpar_file.name)
        
        # 拷贝文件
        shutil.copy2(str(mpar_file), dest_file)
        
        logger.info(f"拷贝mpar文件: {mpar_file} -> {dest_file}")
        return dest_file
        
    except Exception as e:
        logger.error(f"拷贝mpar文件失败: {e}")
        return ""


def create_temp_config_in_dir(temp_dir: str, params: Dict[str, float], format_value_func) -> str:
    """
    在指定目录中创建临时配置文件
    
    Args:
        temp_dir: 临时目录路径
        params: 参数字典
        format_value_func: 参数值格式化函数
        
    Returns:
        配置文件路径
    """
    try:
        # 在临时目录中创建配置文件
        config_file_path = os.path.join(temp_dir, "mesh_config.json")
        
        # 创建JSON格式的配置数据
        config_data = {}
        for key, value in params.items():
            formatted_value = format_value_func(key, value)
            config_data[key] = formatted_value
        
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"在临时目录创建配置文件: {config_file_path}")
        return config_file_path
        
    except Exception as e:
        logger.error(f"在临时目录创建配置文件失败: {e}")
        raise


def create_temp_config(params: Dict[str, float], format_value_func) -> str:
    """
    创建临时配置文件
    
    Args:
        params: 参数字典
        format_value_func: 参数值格式化函数
        
    Returns:
        临时文件路径
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for key, value in params.items():
                formatted_value = format_value_func(key, value)
                f.write(f"{key} = {formatted_value}\n")
            temp_file = f.name
        
        logger.debug(f"创建临时配置文件: {temp_file}")
        return temp_file
        
    except Exception as e:
        logger.error(f"创建临时配置文件失败: {e}")
        raise


def parse_ansa_output(output: str) -> float:
    """
    解析Ansa输出 - 增强版本
    
    Args:
        output: Ansa程序输出
        
    Returns:
        解析出的不合格网格数量
    """
    try:
        # 查找不合格网格数量的多种模式
        patterns = [
            r'bad elements:\s*(\d+)',
            r'failed elements:\s*(\d+)',
            r'poor quality elements:\s*(\d+)',
            r'质量不合格元素:\s*(\d+)',
            r'不合格单元:\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                count = int(match.group(1))
                logger.info(f"找到不合格网格数量: {count}")
                return float(count)
        
        # 如果没有找到，尝试从最后几行提取数字
        lines = output.strip().split('\n')
        for line in reversed(lines[-10:]):  # 检查最后10行
            # 查找数字
            numbers = re.findall(r'\d+', line)
            if numbers:
                # 取最大的数字（通常是元素数量）
                max_number = max(int(n) for n in numbers)
                if max_number > 0:
                    logger.info(f"从输出行解析得到数字: {max_number}")
                    return float(max_number)
        
        logger.warning("无法从输出中解析不合格网格数量")
        logger.debug(f"Ansa输出: {output}")
        return 99999.0
        
    except Exception as e:
        logger.error(f"解析Ansa输出失败: {e}")
        return 99999.0


def cleanup_temp_files(files: List[Optional[str]]) -> None:
    """
    清理临时文件
    
    Args:
        files: 需要清理的文件路径列表
    """
    for file_path in files:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"删除临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    清理临时目录
    
    Args:
        temp_dir: 临时目录路径
    """
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"删除临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"删除临时目录失败: {e}")


def process_parameter_files_in_temp_dir(temp_dir: str, params: Dict[str, float], parameter_replacer) -> None:
    """
    在临时文件夹中处理参数文件替换
    
    Args:
        temp_dir: 临时目录路径
        params: 参数字典
        parameter_replacer: 参数替换管理器
    """
    try:
        # 在临时目录中查找mpar文件
        temp_mpar_files = list(Path(temp_dir).glob("*.ansa_mpar"))
        
        if not temp_mpar_files:
            logger.warning("临时目录中未找到mpar文件，跳过参数文件处理")
            return
        
        # 只处理第一个mpar文件
        mpar_file = temp_mpar_files[0]
        original_mpar_path = str(mpar_file)
        logger.info(f"处理临时目录中的mpar文件: {mpar_file}")
        
        # 记录处理前临时目录中的所有文件，以便后续清理
        temp_dir_path = Path(temp_dir)
        files_before = set(temp_dir_path.iterdir())
        
        # 使用参数替换管理器处理参数
        updated_file_path = parameter_replacer.process_parameter_replacements(original_mpar_path, params)
        
        # 如果创建了新文件，需要将内容复制回原文件并清理
        if updated_file_path != original_mpar_path:
            # 读取更新后的文件内容
            with open(updated_file_path, 'r', encoding='utf-8') as f:
                updated_content = f.read()
            
            # 写入原文件
            with open(mpar_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            # 清理所有在处理过程中创建的临时文件
            files_after = set(temp_dir_path.iterdir())
            temp_files_created = files_after - files_before
            
            cleaned_count = 0
            for temp_file in temp_files_created:
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"已清理临时文件: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"清理临时文件失败 {temp_file}: {cleanup_error}")
            
            # 确保清理最终的更新文件（如果它不在上面的集合中）
            if Path(updated_file_path).exists() and Path(updated_file_path) not in temp_files_created:
                try:
                    Path(updated_file_path).unlink()
                    cleaned_count += 1
                    logger.debug(f"已清理最终更新文件: {updated_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"清理最终更新文件失败 {updated_file_path}: {cleanup_error}")
            
            logger.info(f"已将更新内容复制回临时目录文件: {mpar_file}，并清理了 {cleaned_count} 个临时文件")
            
    except Exception as e:
        logger.error(f"在临时目录处理参数文件失败: {e}")


def simulate_evaluation(base_range: tuple = (50, 500)) -> float:
    """
    模拟评估（用于测试和备用）
    
    Args:
        base_range: 基础分数范围
        
    Returns:
        模拟的评估结果
    """
    import random
    # 基于参数生成模拟结果
    base_score = random.uniform(*base_range)
    logger.info(f"使用模拟评估，返回结果: {base_score}")
    return base_score