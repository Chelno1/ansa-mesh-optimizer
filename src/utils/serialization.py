#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
序列化工具模块

包含JSON序列化、配置文件加载和保存等相关功能。

作者: Chel
创建日期: 2025-08-23
版本: 1.0.0
"""

import json
import logging
from typing import Any, Dict, Union
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def safe_json_serialize(obj: Any) -> str:
    """
    安全的JSON序列化，处理numpy类型和其他特殊类型
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        JSON字符串
        
    Examples:
        >>> data = {'array': np.array([1, 2, 3]), 'value': np.float64(3.14)}
        >>> json_str = safe_json_serialize(data)
        >>> print(json_str)
    """
    def convert_types(obj):
        if isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # 其他numpy类型
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'isoformat'):  # datetime对象
            return obj.isoformat()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        else:
            return obj
    
    try:
        converted_obj = convert_types(obj)
        return json.dumps(converted_obj, sort_keys=True, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        return json.dumps({"error": f"Serialization failed: {str(e)}"})


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True,
                      create_dir: bool = False) -> Path:
    """
    验证文件路径
    
    Args:
        file_path: 文件路径
        must_exist: 文件是否必须存在
        create_dir: 是否创建目录
        
    Returns:
        验证后的Path对象
        
    Raises:
        ValueError: 路径验证失败
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise ValueError(f"文件不存在: {path}")
    
    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def load_json_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    安全加载JSON配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        ValueError: 配置文件加载失败
        
    Examples:
        >>> config = load_json_config('config.json')
        >>> print(config)
    """
    try:
        config_path = validate_file_path(config_file, must_exist=True)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"成功加载配置文件: {config_path}")
        return dict(config)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {e}")
    except Exception as e:
        raise ValueError(f"配置文件加载失败: {e}")


def save_json_config(config: Dict[str, Any], 
                    config_file: Union[str, Path],
                    backup: bool = True) -> None:
    """
    安全保存JSON配置文件
    
    Args:
        config: 配置字典
        config_file: 配置文件路径
        backup: 是否备份现有文件
        
    Examples:
        >>> config = {'param1': 1.0, 'param2': 'value'}
        >>> save_json_config(config, 'config.json', backup=True)
    """
    config_path = Path(config_file)
    
    # 备份现有文件
    if backup and config_path.exists():
        backup_path = config_path.with_suffix(f"{config_path.suffix}.bak")
        config_path.rename(backup_path)
        logger.info(f"备份文件已创建: {backup_path}")
    
    # 保存新配置
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用安全的JSON序列化
        json_str = safe_json_serialize(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        logger.info(f"配置文件已保存: {config_path}")
        
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        raise


def parse_json_string(json_str: str) -> Dict[str, Any]:
    """
    解析JSON字符串
    
    Args:
        json_str: JSON字符串
        
    Returns:
        解析后的字典
        
    Raises:
        ValueError: JSON解析失败
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")


def serialize_numpy_object(obj: Any) -> Any:
    """
    序列化numpy对象为Python原生类型
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        序列化后的对象
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # 其他numpy类型
        return obj.item()
    else:
        return obj


if __name__ == "__main__":
    # 测试序列化功能
    print("=== Serialization Testing ===")
    
    # 测试numpy对象序列化
    test_data = {
        'array': np.array([1, 2, 3]),
        'float': np.float64(3.14),
        'int': np.int32(42),
        'nested': {
            'values': np.array([0.1, 0.2, 0.3])
        }
    }
    
    json_str = safe_json_serialize(test_data)
    print("Serialized JSON:")
    print(json_str)
    
    # 测试解析
    parsed = parse_json_string(json_str)
    print(f"Parsed data: {parsed}")
    
    print("Serialization testing completed!")