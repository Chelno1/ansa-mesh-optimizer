#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实用工具函数 - 改进版本

作者: Chel
创建日期: 2025-06-19
版本: 1.2.0
更新日期: 2025-06-20
修复: 参数处理，错误处理，性能监控
"""

import numpy as np
import json
import time
import logging
from typing import Any, Dict, List, Union, Tuple, Optional, Callable
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

def normalize_params(params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """
    标准化参数字典，将numpy类型转换为Python原生类型
    
    Args:
        params: 参数字典
        
    Returns:
        标准化后的参数字典
    """
    normalized = {}
    
    for key, value in params.items():
        try:
            if hasattr(value, 'item'):  # numpy标量类型
                normalized[key] = value.item()
            elif isinstance(value, (np.integer, np.floating)):
                normalized[key] = value.item()
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    normalized[key] = value.item()
                else:
                    normalized[key] = value.tolist()
            elif isinstance(value, (list, tuple)) and len(value) == 1:
                # 处理单元素序列
                normalized[key] = normalize_params({'temp': value[0]})['temp']
            else:
                normalized[key] = value
        except Exception as e:
            logger.warning(f"Failed to normalize parameter {key}={value}: {e}")
            normalized[key] = value
    
    return normalized

def safe_json_serialize(obj: Any) -> str:
    """
    安全的JSON序列化，处理numpy类型和其他特殊类型
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        JSON字符串
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

def validate_param_types(params: Dict[str, Any], param_space) -> Dict[str, Union[int, float]]:
    """
    验证并转换参数类型
    
    Args:
        params: 参数字典
        param_space: 参数空间定义
        
    Returns:
        验证后的参数字典
    """
    validated_params = {}
    
    try:
        param_types = param_space.get_param_types()
        param_names = param_space.get_param_names()
        bounds = param_space.get_bounds()
        
        for i, name in enumerate(param_names):
            if name in params:
                value = params[name]
                expected_type = param_types[i]
                low, high = bounds[i]
                
                # 转换numpy类型
                if hasattr(value, 'item'):
                    value = value.item()
                
                # 类型转换
                try:
                    if expected_type == int:
                        converted_value = int(round(float(value)))
                    elif expected_type == float:
                        converted_value = float(value)
                    else:
                        converted_value = value
                    
                    # 边界检查
                    if low <= converted_value <= high:
                        validated_params[name] = converted_value
                    else:
                        logger.warning(f"Parameter {name}={converted_value} out of bounds [{low}, {high}]")
                        # 截断到边界内
                        validated_params[name] = max(low, min(high, converted_value))
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"Type conversion failed for {name}={value}: {e}")
                    # 使用默认值（边界中点）
                    default_value = (low + high) / 2
                    if expected_type == int:
                        default_value = int(round(default_value))
                    validated_params[name] = default_value
                    
            else:
                logger.warning(f"参数 {name} 缺失")
                # 使用默认值
                low, high = bounds[i]
                default_value = (low + high) / 2
                if param_types[i] == int:
                    default_value = int(round(default_value))
                validated_params[name] = default_value
        
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        raise ValueError(f"Parameter validation error: {e}")
    
    return validated_params

def format_execution_time(seconds: float) -> str:
    """
    格式化执行时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 0:
        return "0秒"
    elif seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"

def create_summary_table(data: List[Dict[str, Any]], 
                        columns: List[str] = None,
                        max_width: int = 120) -> str:
    """
    创建简单的表格摘要
    
    Args:
        data: 数据列表
        columns: 要显示的列名
        max_width: 最大表格宽度
        
    Returns:
        格式化的表格字符串
    """
    if not data:
        return "无数据"
    
    if columns is None:
        columns = list(data[0].keys())
    
    # 计算列宽
    col_widths = {}
    for col in columns:
        col_name_width = len(str(col))
        max_value_width = max(len(str(row.get(col, ''))) for row in data)
        col_widths[col] = min(max(col_name_width, max_value_width), max_width // len(columns))
    
    # 创建表格
    lines = []
    
    # 标题行
    header = " | ".join(str(col).ljust(col_widths[col])[:col_widths[col]] for col in columns)
    lines.append(header)
    
    # 分隔线
    separator = " | ".join("-" * col_widths[col] for col in columns)
    lines.append(separator)
    
    # 数据行
    for row in data:
        data_line = " | ".join(
            truncate_string(str(row.get(col, '')), col_widths[col]).ljust(col_widths[col])
            for col in columns
        )
        lines.append(data_line)
    
    return "\n".join(lines)

def truncate_string(s: str, max_length: int = 50) -> str:
    """
    截断字符串
    
    Args:
        s: 原始字符串
        max_length: 最大长度
        
    Returns:
        截断后的字符串
    """
    if len(s) <= max_length:
        return s
    return s[:max_length-3] + "..."

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        
    Returns:
        除法结果
    """
    try:
        if abs(denominator) < 1e-10:  # 更精确的零检查
            return default
        return float(numerator) / float(denominator)
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def check_memory_usage() -> Dict[str, float]:
    """
    检查内存使用情况
    
    Returns:
        内存使用信息字典
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent(),       # 内存使用百分比
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    except ImportError:
        return {'error': 'psutil not available'}
    except Exception as e:
        return {'error': str(e)}

def setup_numpy_print_options():
    """设置numpy打印选项"""
    try:
        np.set_printoptions(
            precision=6,
            suppress=True,
            threshold=10,
            edgeitems=3,
            linewidth=120
        )
    except Exception as e:
        logger.warning(f"Failed to set numpy print options: {e}")

@contextmanager
def performance_monitor(operation_name: str, 
                       log_memory: bool = True,
                       log_level: int = logging.INFO):
    """
    性能监控上下文管理器
    
    Args:
        operation_name: 操作名称
        log_memory: 是否记录内存使用
        log_level: 日志级别
    """
    start_time = time.time()
    start_memory = check_memory_usage() if log_memory else None
    
    logger.log(log_level, f"开始 {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.log(log_level, f"{operation_name} 完成，耗时: {format_execution_time(execution_time)}")
        
        if log_memory and start_memory and 'error' not in start_memory:
            end_memory = check_memory_usage()
            if 'error' not in end_memory:
                memory_delta = end_memory['rss_mb'] - start_memory['rss_mb']
                logger.log(log_level, f"内存变化: {memory_delta:+.1f}MB")

def create_progress_callback(total_iterations: int, 
                           verbose: bool = True,
                           update_interval: int = 1) -> Callable:
    """
    创建进度回调函数
    
    Args:
        total_iterations: 总迭代次数
        verbose: 是否显示详细信息
        update_interval: 更新间隔
        
    Returns:
        进度回调函数
    """
    last_update = 0
    start_time = time.time()
    
    def progress_callback(iteration: int, 
                         current_best: float = None, 
                         message: str = None):
        nonlocal last_update
        
        if not verbose or (iteration - last_update) < update_interval:
            return
        
        last_update = iteration
        progress = (iteration / total_iterations) * 100
        elapsed_time = time.time() - start_time
        
        status_parts = [f"进度: {progress:.1f}% ({iteration}/{total_iterations})"]
        
        if current_best is not None:
            status_parts.append(f"当前最佳: {current_best:.6f}")
        
        if elapsed_time > 0:
            rate = iteration / elapsed_time
            eta = (total_iterations - iteration) / rate if rate > 0 else 0
            status_parts.append(f"ETA: {format_execution_time(eta)}")
        
        if message:
            status_parts.append(message)
        
        status = " | ".join(status_parts)
        print(f"\r{status:<120}", end="", flush=True)
        
        if iteration >= total_iterations:
            print()  # 换行
    
    return progress_callback

def retry_on_exception(max_retries: int = 3, 
                      delay: float = 1.0,
                      backoff_factor: float = 2.0,
                      exceptions: tuple = (Exception,)) -> Callable:
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始重试间隔
        backoff_factor: 退避因子
        exceptions: 要捕获的异常类型
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败: {e}")
                        raise
                    else:
                        logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        logger.info(f"等待 {current_delay:.1f} 秒后重试...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
        return wrapper
    return decorator

def estimate_completion_time(start_time: float, 
                           current_iteration: int, 
                           total_iterations: int) -> str:
    """
    估算完成时间
    
    Args:
        start_time: 开始时间戳
        current_iteration: 当前迭代次数
        total_iterations: 总迭代次数
        
    Returns:
        估算完成时间字符串
    """
    if current_iteration <= 0:
        return "估算中..."
    
    elapsed_time = time.time() - start_time
    avg_time_per_iteration = elapsed_time / current_iteration
    remaining_iterations = total_iterations - current_iteration
    estimated_remaining_time = avg_time_per_iteration * remaining_iterations
    
    return format_execution_time(estimated_remaining_time)

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
    """
    try:
        config_path = validate_file_path(config_file, must_exist=True)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"成功加载配置文件: {config_path}")
        return config
        
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

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    计算数值列表的统计信息
    
    Args:
        values: 数值列表
        
    Returns:
        统计信息字典
    """
    if not values:
        return {'error': 'No values provided'}
    
    try:
        values_array = np.array(values)
        
        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'q25': float(np.percentile(values_array, 25)),
            'q75': float(np.percentile(values_array, 75)),
            'range': float(np.max(values_array) - np.min(values_array))
        }
    except Exception as e:
        return {'error': str(e)}

def create_backup_filename(original_path: Union[str, Path], 
                          timestamp: bool = True) -> Path:
    """
    创建备份文件名
    
    Args:
        original_path: 原始文件路径
        timestamp: 是否添加时间戳
        
    Returns:
        备份文件路径
    """
    path = Path(original_path)
    
    if timestamp:
        from datetime import datetime
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        return path.with_name(f"{path.stem}_{timestamp_str}{path.suffix}")
    else:
        return path.with_name(f"{path.stem}_backup{path.suffix}")

def filter_dict_by_keys(data: Dict[str, Any], 
                       keys: List[str], 
                       include: bool = True) -> Dict[str, Any]:
    """
    根据键过滤字典
    
    Args:
        data: 原始字典
        keys: 键列表
        include: True为包含模式，False为排除模式
        
    Returns:
        过滤后的字典
    """
    if include:
        return {k: v for k, v in data.items() if k in keys}
    else:
        return {k: v for k, v in data.items() if k not in keys}

# 设置numpy打印选项
setup_numpy_print_options()

# 常用的正则表达式模式
PATTERNS = {
    'number': r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',
    'integer': r'[-+]?\d+',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    'filename': r'[^<>:"/\\|?*\x00-\x1f]+',
}

def extract_numbers_from_text(text: str, pattern: str = 'number') -> List[float]:
    """
    从文本中提取数字
    
    Args:
        text: 输入文本
        pattern: 使用的正则表达式模式
        
    Returns:
        提取的数字列表
    """
    import re
    
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    matches = re.findall(PATTERNS[pattern], text)
    
    try:
        if pattern == 'integer':
            return [float(int(match)) for match in matches]
        else:
            return [float(match) for match in matches]
    except ValueError as e:
        logger.warning(f"Number extraction failed: {e}")
        return []

if __name__ == "__main__":
    # 测试工具函数
    print("=== Utils Testing ===")
    
    # 测试参数标准化
    test_params = {
        'element_size': np.float64(1.5),
        'perimeter_length': np.array([2.0]),
        'normal_param': 3.0
    }
    
    normalized = normalize_params(test_params)
    print(f"Normalized params: {normalized}")
    
    # 测试性能监控
    with performance_monitor("Test operation"):
        time.sleep(0.1)
    
    # 测试进度回调
    progress_cb = create_progress_callback(10)
    for i in range(11):
        progress_cb(i, current_best=10-i)
        time.sleep(0.05)
    
    # 测试统计计算
    test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = calculate_statistics(test_values)
    print(f"Statistics: {stats}")
    
    print("Utils testing completed!")