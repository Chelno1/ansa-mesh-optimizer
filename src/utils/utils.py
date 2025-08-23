#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具函数模块 - 重构版本

包含参数验证、性能监控、数学计算等通用工具函数。
序列化和格式化功能已分离到专门模块。

作者: Chel
创建日期: 2025-06-19
版本: 2.0.0
更新日期: 2025-08-23
重构: 将序列化和格式化功能分离，遵循单一职责原则
"""

import numpy as np
import time
import logging
from typing import Any, Dict, List, Union, Tuple, Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 导入统一的参数验证功能
from .parameter_validator import normalize_params as _normalize_params
from .parameter_validator import validate_param_types as _validate_param_types


def normalize_params(params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """
    标准化参数字典，将numpy类型转换为Python原生类型
    使用统一的参数验证器
    
    Args:
        params: 参数字典
        
    Returns:
        标准化后的参数字典
        
    Examples:
        >>> params = {'quality_threshold': np.array([0.6]), 'value': np.float64(3.14)}
        >>> normalized = normalize_params(params)
        >>> print(normalized)
    """
    return _normalize_params(params)


def validate_param_types(params: Dict[str, Any], param_space) -> Dict[str, Union[int, float]]:
    """
    验证并转换参数类型
    使用统一的参数验证器
    
    Args:
        params: 参数字典
        param_space: 参数空间定义
        
    Returns:
        验证后的参数字典
        
    Examples:
        >>> validated = validate_param_types(params, param_space_config)
        >>> print(validated)
    """
    return _validate_param_types(params, param_space)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        
    Returns:
        除法结果
        
    Examples:
        >>> result = safe_divide(10, 2)  # 返回 5.0
        >>> result = safe_divide(10, 0)  # 返回 0.0
        >>> result = safe_divide(10, 0, -1)  # 返回 -1.0
    """
    try:
        if abs(denominator) < 1e-10:  # 更精确的零检查
            return default
        return float(numerator) / float(denominator)
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def check_memory_usage() -> Dict[str, Union[float, str]]:
    """
    检查当前进程的内存使用情况
    
    Returns:
        内存使用信息字典
        
    Examples:
        >>> memory_info = check_memory_usage()
        >>> print(f"RSS: {memory_info['rss_mb']:.1f}MB")
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
    """
    设置numpy打印选项，优化数组显示格式
    
    Examples:
        >>> setup_numpy_print_options()
        >>> print(np.array([1.23456789, 2.3456789]))  # 格式化输出
    """
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
    性能监控上下文管理器，自动记录操作耗时和内存变化
    
    Args:
        operation_name: 操作名称
        log_memory: 是否记录内存使用
        log_level: 日志级别
        
    Examples:
        >>> with performance_monitor("数据处理"):
        ...     # 执行一些操作
        ...     time.sleep(1)
    """
    start_time = time.time()
    start_memory = check_memory_usage() if log_memory else None
    
    logger.log(log_level, f"开始 {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 导入格式化函数
        from .formatting import format_execution_time
        
        logger.log(log_level, f"{operation_name} 完成，耗时: {format_execution_time(execution_time)}")
        
        if log_memory and start_memory and 'error' not in start_memory:
            end_memory = check_memory_usage()
            if 'error' not in end_memory:
                start_rss = start_memory.get('rss_mb', 0)
                end_rss = end_memory.get('rss_mb', 0)
                if isinstance(start_rss, (int, float)) and isinstance(end_rss, (int, float)):
                    memory_delta = end_rss - start_rss
                    logger.log(log_level, f"内存变化: {memory_delta:+.1f}MB")


def create_progress_callback(total_iterations: int, 
                           verbose: bool = True,
                           update_interval: int = 1) -> Callable:
    """
    创建进度回调函数，用于显示操作进度
    
    Args:
        total_iterations: 总迭代次数
        verbose: 是否显示详细信息
        update_interval: 更新间隔
        
    Returns:
        进度回调函数
        
    Examples:
        >>> progress_cb = create_progress_callback(100)
        >>> for i in range(101):
        ...     progress_cb(i, current_best=100-i)
    """
    last_update = 0
    start_time = time.time()
    
    def progress_callback(iteration: int, 
                         current_best: Optional[float] = None, 
                         message: Optional[str] = None):
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
            # 导入格式化函数
            from .formatting import format_execution_time
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
    重试装饰器，自动重试失败的函数调用
    
    Args:
        max_retries: 最大重试次数
        delay: 初始重试间隔
        backoff_factor: 退避因子
        exceptions: 要捕获的异常类型
        
    Returns:
        装饰器函数
        
    Examples:
        >>> @retry_on_exception(max_retries=3, delay=1.0)
        ... def unstable_function():
        ...     # 可能失败的操作
        ...     pass
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


def calculate_statistics(values: List[float]) -> Dict[str, Union[float, str, int]]:
    """
    计算数值列表的统计信息
    
    Args:
        values: 数值列表
        
    Returns:
        统计信息字典
        
    Examples:
        >>> values = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> stats = calculate_statistics(values)
        >>> print(f"平均值: {stats['mean']}")
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
        
    Examples:
        >>> data = {'a': 1, 'b': 2, 'c': 3}
        >>> filtered = filter_dict_by_keys(data, ['a', 'c'], include=True)
        >>> print(filtered)  # {'a': 1, 'c': 3}
    """
    if include:
        return {k: v for k, v in data.items() if k in keys}
    else:
        return {k: v for k, v in data.items() if k not in keys}


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
        
    Examples:
        >>> numbers = extract_numbers_from_text("价格是 123.45 元和 67.89 元")
        >>> print(numbers)  # [123.45, 67.89]
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


# 设置numpy打印选项
setup_numpy_print_options()


if __name__ == "__main__":
    # 测试工具函数
    print("=== Utils Testing ===")
    
    # 测试参数标准化
    test_params = {
        'quality_threshold': np.array([0.6]),
        'distortion_distance': np.float64(20.0),
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
    
    # 测试数字提取
    text = "价格是 123.45 元和 67.89 元"
    numbers = extract_numbers_from_text(text)
    print(f"Extracted numbers: {numbers}")
    
    print("Utils testing completed!")
