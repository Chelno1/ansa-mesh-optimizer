#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
格式化工具模块

包含时间格式化、表格格式化、字符串处理等相关功能。

作者: Chel
创建日期: 2025-08-23
版本: 1.0.0
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def format_execution_time(seconds: float) -> str:
    """
    格式化执行时间

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串

    Examples:
        >>> print(format_execution_time(30.5))
        30.5秒
        >>> print(format_execution_time(120))
        2.0分钟
        >>> print(format_execution_time(3661))
        1.0小时
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


def create_summary_table(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    max_width: int = 120,
) -> str:
    """
    创建简单的表格摘要

    Args:
        data: 数据列表
        columns: 要显示的列名
        max_width: 最大表格宽度

    Returns:
        格式化的表格字符串

    Examples:
        >>> data = [{'name': 'test1', 'value': 1.23}, {'name': 'test2', 'value': 4.56}]
        >>> table = create_summary_table(data)
        >>> print(table)
    """
    if not data:
        return "无数据"

    if columns is None:
        columns = list(data[0].keys())

    # 计算列宽
    col_widths = {}
    for col in columns:
        col_name_width = len(str(col))
        max_value_width = max(len(str(row.get(col, ""))) for row in data)
        col_widths[col] = min(
            max(col_name_width, max_value_width), max_width // len(columns)
        )

    # 创建表格
    lines = []

    # 标题行
    header = " | ".join(
        str(col).ljust(col_widths[col])[: col_widths[col]] for col in columns
    )
    lines.append(header)

    # 分隔线
    separator = " | ".join("-" * col_widths[col] for col in columns)
    lines.append(separator)

    # 数据行
    for row in data:
        data_line = " | ".join(
            truncate_string(str(row.get(col, "")), col_widths[col]).ljust(
                col_widths[col]
            )
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

    Examples:
        >>> result = truncate_string("This is a very long string", 10)
        >>> print(result)
        This is...
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."


def estimate_completion_time(
    start_time: float, current_iteration: int, total_iterations: int
) -> str:
    """
    估算完成时间

    Args:
        start_time: 开始时间戳
        current_iteration: 当前迭代次数
        total_iterations: 总迭代次数

    Returns:
        估算完成时间字符串

    Examples:
        >>> start = time.time()
        >>> time.sleep(0.1)  # 模拟一些工作
        >>> eta = estimate_completion_time(start, 1, 10)
        >>> print(eta)
    """
    if current_iteration <= 0:
        return "估算中..."

    elapsed_time = time.time() - start_time
    avg_time_per_iteration = elapsed_time / current_iteration
    remaining_iterations = total_iterations - current_iteration
    estimated_remaining_time = avg_time_per_iteration * remaining_iterations

    return format_execution_time(estimated_remaining_time)


def create_backup_filename(
    original_path: Union[str, Path], timestamp: bool = True
) -> Path:
    """
    创建备份文件名

    Args:
        original_path: 原始文件路径
        timestamp: 是否添加时间戳

    Returns:
        备份文件路径

    Examples:
        >>> from pathlib import Path
        >>> backup = create_backup_filename(Path('config.json'))
        >>> print(backup.name)
    """
    from pathlib import Path

    path = Path(original_path)

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{timestamp_str}{path.suffix}")
    else:
        return path.with_name(f"{path.stem}_backup{path.suffix}")


def format_number(
    value: Union[int, float], precision: int = 2, use_thousands_separator: bool = True
) -> str:
    """
    格式化数字显示

    Args:
        value: 数值
        precision: 小数精度
        use_thousands_separator: 是否使用千分位分隔符

    Returns:
        格式化的数字字符串

    Examples:
        >>> print(format_number(1234.5678))
        1,234.57
        >>> print(format_number(1234.5678, precision=1))
        1,234.6
    """
    try:
        if isinstance(value, float):
            formatted = f"{value:.{precision}f}"
        else:
            formatted = str(value)

        if use_thousands_separator and "." in formatted:
            integer_part, decimal_part = formatted.split(".")
            integer_part = f"{int(integer_part):,}"
            return f"{integer_part}.{decimal_part}"
        elif use_thousands_separator:
            return f"{int(formatted):,}"
        else:
            return formatted
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value: float, precision: int = 1) -> str:
    """
    格式化百分比显示

    Args:
        value: 数值 (0.0 到 1.0)
        precision: 小数精度

    Returns:
        格式化的百分比字符串

    Examples:
        >>> print(format_percentage(0.1234))
        12.3%
        >>> print(format_percentage(0.1234, precision=2))
        12.34%
    """
    try:
        percentage = value * 100
        return f"{percentage:.{precision}f}%"
    except (ValueError, TypeError):
        return f"{value}%"


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小显示

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        格式化的文件大小字符串

    Examples:
        >>> print(format_file_size(1024))
        1.0 KB
        >>> print(format_file_size(1048576))
        1.0 MB
    """
    try:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    except (ValueError, TypeError):
        return str(size_bytes)


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    创建文本进度条

    Args:
        current: 当前进度
        total: 总数
        width: 进度条宽度

    Returns:
        进度条字符串

    Examples:
        >>> print(create_progress_bar(30, 100))
        [===============               ] 30.0%
    """
    try:
        if total <= 0:
            return "[" + " " * width + "] 0.0%"

        progress = min(current / total, 1.0)
        filled_width = int(width * progress)

        bar = "[" + "=" * filled_width + " " * (width - filled_width) + "]"
        percentage = progress * 100

        return f"{bar} {percentage:.1f}%"
    except (ValueError, TypeError, ZeroDivisionError):
        return "[" + " " * width + "] N/A"


# 常用的正则表达式模式
PATTERNS = {
    "number": r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?",
    "integer": r"[-+]?\d+",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "filename": r'[^<>:"/\\|?*\x00-\x1f]+',
}


def extract_numbers_from_text(text: str, pattern: str = "number") -> List[float]:
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
    import logging
    import re

    logger = logging.getLogger(__name__)

    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}")

    matches = re.findall(PATTERNS[pattern], text)

    try:
        if pattern == "integer":
            return [float(int(match)) for match in matches]
        else:
            return [float(match) for match in matches]
    except ValueError as e:
        logger.warning(f"Number extraction failed: {e}")
        return []


if __name__ == "__main__":
    # 测试格式化功能
    print("=== Formatting Testing ===")

    # 测试时间格式化
    print("时间格式化测试:")
    print(f"30.5秒: {format_execution_time(30.5)}")
    print(f"120秒: {format_execution_time(120)}")
    print(f"3661秒: {format_execution_time(3661)}")

    # 测试表格创建
    print("\n表格格式化测试:")
    test_data = [
        {"name": "test1", "value": 1.23, "status": "ok"},
        {"name": "test2", "value": 4.56, "status": "running"},
    ]
    table = create_summary_table(test_data)
    print(table)

    # 测试数字格式化
    print("\n数字格式化测试:")
    print(f"1234.5678: {format_number(1234.5678)}")
    print(f"百分比 0.1234: {format_percentage(0.1234)}")

    # 测试进度条
    print("\n进度条测试:")
    print(create_progress_bar(30, 100))
    print(create_progress_bar(75, 100))

    # 测试数字提取
    print("\n数字提取测试:")
    text = "价格是 123.45 元和 67.89 元"
    numbers = extract_numbers_from_text(text)
    print(f"从文本 '{text}' 提取的数字: {numbers}")

    # 测试整数提取
    text2 = "总共有 10 个项目和 25 个子项目"
    integers = extract_numbers_from_text(text2, "integer")
    print(f"从文本 '{text2}' 提取的整数: {integers}")

    print("Formatting testing completed!")
