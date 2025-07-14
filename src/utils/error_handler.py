#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一错误处理模块

作者: Chel
创建日期: 2025-07-14
版本: 1.0.0
功能: 提供统一的错误处理和日志记录
"""

import logging
import traceback
import functools
from typing import Any, Callable, Optional, Type, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class MeshOptimizerError(Exception):
    """网格优化器基础异常"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(MeshOptimizerError):
    """配置错误"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class ValidationError(MeshOptimizerError):
    """验证错误"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value


class OptimizationError(MeshOptimizerError):
    """优化过程错误"""
    
    def __init__(self, message: str, optimizer_type: Optional[str] = None, 
                 iteration: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="OPTIMIZATION_ERROR", **kwargs)
        self.optimizer_type = optimizer_type
        self.iteration = iteration


class EvaluationError(MeshOptimizerError):
    """评估过程错误"""
    
    def __init__(self, message: str, evaluator_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EVALUATION_ERROR", **kwargs)
        self.evaluator_type = evaluator_type


class FileOperationError(MeshOptimizerError):
    """文件操作错误"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="FILE_ERROR", **kwargs)
        self.file_path = file_path
        self.operation = operation


class DependencyError(MeshOptimizerError):
    """依赖错误"""
    
    def __init__(self, message: str, dependency: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DEPENDENCY_ERROR", **kwargs)
        self.dependency = dependency


def safe_execute(func: Callable, *args, default_return=None, 
                 log_errors: bool = True, **kwargs) -> Any:
    """
    安全执行函数，捕获并记录异常
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        default_return: 异常时的默认返回值
        log_errors: 是否记录错误
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"执行 {func.__name__} 时发生错误: {e}")
            logger.debug(f"错误详情: {traceback.format_exc()}")
        return default_return


def handle_exceptions(
    exception_map: Optional[dict] = None,
    default_exception: Type[Exception] = MeshOptimizerError,
    log_errors: bool = True,
    return_none_on_error: bool = False
):
    """
    异常处理装饰器
    
    Args:
        exception_map: 异常映射字典
        default_exception: 默认异常类型
        log_errors: 是否记录错误
        return_none_on_error: 异常时是否返回None
    """
    if exception_map is None:
        exception_map = {
            FileNotFoundError: FileOperationError,
            PermissionError: FileOperationError,
            ValueError: ValidationError,
            TypeError: ValidationError,
            ImportError: DependencyError,
        }
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"函数 {func.__name__} 执行失败: {e}")
                    logger.debug(f"错误详情: {traceback.format_exc()}")
                
                # 如果已经是自定义异常，直接抛出
                if isinstance(e, MeshOptimizerError):
                    if return_none_on_error:
                        return None
                    raise
                
                # 映射标准异常为自定义异常
                exception_type = type(e)
                if exception_type in exception_map:
                    custom_exception = exception_map[exception_type]
                    mapped_exc = custom_exception(
                        str(e), 
                        details={'original_exception': str(e)}
                    )
                    if return_none_on_error:
                        return None
                    raise mapped_exc from e
                
                # 未知异常包装为默认异常
                default_exc = default_exception(
                    f"未知错误: {str(e)}", 
                    details={'original_exception': str(e)}
                )
                if return_none_on_error:
                    return None
                raise default_exc from e
        
        return wrapper
    return decorator


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    验证文件路径
    
    Args:
        file_path: 文件路径
        must_exist: 文件是否必须存在
        
    Returns:
        验证后的Path对象
        
    Raises:
        FileOperationError: 路径验证失败
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise FileOperationError(
                f"文件不存在: {path}",
                file_path=str(path),
                operation="validate"
            )
        
        return path
    except Exception as e:
        if isinstance(e, FileOperationError):
            raise
        raise FileOperationError(
            f"路径验证失败: {e}",
            file_path=str(file_path),
            operation="validate"
        ) from e


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在
    
    Args:
        dir_path: 目录路径
        
    Returns:
        目录Path对象
        
    Raises:
        FileOperationError: 目录创建失败
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise FileOperationError(
            f"创建目录失败: {e}",
            file_path=str(dir_path),
            operation="create_directory"
        ) from e


def log_function_call(func: Callable) -> Callable:
    """
    记录函数调用的装饰器
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"调用函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise
    
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0,
                    exceptions: tuple = (Exception,)) -> Callable:
    """
    失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间
        backoff_factor: 退避因子
        exceptions: 要捕获的异常类型
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败: {e}"
                        )
                        raise
                    else:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, "
                            f"等待 {current_delay:.1f} 秒后重试"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # 这行代码理论上不会执行到，但为了类型检查
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class ErrorContext:
    """错误上下文管理器"""
    
    def __init__(self, operation: str, log_errors: bool = True):
        self.operation = operation
        self.log_errors = log_errors
        self.error_occurred = False
        self.error_details = None
    
    def __enter__(self):
        logger.debug(f"开始操作: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            self.error_details = {
                'type': exc_type.__name__,
                'message': str(exc_val),
                'traceback': traceback.format_exc()
            }
            
            if self.log_errors:
                logger.error(f"操作 {self.operation} 失败: {exc_val}")
                logger.debug(f"错误详情: {traceback.format_exc()}")
        else:
            logger.debug(f"操作 {self.operation} 成功完成")
        
        # 返回False表示不抑制异常
        return False


# 便捷函数
def create_error_context(operation: str, log_errors: bool = True) -> ErrorContext:
    """创建错误上下文"""
    return ErrorContext(operation, log_errors)


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """格式化错误消息"""
    message = f"{type(error).__name__}: {str(error)}"
    
    if include_traceback:
        message += f"\n{traceback.format_exc()}"
    
    return message


# 全局错误处理器
def setup_global_error_handler():
    """设置全局错误处理器"""
    import sys
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            "未捕获的异常",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception