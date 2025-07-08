#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义异常类模块

作者: Chel
创建日期: 2025-07-04
版本: 1.0.0
功能: 定义项目专用的异常类，提供更好的错误处理
"""

from typing import Optional, Dict, Any


class AnsaMeshOptimizerError(Exception):
    """ANSA网格优化器基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AnsaMeshOptimizerError):
    """配置相关错误"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class OptimizationError(AnsaMeshOptimizerError):
    """优化过程错误"""
    
    def __init__(self, message: str, optimizer_type: Optional[str] = None, iteration: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="OPTIMIZATION_ERROR", **kwargs)
        self.optimizer_type = optimizer_type
        self.iteration = iteration


class EvaluationError(AnsaMeshOptimizerError):
    """评估过程错误"""
    
    def __init__(self, message: str, evaluator_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EVALUATION_ERROR", **kwargs)
        self.evaluator_type = evaluator_type


class FileOperationError(AnsaMeshOptimizerError):
    """文件操作错误"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="FILE_ERROR", **kwargs)
        self.file_path = file_path
        self.operation = operation


class DependencyError(AnsaMeshOptimizerError):
    """依赖相关错误"""
    
    def __init__(self, message: str, dependency: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DEPENDENCY_ERROR", **kwargs)
        self.dependency = dependency


class ValidationError(AnsaMeshOptimizerError):
    """数据验证错误"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value


class ResourceError(AnsaMeshOptimizerError):
    """资源相关错误（内存、CPU等）"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type


class TimeoutError(AnsaMeshOptimizerError):
    """超时错误"""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.timeout_duration = timeout_duration


class ConvergenceError(OptimizationError):
    """收敛性错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONVERGENCE_ERROR", **kwargs)


class ParameterError(ValidationError):
    """参数错误"""
    
    def __init__(self, message: str, parameter_name: Optional[str] = None, **kwargs):
        super().__init__(message, field=parameter_name, error_code="PARAMETER_ERROR", **kwargs)
        self.parameter_name = parameter_name


# 异常处理装饰器
def handle_exceptions(exception_map: Optional[Dict[type, type]] = None):
    """
    异常处理装饰器
    
    Args:
        exception_map: 异常映射字典，将标准异常映射为自定义异常
    """
    if exception_map is None:
        exception_map = {
            FileNotFoundError: FileOperationError,
            PermissionError: FileOperationError,
            ValueError: ValidationError,
            TypeError: ValidationError,
            ImportError: DependencyError,
        }
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 如果已经是自定义异常，直接抛出
                if isinstance(e, AnsaMeshOptimizerError):
                    raise
                
                # 映射标准异常为自定义异常
                exception_type = type(e)
                if exception_type in exception_map:
                    custom_exception = exception_map[exception_type]
                    raise custom_exception(str(e), details={'original_exception': str(e)}) from e
                
                # 未知异常包装为基础异常
                raise AnsaMeshOptimizerError(f"未知错误: {str(e)}", details={'original_exception': str(e)}) from e
        
        return wrapper
    return decorator


# 错误代码常量
class ErrorCodes:
    """错误代码常量"""
    CONFIG_ERROR = "CONFIG_ERROR"
    OPTIMIZATION_ERROR = "OPTIMIZATION_ERROR"
    EVALUATION_ERROR = "EVALUATION_ERROR"
    FILE_ERROR = "FILE_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    CONVERGENCE_ERROR = "CONVERGENCE_ERROR"
    PARAMETER_ERROR = "PARAMETER_ERROR"