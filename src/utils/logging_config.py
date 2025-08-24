#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一日志配置模块

提供统一的日志配置接口，避免各模块重复实现日志设置
支持控制台输出、文件输出、不同级别等配置选项

作者: Chel
创建日期: 2025-08-23
版本: 1.0.0
"""

import json
import logging
import logging.handlers
import sys
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LoggingConfig:
    """统一的日志配置类"""

    # 默认配置
    DEFAULT_CONFIG = {
        "level": "INFO",
        "format": {
            "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "simple": "%(levelname)s - %(message)s",
            "console": "%(levelname)s - %(message)s",
            "file": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        },
        "console": {"enabled": True, "level": "INFO", "format": "simple"},
        "file": {
            "enabled": False,
            "level": "DEBUG",
            "format": "detailed",
            "filename": None,
            "max_size": "10MB",
            "backup_count": 5,
            "rotation": True,
        },
        "modules": {
            "src.optimizers": "INFO",
            "src.evaluators": "INFO",
            "src.utils": "WARNING",
            "src.visualizations": "WARNING",
        },
    }

    # 级别映射
    LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "NOTSET": logging.NOTSET,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化日志配置

        Args:
            config: 日志配置字典
        """
        self.config = self._merge_config(config or {})
        self._formatters = {}
        self._handlers = {}
        self._is_configured = False

    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并用户配置和默认配置"""
        config = copy.deepcopy(self.DEFAULT_CONFIG)

        # 深度合并配置
        for key, value in user_config.items():
            if (
                key in config
                and isinstance(config[key], dict)
                and isinstance(value, dict)
            ):
                config[key].update(value)
            else:
                config[key] = value

        return config

    def _get_level(self, level: Union[str, int]) -> int:
        """获取日志级别"""
        if isinstance(level, int):
            return level
        elif isinstance(level, str):
            return self.LEVEL_MAP.get(level.upper(), logging.INFO)
        else:
            return logging.INFO

    def _create_formatter(self, format_name: str) -> logging.Formatter:
        """创建格式化器"""
        if format_name in self._formatters:
            return self._formatters[format_name]

        format_string = self.config["format"].get(
            format_name, self.config["format"]["simple"]
        )
        formatter = logging.Formatter(format_string)
        self._formatters[format_name] = formatter
        return formatter

    def _parse_file_size(self, size_str: str) -> int:
        """解析文件大小字符串"""
        size_str = size_str.upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

    def _create_console_handler(self) -> Optional[logging.Handler]:
        """创建控制台处理器"""
        console_config = self.config["console"]
        if not console_config.get("enabled", True):
            return None

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._get_level(console_config.get("level", "INFO")))

        format_name = console_config.get("format", "console")
        formatter = self._create_formatter(format_name)
        handler.setFormatter(formatter)

        self._handlers["console"] = handler
        return handler

    def _create_file_handler(self) -> Optional[logging.Handler]:
        """创建文件处理器"""
        file_config = self.config["file"]
        if not file_config.get("enabled", False):
            return None

        filename = file_config.get("filename")
        if not filename:
            return None

        # 确保日志目录存在
        log_file = Path(filename)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建文件处理器
        if file_config.get("rotation", True):
            max_size = self._parse_file_size(file_config.get("max_size", "10MB"))
            backup_count = file_config.get("backup_count", 5)
            handler = logging.handlers.RotatingFileHandler(
                filename=str(log_file),
                maxBytes=max_size,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            handler = logging.FileHandler(str(log_file), encoding="utf-8")

        handler.setLevel(self._get_level(file_config.get("level", "DEBUG")))

        format_name = file_config.get("format", "file")
        formatter = self._create_formatter(format_name)
        handler.setFormatter(formatter)

        self._handlers["file"] = handler
        return handler

    def setup_logging(self) -> None:
        """设置日志配置"""
        if self._is_configured:
            return

        # 获取根日志记录器
        root_logger = logging.getLogger()

        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 设置根日志级别
        root_level = self._get_level(self.config.get("level", "INFO"))
        root_logger.setLevel(root_level)

        # 创建并添加处理器
        handlers = []

        # 控制台处理器
        console_handler = self._create_console_handler()
        if console_handler:
            handlers.append(console_handler)
            root_logger.addHandler(console_handler)

        # 文件处理器
        file_handler = self._create_file_handler()
        if file_handler:
            handlers.append(file_handler)
            root_logger.addHandler(file_handler)

        # 配置模块特定的日志级别
        self._configure_module_loggers()

        self._is_configured = True

        # 记录配置信息
        logger = logging.getLogger(__name__)
        logger.info("日志系统已配置")
        logger.debug(f"处理器数量: {len(handlers)}")
        if console_handler:
            logger.debug("控制台处理器已启用")
        if file_handler:
            logger.debug(f"文件处理器已启用: {self.config['file']['filename']}")

    def _configure_module_loggers(self) -> None:
        """配置模块特定的日志记录器"""
        modules_config = self.config.get("modules", {})

        for module_name, level in modules_config.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(self._get_level(level))

    def update_level(
        self, level: Union[str, int], module: Optional[str] = None
    ) -> None:
        """
        更新日志级别

        Args:
            level: 新的日志级别
            module: 模块名（如果为None则更新根日志记录器）
        """
        target_level = self._get_level(level)

        if module:
            logger = logging.getLogger(module)
            logger.setLevel(target_level)
            self.config["modules"][module] = level
        else:
            root_logger = logging.getLogger()
            root_logger.setLevel(target_level)
            self.config["level"] = level

    def add_file_logging(
        self, filename: str, level: str = "DEBUG", format_name: str = "detailed"
    ) -> None:
        """
        添加文件日志

        Args:
            filename: 日志文件路径
            level: 日志级别
            format_name: 格式名称
        """
        self.config["file"].update(
            {
                "enabled": True,
                "filename": filename,
                "level": level,
                "format": format_name,
            }
        )

        if self._is_configured:
            # 重新配置
            self._is_configured = False
            self.setup_logging()

    def disable_console_logging(self) -> None:
        """禁用控制台日志"""
        self.config["console"]["enabled"] = False

        if self._is_configured and "console" in self._handlers:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._handlers["console"])
            del self._handlers["console"]

    def enable_verbose_mode(self) -> None:
        """启用详细模式"""
        self.config["level"] = "DEBUG"
        self.config["console"]["level"] = "DEBUG"
        self.config["console"]["format"] = "detailed"

        if self._is_configured:
            self.update_level("DEBUG")
            # 更新控制台处理器格式
            if "console" in self._handlers:
                handler = self._handlers["console"]
                formatter = self._create_formatter("detailed")
                handler.setFormatter(formatter)
                handler.setLevel(logging.DEBUG)

    def enable_quiet_mode(self) -> None:
        """启用静默模式（仅显示ERROR及以上级别）"""
        self.config["level"] = "ERROR"
        self.config["console"]["level"] = "ERROR"

        if self._is_configured:
            self.update_level("ERROR")
            if "console" in self._handlers:
                self._handlers["console"].setLevel(logging.ERROR)

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    def save_config(self, filename: str) -> None:
        """保存配置到文件"""
        config_file = Path(filename)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_config(cls, filename: str) -> "LoggingConfig":
        """从文件加载配置"""
        config_file = Path(filename)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {filename}")

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(config)


# 全局日志配置实例
_global_logging_config: Optional[LoggingConfig] = None


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> LoggingConfig:
    """
    设置日志配置（简化接口）

    Args:
        level: 日志级别
        log_file: 日志文件路径
        verbose: 是否启用详细模式
        quiet: 是否启用静默模式
        config: 自定义配置字典

    Returns:
        日志配置实例
    """
    global _global_logging_config

    # 创建配置
    if config is None:
        config = {}

    # 应用参数设置
    if verbose:
        level = "DEBUG"
        config.setdefault("console", {})["format"] = "detailed"
    elif quiet:
        level = "ERROR"

    config["level"] = level

    if log_file:
        config.setdefault("file", {}).update(
            {"enabled": True, "filename": log_file, "level": "DEBUG"}
        )

    # 创建或更新全局配置
    _global_logging_config = LoggingConfig(config)
    _global_logging_config.setup_logging()

    return _global_logging_config


def get_logging_config() -> Optional[LoggingConfig]:
    """获取全局日志配置实例"""
    return _global_logging_config


def create_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    创建带有统一配置的日志记录器

    Args:
        name: 日志记录器名称
        level: 可选的特定级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(
            LoggingConfig.LEVEL_MAP.get(str(level).upper(), logging.INFO)
            if isinstance(level, str)
            else level
        )

    return logger


def configure_module_logging(module_name: str, level: Union[str, int]) -> None:
    """
    配置特定模块的日志级别

    Args:
        module_name: 模块名称
        level: 日志级别
    """
    global _global_logging_config

    if _global_logging_config:
        _global_logging_config.update_level(level, module_name)
    else:
        # 如果全局配置不存在，直接配置模块日志记录器
        logger = logging.getLogger(module_name)
        logger.setLevel(
            LoggingConfig.LEVEL_MAP.get(str(level).upper(), logging.INFO)
            if isinstance(level, str)
            else level
        )


def reset_logging() -> None:
    """重置日志配置"""
    global _global_logging_config

    # 清除所有处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    _global_logging_config = None


# 便捷函数
def enable_debug_logging() -> None:
    """启用调试日志"""
    global _global_logging_config
    if _global_logging_config:
        _global_logging_config.enable_verbose_mode()


def disable_console_logging() -> None:
    """禁用控制台日志"""
    global _global_logging_config
    if _global_logging_config:
        _global_logging_config.disable_console_logging()


def add_file_logging(filename: str, level: str = "DEBUG") -> None:
    """添加文件日志"""
    global _global_logging_config
    if _global_logging_config:
        _global_logging_config.add_file_logging(filename, level)
    else:
        # 如果没有全局配置，创建一个简单的配置
        setup_logging(log_file=filename)


# 向后兼容的接口（对应原cli_main.py中的setup_logging函数）
def setup_cli_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    CLI兼容的日志设置函数

    Args:
        verbose: 是否启用详细输出
        log_file: 日志文件路径
    """
    setup_logging(
        level="DEBUG" if verbose else "INFO", log_file=log_file, verbose=verbose
    )


# 对应batch_mesh.py中的setup_logging函数
def setup_batch_logging(
    log_level: int = logging.INFO, log_dir: Optional[Path] = None
) -> None:
    """
    批处理兼容的日志设置函数

    Args:
        log_level: 日志级别
        log_dir: 日志目录
    """
    level_name = logging.getLevelName(log_level)
    log_file = None

    if log_dir:
        log_file = str(log_dir / "ansa_batch.log")

    setup_logging(level=level_name, log_file=log_file)
