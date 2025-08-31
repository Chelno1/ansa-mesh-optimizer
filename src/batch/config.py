#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa批处理配置管理模块

作者: Chel
创建日期: 2025-08-24
版本: 2.0.0
功能: 从 batch_mesh.py 中提取的配置处理逻辑
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AnsaBatchConfig:
    """Ansa批处理配置类 - 重构版本"""

    # 默认配置常量
    DEFAULT_THRESHOLDS = {
        "min_length": 1.5,
        "max_length": 6.0,
        "min_angle_quads": 45.0,
        "max_angle_quads": 135.0,
        "min_angle_trias": 30.0,
        "max_angle_trias": 120.0,
        "aspect_ratio": 4.0,
        "skewness": 60.0,
        "warping": 15.0,
        "jacobian": 0.65,
        "triangles %": 10.0,
        "triangles per node": 3.0,
    }

    DEFAULT_EXECUTION = {"timeout": 300, "retry_attempts": 3, "retry_delay": 1.0}

    DEFAULT_FILES = {
        # "qual_file": "8mm_v23.ansa_qual",
        "output_model": "output_mesh.ansa",
    }

    DEFAULT_QUALITY_CHECKS = {
        "min_length": True,
        "max_length": True,
        "aspect_ratio": True,
        "skewness": True,
        "jacobian": True,
        "warping": True,
        "min_angle_quads": True,
        "max_angle_quads": True,
        "min_angle_trias": True,
        "max_angle_trias": True,
        "triangles %": False,
        "triangles per node": True,
    }

    def __init__(self, cwd_dir: Optional[Path] = None):
        """
        初始化配置

        Args:
            cwd_dir: 当前工作目录路径
        """
        self.cwd_dir = cwd_dir or Path.cwd().resolve()

        # 显式初始化所有属性以避免类型检查错误
        # 阈值配置
        self.min_length = self.DEFAULT_THRESHOLDS["min_length"]
        self.max_length = self.DEFAULT_THRESHOLDS["max_length"]
        self.min_angle_quads = self.DEFAULT_THRESHOLDS["min_angle_quads"]
        self.max_angle_quads = self.DEFAULT_THRESHOLDS["max_angle_quads"]
        self.min_angle_trias = self.DEFAULT_THRESHOLDS["min_angle_trias"]
        self.max_angle_trias = self.DEFAULT_THRESHOLDS["max_angle_trias"]
        self.aspect_ratio = self.DEFAULT_THRESHOLDS["aspect_ratio"]
        self.skewness = self.DEFAULT_THRESHOLDS["skewness"]
        self.warping = self.DEFAULT_THRESHOLDS["warping"]
        self.jacobian = self.DEFAULT_THRESHOLDS["jacobian"]
        self.triangles_percent = self.DEFAULT_THRESHOLDS["triangles %"]
        self.triangles_per_node = self.DEFAULT_THRESHOLDS["triangles per node"]

        # 执行配置
        self.timeout = self.DEFAULT_EXECUTION["timeout"]
        self.retry_attempts = self.DEFAULT_EXECUTION["retry_attempts"]
        self.retry_delay = self.DEFAULT_EXECUTION["retry_delay"]

        # 文件配置
        # self.qual_file = self.DEFAULT_FILES["qual_file"]
        self.output_model = self.DEFAULT_FILES["output_model"]

        # 动态查找mpar与qual文件
        self.mpar_file = self._find_mpar_file()
        self.qual_file = self._find_qual_file()

        # 初始化质量检查配置
        self.quality_checks = self.DEFAULT_QUALITY_CHECKS.copy()

    def _find_mpar_file(self) -> str:
        """在当前工作目录下查找.ansa_mpar文件"""
        try:
            mpar_files = list(self.cwd_dir.glob("*.ansa_mpar"))
            if mpar_files:
                return mpar_files[0].name
            else:
                # 如果没有找到，返回默认值
                return "mend.ansa_mpar"
        except Exception as e:
            logger.warning(f"查找.ansa_mpar文件失败: {e}")
            return "mend.ansa_mpar"

    def _find_qual_file(self) -> str:
        """在当前工作目录下查找.ansa_qual文件"""
        try:
            qual_files = list(self.cwd_dir.glob("*.ansa_qual"))
            if qual_files:
                return qual_files[0].name
            else:
                # 如果没有找到，返回默认值
                return "mend.ansa_qual"
        except Exception as e:
            logger.warning(f"查找.ansa_qual文件失败: {e}")
            return "mend.ansa_qual"

    def load_from_file(self, json_config_file: Path) -> None:
        """
        从文件加载配置

        Args:
            json_config_file: 配置文件路径
        """
        try:
            if json_config_file.exists():
                with open(json_config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                logger.info(f"配置已从{json_config_file}加载")
            else:
                logger.info("配置文件不存在，使用默认配置")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")

    def save_to_file(self, config_file: Path) -> None:
        """
        保存配置到文件

        Args:
            config_file: 配置文件路径
        """
        try:
            config_data = {}
            for attr in dir(self):
                if not attr.startswith("_") and not callable(getattr(self, attr)):
                    value = getattr(self, attr)
                    # 将Path对象转换为字符串
                    if isinstance(value, Path):
                        value = str(value)
                    # 跳过复杂的对象类型
                    elif isinstance(value, (str, int, float, bool, list, dict)):
                        config_data[attr] = value
                    elif value is None:
                        config_data[attr] = value

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"配置已保存到{config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证配置

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        if self.min_length <= 0:
            errors.append("min_element_length must be positive")

        if self.max_length <= self.min_length:
            errors.append("max_element_length must be greater than min_element_length")

        if self.timeout <= 0:
            errors.append("timeout must be positive")

        if self.retry_attempts < 0:
            errors.append("retry_attempts must be non-negative")

        return len(errors) == 0, errors

    def update_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        更新质量阈值

        Args:
            thresholds: 新的阈值字典
        """
        for key, value in thresholds.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"更新阈值 {key}: {value}")

    def get_effective_thresholds(
        self, custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        获取有效的质量阈值

        Args:
            custom_thresholds: 自定义阈值（可选）

        Returns:
            有效的阈值字典
        """
        default_thresholds = {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "aspect_ratio": self.aspect_ratio,
            "skewness": self.skewness,
            "warping": self.warping,
            "min_angle_quads": self.min_angle_quads,
            "max_angle_quads": self.max_angle_quads,
            "min_angle_trias": self.min_angle_trias,
            "max_angle_trias": self.max_angle_trias,
            "jacobian": self.jacobian,
            "triangles %": self.triangles_percent,
            "triangles per node": self.triangles_per_node,
        }

        if custom_thresholds:
            default_thresholds.update(custom_thresholds)

        return default_thresholds

    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要

        Returns:
            配置摘要字典
        """
        return {
            "thresholds": self.get_effective_thresholds(),
            "execution": {
                "timeout": self.timeout,
                "retry_attempts": self.retry_attempts,
                "retry_delay": self.retry_delay,
            },
            "files": {
                "qual_file": self.qual_file,
                "output_model": self.output_model,
                "mpar_file": self.mpar_file,
            },
            "quality_checks": self.quality_checks.copy(),
        }


def create_default_config(cwd_dir: Optional[Path] = None) -> AnsaBatchConfig:
    """
    创建默认配置

    Args:
        cwd_dir: 当前工作目录路径

    Returns:
        默认配置实例
    """
    return AnsaBatchConfig(cwd_dir)


def load_config_from_file(
    config_file: Path, cwd_dir: Optional[Path] = None
) -> AnsaBatchConfig:
    """
    从文件加载配置

    Args:
        config_file: 配置文件路径
        cwd_dir: 当前工作目录路径

    Returns:
        配置实例
    """
    config = AnsaBatchConfig(cwd_dir)
    config.load_from_file(config_file)
    return config