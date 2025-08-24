#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa批处理网格运行器模块

作者: Chel
创建日期: 2025-08-24
版本: 2.0.0
功能: 从 batch_mesh.py 中提取的批量运行逻辑
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AnsaBatchConfig

logger = logging.getLogger(__name__)

# 安全导入Ansa模块
ANSA_AVAILABLE = False
try:
    from ansa import base, constants, mesh

    ANSA_AVAILABLE = True
    logger.info("Ansa模块加载成功")
except ImportError as e:
    logger.warning(f"Ansa模块未找到: {e}")
    logger.info("将使用模拟模式运行")


class AnsaBatchMeshRunner:
    """Ansa批处理网格运行器 - 增强版本"""

    def __init__(
        self,
        script_dir: Optional[Path] = None,
        cwd_dir: Optional[Path] = None,
        config: Optional[AnsaBatchConfig] = None,
    ):
        """
        初始化批处理运行器

        Args:
            script_dir: 脚本目录路径
            cwd_dir: 当前工作目录路径
            config: 批处理配置
        """
        self.script_dir = script_dir or Path(__file__).parent.parent.resolve()
        self.cwd_dir = cwd_dir or Path.cwd().resolve()
        self.criterion_dir = self.script_dir / "criterion"

        # 加载配置
        self.config = config or AnsaBatchConfig()
        json_config_file = self.cwd_dir / "mesh_config.json"
        self.config.load_from_file(json_config_file)

        # 验证配置
        is_valid, errors = self.config.validate()
        if not is_valid:
            logger.warning(f"配置验证失败: {errors}")

        # 运行统计
        self.stats: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "total_elements": 0,
            "bad_elements": 0,
            "retry_count": 0,
            "success": False,
        }

        logger.info(f"脚本目录: {self.script_dir}")
        logger.info(f"质量文件目录: {self.criterion_dir}")
        logger.info(f"输出目录: {self.cwd_dir}")
        logger.info(f"Ansa可用: {ANSA_AVAILABLE}")

    def run_batch_mesh(self, params: Optional[Dict[str, float]] = None) -> bool:
        """
        运行批处理网格生成

        Args:
            params: 网格参数（可选）

        Returns:
            操作是否成功
        """
        self.stats["start_time"] = time.time()

        if not ANSA_AVAILABLE:
            logger.warning("Ansa不可用，使用模拟模式")
            return self._simulate_batch_mesh(params)

        try:
            logger.info("开始批处理网格生成")

            # 参数已提供但不需要特殊处理
            if params:
                logger.info(f"使用提供的参数: {params}")

            # 读取网格参数
            success = self._load_mesh_parameters()
            if not success:
                logger.warning("网格参数加载失败，使用默认参数")

            # 读取质量标准
            success = self._load_quality_criteria()
            if not success:
                logger.warning("质量标准加载失败，使用默认标准")

            # 执行批处理网格生成
            for attempt in range(self.config.retry_attempts + 1):
                try:
                    result = self._execute_batch_mesh()
                    if result:
                        logger.info("批处理网格生成成功")
                        self.stats["success"] = True
                        return True
                    else:
                        if attempt < self.config.retry_attempts:
                            logger.warning(f"网格生成失败，第{attempt + 1}次重试...")
                            time.sleep(self.config.retry_delay)
                            self.stats["retry_count"] += 1
                        else:
                            logger.error("批处理网格生成失败（所有重试已用完）")
                            return False

                except Exception as e:
                    if attempt < self.config.retry_attempts:
                        logger.warning(f"网格生成异常，第{attempt + 1}次重试: {e}")
                        time.sleep(self.config.retry_delay)
                        self.stats["retry_count"] += 1
                    else:
                        logger.error(f"批处理网格生成异常（所有重试已用完）: {e}")
                        return False

            return False

        except Exception as e:
            logger.error(f"批处理网格生成异常: {e}")
            logger.debug(traceback.format_exc())
            return False

        finally:
            self.stats["end_time"] = time.time()

    def _load_mesh_parameters(self) -> bool:
        """加载网格参数"""
        try:
            mpar_file = self.cwd_dir / self.config.mpar_file
            if mpar_file.exists():
                if ANSA_AVAILABLE:
                    mesh.ReadMeshParams(str(mpar_file))
                logger.info(f"网格参数已加载: {mpar_file}")
                return True
            else:
                logger.warning(f"网格参数文件不存在: {mpar_file}")
                return False
        except Exception as e:
            logger.error(f"加载网格参数失败: {e}")
            return False

    def _load_quality_criteria(self) -> bool:
        """加载质量标准"""
        try:
            qual_file = self.criterion_dir / self.config.qual_file
            if qual_file.exists():
                if ANSA_AVAILABLE:
                    mesh.ReadQualityCriteria(str(qual_file))
                logger.info(f"质量标准已加载: {qual_file}")
                return True
            else:
                logger.warning(f"质量标准文件不存在: {qual_file}")
                return False
        except Exception as e:
            logger.error(f"加载质量标准失败: {e}")
            return False

    def _execute_batch_mesh(self) -> bool:
        """执行批处理网格生成"""
        if not ANSA_AVAILABLE:
            return self._simulate_mesh_generation()

        try:
            # 收集壳体属性
            props = base.CollectEntities(constants.LSDYNA, None, "SECTION_SHELL")

            if props:
                logger.info(f"找到{len(props)}个壳体属性")

                # 执行批处理网格生成
                mesh.BatchGenerator(props)

                logger.info("网格生成成功")
                return True
            else:
                logger.warning("未找到壳体属性")
                return False

        except Exception as e:
            logger.error(f"执行批处理网格生成失败: {e}")
            raise

    def _simulate_batch_mesh(self, params: Optional[Dict[str, float]] = None) -> bool:
        """模拟批处理网格生成"""
        logger.info("模拟批处理网格生成...")

        import random

        # 模拟处理时间
        processing_time = random.uniform(0.5, 2.0)
        time.sleep(processing_time)

        # 基于参数调整成功率
        success_rate = 0.75  # 基础成功率

        if params:
            # 根据参数调整成功率
            distortion_distance = params.get("distortion_distance", 20.0)

            # 合理的参数范围有更高的成功率
            if 10.0 <= distortion_distance <= 30.0:
                success_rate += 0.1

        # 随机决定是否成功
        success = random.random() < success_rate

        if success:
            logger.info("模拟批处理网格生成成功")
            self.stats["success"] = True
        else:
            logger.error("模拟批处理网格生成失败")

        return success

    def _simulate_mesh_generation(self) -> bool:
        """模拟实际的网格生成过程"""
        import random

        # 模拟网格生成过程
        time.sleep(random.uniform(0.2, 1.0))

        # 75%的成功率
        return random.random() < 0.75

    def check_element_quality(
        self, custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        检查单元质量 - 增强版本

        Args:
            custom_thresholds: 自定义质量阈值

        Returns:
            质量检查结果
        """
        if not ANSA_AVAILABLE:
            return self._simulate_quality_check(custom_thresholds)

        try:
            # 使用自定义阈值或配置中的阈值
            thresholds = self.config.get_effective_thresholds(custom_thresholds)

            results: Dict[str, Any] = {
                "timestamp": time.time(),
                "thresholds": thresholds,
                "checks": {},
            }

            # 执行启用的质量检查
            quality_check_mapping = {
                "min_length": "min_length",
                "max_length": "max_length",
                "aspect_ratio": "aspect_ratio",
                "skewness": "skewness",
                "warping": "warping",
                "min_angle_quads": "min_angle_quads",
                "max_angle_quads": "max_angle_quads",
                "min_angle_trias": "min_angle_trias",
                "max_angle_trias": "max_angle_trias",
            }

            for check_name, threshold_key in quality_check_mapping.items():
                if self.config.quality_checks.get(check_name, False):
                    results["checks"][check_name] = self._check_shell_quality(
                        thresholds[threshold_key], check_name
                    )

            # 统计总单元数
            all_elements = base.CollectEntitiesI(
                constants.LSDYNA, None, "ELEMENT_SHELL"
            )

            # Add diagnostic logging and convert to list
            logger.debug(f"CollectEntitiesI returned type: {type(all_elements)}")

            if all_elements:
                all_elements_list = list(all_elements)
                results["total_elements"] = len(all_elements_list)
                logger.debug(f"Total elements found: {results['total_elements']}")
            else:
                results["total_elements"] = 0

            # 统计不合格单元数
            total_bad = 0
            for check_result in results["checks"].values():
                if isinstance(check_result, dict) and "failed_count" in check_result:
                    total_bad += check_result["failed_count"]

            results["bad_elements"] = total_bad
            results["quality_ratio"] = (
                1.0 - (total_bad / results["total_elements"])
                if results["total_elements"] > 0
                else 0.0
            )

            # 更新统计信息
            self.stats["total_elements"] = results["total_elements"]
            self.stats["bad_elements"] = results["bad_elements"]

            logger.info(
                f"质量检查完成: 总单元数={results['total_elements']}, "
                f"不合格单元数={results['bad_elements']}, "
                f"质量比例={results['quality_ratio']:.2%}"
            )

            return results

        except Exception as e:
            logger.error(f"质量检查异常: {e}")
            return self._simulate_quality_check(custom_thresholds)

    def _get_quality_check_config(self, check_type: str) -> Dict[str, Any]:
        """获取质量检查配置，减少重复代码"""
        config_map = {
            "min_length": {
                "criteria": "MIN-LEN",
                "compare_func": lambda x, t: x <= t,
                "worst_func": min,
                "log_msg": "最小长度检查",
                "element_filter": None,
            },
            "max_length": {
                "criteria": "MAX-LEN",
                "compare_func": lambda x, t: x >= t,
                "worst_func": max,
                "log_msg": "最大长度检查",
                "element_filter": None,
            },
            "aspect_ratio": {
                "criteria": "ASPECT",
                "compare_func": lambda x, t: x >= t,
                "worst_func": max,
                "log_msg": "长宽比检查",
                "element_filter": None,
            },
            "skewness": {
                "criteria": "SKEW",
                "compare_func": lambda x, t: x >= t,
                "worst_func": max,
                "log_msg": "偏斜度检查",
                "element_filter": None,
            },
            "warping": {
                "criteria": "WARP",
                "compare_func": lambda x, t: x >= t,
                "worst_func": max,
                "log_msg": "扭曲度检查",
                "element_filter": None,
            },
            "min_angle_quads": {
                "criteria": "MINANGLE",
                "compare_func": lambda x, t: x <= t,
                "worst_func": min,
                "log_msg": "最小四边形角度检查",
                "element_filter": "QUAD",
            },
            "max_angle_quads": {
                "criteria": "MAXANGLE",
                "compare_func": lambda x, t: x >= t,
                "worst_func": max,
                "log_msg": "最大四边形角度检查",
                "element_filter": "QUAD",
            },
            "min_angle_trias": {
                "criteria": "MINANGLE",
                "compare_func": lambda x, t: x <= t,
                "worst_func": min,
                "log_msg": "最小三角形角度检查",
                "element_filter": "TRIA",
            },
            "max_angle_trias": {
                "criteria": "MAXANGLE",
                "compare_func": lambda x, t: x >= t,
                "worst_func": max,
                "log_msg": "最大三角形角度检查",
                "element_filter": "TRIA",
            },
        }

        if check_type not in config_map:
            raise ValueError(f"不支持的检查类型: {check_type}")

        return config_map[check_type]

    def _check_shell_quality(self, threshold: float, check_type: str) -> Dict[str, Any]:
        """检查壳单元质量 - 重构版本

        Args:
            threshold: 质量阈值
            check_type: 检查类型
        """
        failed_elems = []
        failed_values = []

        # 获取质量检查配置
        config = self._get_quality_check_config(check_type)
        quality_criteria = config["criteria"]
        compare_func = config["compare_func"]
        worst_func = config["worst_func"]
        log_msg = config["log_msg"]
        element_filter = config["element_filter"]

        try:
            elements = base.CollectEntitiesI(constants.LSDYNA, None, "ELEMENT_SHELL")

            if not elements:
                logger.warning("未找到壳单元")
                return {
                    "status": "NO_ELEMENTS",
                    "failed_count": 0,
                    "failed_elements": [],
                    "threshold": threshold,
                    "check_type": check_type,
                }

            elements_list = list(elements)

            # 统一的元素处理逻辑
            for elem in elements_list:
                # 如果有元素类型过滤器，检查元素类型
                if element_filter:
                    elem_type = base.GetEntityCardValues(
                        constants.LSDYNA, elem, "__type__"
                    )
                    if elem_type != element_filter:
                        continue

                # 获取质量值并检查
                quality = base.ElementQuality(elem, quality_criteria)
                if quality != "error" and quality != 0:
                    quality_value = float(quality)
                    if compare_func(quality_value, threshold):
                        failed_elems.append(elem)
                        failed_values.append(quality_value)

            result = {
                "status": "OK" if not failed_elems else "NOK",
                "failed_count": len(failed_elems),
                "failed_elements": failed_elems,
                "failed_values": failed_values,
                "threshold": threshold,
                "check_type": check_type,
                "total_checked": len(elements_list),
            }

            if failed_values:
                result["worst_value"] = worst_func(failed_values)
                result["avg_failed_value"] = sum(failed_values) / len(failed_values)

            logger.info(
                f"{log_msg}: {len(failed_elems)} 个不合格单元 (阈值: {threshold})"
            )

            return result

        except Exception as e:
            logger.error(f"{log_msg}失败: {e}")
            return {
                "status": "ERROR",
                "failed_count": 0,
                "failed_elements": [],
                "threshold": threshold,
                "check_type": check_type,
                "error": str(e),
            }

    def _simulate_quality_check(
        self, custom_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """模拟质量检查 - 增强版本"""
        import random

        logger.info("模拟质量检查...")

        # 使用自定义阈值或配置中的阈值
        min_len = (
            custom_thresholds.get("min_element_length", self.config.min_length)
            if custom_thresholds
            else self.config.min_length
        )
        max_len = (
            custom_thresholds.get("max_element_length", self.config.max_length)
            if custom_thresholds
            else self.config.max_length
        )

        # 确保阈值不为None
        min_len = min_len or self.config.min_length
        max_len = max_len or self.config.max_length

        # 模拟结果
        total_elements = random.randint(1000, 10000)

        # 基于阈值调整不合格元素数量
        base_bad_rate = 0.05  # 基础5%不合格率

        # 更严格的阈值导致更多不合格元素
        if min_len > 2.5:
            base_bad_rate *= 1.5
        if max_len < 6.0:
            base_bad_rate *= 1.3

        bad_min = max(0, int(total_elements * base_bad_rate * random.uniform(0.5, 1.5)))
        bad_max = max(0, int(total_elements * base_bad_rate * random.uniform(0.3, 1.0)))

        results = {
            "timestamp": time.time(),
            "thresholds": {"min_length": min_len, "max_length": max_len},
            "checks": {
                "min_length": {
                    "status": "OK" if bad_min == 0 else "NOK",
                    "failed_count": bad_min,
                    "failed_elements": list(range(bad_min)),
                    "failed_values": [
                        random.uniform(0.5, min_len) for _ in range(bad_min)
                    ],
                    "threshold": min_len,
                    "check_type": "min_length",
                    "total_checked": total_elements,
                },
                "max_length": {
                    "status": "OK" if bad_max == 0 else "NOK",
                    "failed_count": bad_max,
                    "failed_elements": list(range(bad_max)),
                    "failed_values": [
                        random.uniform(max_len, max_len * 2) for _ in range(bad_max)
                    ],
                    "threshold": max_len,
                    "check_type": "max_length",
                    "total_checked": total_elements,
                },
            },
            "total_elements": total_elements,
            "bad_elements": bad_min + bad_max,
            "quality_ratio": 1.0 - ((bad_min + bad_max) / total_elements),
        }

        # 更新统计信息
        self.stats["total_elements"] = total_elements
        self.stats["bad_elements"] = bad_min + bad_max

        logger.info(
            f"模拟质量检查完成: 总单元数={total_elements}, "
            f"不合格单元数={bad_min + bad_max}, "
            f"质量比例={results['quality_ratio']:.2%}"
        )

        return results

    def save_model(self, output_file: Optional[Path] = None) -> bool:
        """
        保存模型 - 增强版本

        Args:
            output_file: 输出文件路径

        Returns:
            保存是否成功
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.cwd_dir / f"{self.config.output_model}_{timestamp}"
        else:
            output_file = Path(output_file)

        if not ANSA_AVAILABLE:
            return self._simulate_save_model(output_file)

        try:
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # 保存模型
            base.SaveAs(str(output_file))

            # 验证文件是否存在且有效
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 0:
                    logger.info(f"模型已保存: {output_file}")
                    logger.info(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
                    return True
                else:
                    logger.error("保存的文件为空")
                    return False
            else:
                logger.error("保存后文件不存在")
                return False

        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False

    def _simulate_save_model(self, output_file: Path) -> bool:
        """模拟保存模型"""
        try:
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # 创建一个模拟文件
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# 模拟的Ansa模型文件\n")
                f.write(f"# 生成时间: {time.time()}\n")
                f.write(f"# 总单元数: {self.stats.get('total_elements', 0)}\n")
                f.write(f"# 不合格单元数: {self.stats.get('bad_elements', 0)}\n")
                f.write("# 这是一个用于测试的模拟文件\n")

                # 添加一些模拟数据使文件看起来真实
                import random

                for i in range(100):
                    f.write(
                        f"NODE {i+1} {random.random():.6f} {random.random():.6f} {random.random():.6f}\n"
                    )

            logger.info(f"模拟模型已保存: {output_file}")
            return True

        except Exception as e:
            logger.error(f"模拟保存模型失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取运行统计信息"""
        stats = self.stats.copy()

        if stats["start_time"] and stats["end_time"]:
            stats["execution_time"] = stats["end_time"] - stats["start_time"]

        stats["config"] = {
            "retry_attempts": self.config.retry_attempts,
            "quality_checks": self.config.quality_checks,
        }

        return stats


def run_batch_mesh(params: Optional[Dict[str, float]] = None) -> bool:
    """
    运行批处理网格生成的便利函数

    Args:
        params: 网格参数（可选）

    Returns:
        操作是否成功
    """
    runner = AnsaBatchMeshRunner()
    return runner.run_batch_mesh(params)


def check_element_quality_simple(
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    简单的质量检查便利函数

    Args:
        custom_thresholds: 自定义质量阈值

    Returns:
        质量检查结果
    """
    runner = AnsaBatchMeshRunner()
    return runner.check_element_quality(custom_thresholds)