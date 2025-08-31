"""
参数替换策略模块 - 使用策略模式重构参数替换逻辑
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ParameterReplacementStrategy(ABC):
    """参数替换策略抽象基类"""

    @abstractmethod
    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否可以处理给定的参数"""
        pass

    @abstractmethod
    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """执行参数替换"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        pass


class SimpleParameterReplacementStrategy(ParameterReplacementStrategy):
    """简单参数替换策略 - 处理非特殊策略的常规参数"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager

    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含需要简单替换的参数"""
        # 过滤掉需要特殊策略处理的参数
        special_param_prefixes = ["rule_fillet_width_", "rule_chamfer_width_", "treatment_hole_2d_"]
        simple_params = {
            k: v
            for k, v in params.items()
            if not any(k.startswith(prefix) for prefix in special_param_prefixes)
        }
        return len(simple_params) > 0

    def get_strategy_name(self) -> str:
        return "SimpleParameterReplacement"

    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换简单参数 - 直接更新原文件"""
        try:
            # 读取原始文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 执行简单参数替换
            updated_content = self._perform_simple_parameters_replacement(
                content, params
            )

            # 生成新文件路径
            updated_file_path = self._generate_output_path(
                file_path, "_simple_parameters_updated"
            )

            # 写入更新后的内容
            with open(updated_file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            logger.info(f"rule_fillet 参数替换完成，结果保存至: {updated_file_path}")
            return updated_file_path

        except Exception as e:
            logger.error(f"简单参数替换失败: {e}")
            return file_path

    def _perform_simple_parameters_replacement(
        self, content: str, params: Dict[str, float]
    ) -> str:
        """执行简单mpar文件参数 (非rule_fillet, rule_chamfer)替换"""
        # 更新参数值
        updated_content = content
        updated_count = 0

        # 过滤掉需要特殊策略处理的参数
        special_param_prefixes = ["rule_fillet_width_", "rule_chamfer_width_", "treatment_hole_2d_"]
        simple_params = {
            k: v
            for k, v in params.items()
            if not any(k.startswith(prefix) for prefix in special_param_prefixes)
        }

        for param_name, param_value in simple_params.items():
            # 根据参数名映射到mpar文件中的实际参数名
            mpar_param_name = self._map_param_to_mpar_name(param_name)

            # 查找参数行的模式，支持多种格式:
            # 1. PARAM_NAME = value
            # 2. PARAM_NAME: value
            # 3. PARAM_NAME value
            patterns = [
                rf"^(\s*{re.escape(mpar_param_name)}\s*=\s*)[^\r\n]*",  # equals format
                rf"^(\s*{re.escape(mpar_param_name)}\s*:\s*)[^\r\n]*",  # colon format
                rf"^(\s*{re.escape(mpar_param_name)}\s+)[^\r\n]*",  # space format
            ]

            param_found = False
            for pattern in patterns:
                if re.search(pattern, content, flags=re.MULTILINE):
                    # 格式化参数值
                    formatted_value = self._format_mpar_parameter_value(
                        param_name, param_value
                    )
                    replacement = rf"\g<1>{formatted_value}"
                    updated_content = re.sub(
                        pattern, replacement, updated_content, flags=re.MULTILINE
                    )
                    param_found = True
                    updated_count += 1
                    logger.debug(f"更新参数 {mpar_param_name} = {formatted_value}")
                    break

            # 如果没有找到参数，记录警告
            if not param_found:
                logger.warning(
                    f"参数 {param_name} (映射为 {mpar_param_name}) 未在mpar文件中找到"
                )

        logger.debug(f"已更新{updated_count}个简单参数")

        return updated_content

    def _generate_output_path(self, file_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")

    def _map_param_to_mpar_name(self, param_name: str) -> str:
        """将参数名映射到mpar文件中的实际参数名"""
        # 使用配置管理器的参数映射（如果可用）
        if self.config_manager and hasattr(self.config_manager, "parameter_space"):
            param_def = self.config_manager.parameter_space.get_parameter(param_name)
            if param_def and param_def.ansa_mapping:
                return param_def.ansa_mapping

        # 如果没有映射，返回原参数名
        return param_name

    def _format_mpar_parameter_value(self, param_name: str, value: float) -> str:
        """格式化mpar参数值"""
        # 根据参数名进行特殊格式化
        if param_name == "distortion_distance":
            # 扭曲距离需要加上%符号
            return f"{value}%"
        elif param_name == "distortion_angle":
            # 扭曲角度需要加上.符号
            return f"{value}."
        elif param_name == "perimeter_distance":
            # 周边距离需要加上*Lmin
            return f"{value}*Lmin"
        else:
            # 其他参数直接返回数值
            return str(value)


class RuleFilletReplacementStrategy(ParameterReplacementStrategy):
    """Rule Fillet参数替换策略"""

    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含rule_fillet_width参数"""
        return any(key.startswith("rule_fillet_width_") for key in params.keys())

    def get_strategy_name(self) -> str:
        return "RuleFilletReplacement"

    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换rule_fillet参数"""
        try:
            # 提取参数值
            width_values = self._extract_width_values(params)

            logger.info(f"应用 rule_fillet width 参数: {width_values}")

            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 执行替换
            updated_content = self._perform_replacements(content, width_values)

            # 生成新文件路径
            updated_file_path = self._generate_output_path(file_path, "_Fillet_updated")

            # 写入更新后的文件
            with open(updated_file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            logger.info(f"rule_fillet 参数替换完成，结果保存至: {updated_file_path}")
            return updated_file_path

        except Exception as e:
            logger.error(f"rule_fillet 参数替换失败: {e}")
            return file_path

    def _extract_width_values(self, params: Dict[str, float]) -> Dict[str, float]:
        """提取width参数值"""
        return {
            "width_1": params.get("rule_fillet_width_1", 3.0),
            "width_2": params.get("rule_fillet_width_2", 10.0),
            "width_3": params.get("rule_fillet_width_3", 20.0),
            "width_4": params.get("rule_fillet_width_4", 30.0),
        }

    def _perform_replacements(
        self, content: str, width_values: Dict[str, float]
    ) -> str:
        """执行具体的替换操作"""
        # 替换模式定义
        replacement_patterns = [
            {
                "pattern": r"(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*0-)(\d+(?:\.\d+)?)(.*?treatment\s*=\s*7)",
                "replacement": f"\\g<1>{width_values['width_1']}\\g<3>",
                "description": "第一个rule_fillet (treatment=7)",
            },
            {
                "pattern": r"(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*8)",
                "replacement": f"\\g<1>{width_values['width_1']}-{width_values['width_2']}\\g<4>",
                "description": "第二个rule_fillet (treatment=8)",
            },
            {
                "pattern": r"(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*9)",
                "replacement": f"\\g<1>{width_values['width_2']}-{width_values['width_3']}\\g<4>",
                "description": "第三个rule_fillet (treatment=9)",
            },
            {
                "pattern": r"(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*10)",
                "replacement": f"\\g<1>{width_values['width_3']}-{width_values['width_4']}\\g<4>",
                "description": "第四个rule_fillet (treatment=10)",
            },
        ]

        updated_content = content
        for pattern_info in replacement_patterns:
            updated_content = re.sub(
                pattern_info["pattern"], pattern_info["replacement"], updated_content
            )
            logger.debug(f"应用替换: {pattern_info['description']}")

        return updated_content

    def _generate_output_path(self, file_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")


# class RecognizeChampersReplacementStrategy(ParameterReplacementStrategy):
#     """Recognize Chamfers参数替换策略"""

#     def can_handle(self, params: Dict[str, float]) -> bool:
#         """检查是否包含recognize_chamfers参数"""
#         return any(key.startswith('recognize_chamfers_') for key in params.keys())

#     def get_strategy_name(self) -> str:
#         return "RecognizeChampersReplacement"

#     def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
#         """替换recognize_chamfers参数"""
#         try:
#             # 提取参数值
#             chamfer_params = self._extract_chamfer_params(params)

#             logger.info(f"应用 recognize_chamfers 参数: {chamfer_params}")

#             # 读取文件内容
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()

#             # 执行替换
#             updated_content = self._perform_chamfer_replacements(content, chamfer_params)

#             # 生成新文件路径
#             updated_file_path = self._generate_output_path(file_path, "_chamfer_updated")

#             # 写入更新后的内容
#             with open(updated_file_path, 'w', encoding='utf-8') as f:
#                 f.write(updated_content)

#             logger.info(f"recognize_chamfers 参数替换完成，结果保存至: {updated_file_path}")
#             return updated_file_path

#         except Exception as e:
#             logger.error(f"recognize_chamfers 参数替换失败: {e}")
#             return file_path

#     def _extract_chamfer_params(self, params: Dict[str, float]) -> Dict[str, float]:
#         """提取chamfer参数值"""
#         return {
#             'min_angle': params.get('recognize_chamfers_min_angle', 20.0),
#             'max_angle': params.get('recognize_chamfers_max_angle', 70.0),
#             'max_width': params.get('recognize_chamfers_max_width', 20.0)
#         }

#     def _perform_chamfer_replacements(self, content: str, chamfer_params: Dict[str, float]) -> str:
#         """执行chamfer参数替换"""
#         replacement_patterns = [
#             {
#                 'pattern': r'(recognize_chamfers_min_angle\s*=\s*)(\d+(?:\.\d+)?\.?)',
#                 'replacement': f"\\g<1>{chamfer_params['min_angle']}.",
#                 'description': 'min_angle替换'
#             },
#             {
#                 'pattern': r'(recognize_chamfers_max_angle\s*=\s*)(\d+(?:\.\d+)?\.?)',
#                 'replacement': f"\\g<1>{chamfer_params['max_angle']}.",
#                 'description': 'max_angle替换'
#             },
#             {
#                 'pattern': r'(recognize_chamfers_max_width\s*=\s*)(\d+(?:\.\d+)?\.?)',
#                 'replacement': f"\\g<1>{chamfer_params['max_width']}.",
#                 'description': 'max_width替换'
#             }
#         ]

#         updated_content = content
#         for pattern_info in replacement_patterns:
#             updated_content = re.sub(
#                 pattern_info['pattern'],
#                 pattern_info['replacement'],
#                 updated_content
#             )
#             logger.debug(f"应用替换: {pattern_info['description']}")

#         return updated_content

#     def _generate_output_path(self, file_path: str, suffix: str) -> str:
#         """生成输出文件路径"""
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         dir_name = os.path.dirname(file_path)
#         return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")


class RuleChamferReplacementStrategy(ParameterReplacementStrategy):
    """Rule Chamfer参数替换策略"""

    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含rule_chamfer_width参数"""
        return any(key.startswith("rule_chamfer_width_") for key in params.keys())

    def get_strategy_name(self) -> str:
        return "RuleChamferReplacement"

    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换rule_chamfer参数"""
        try:
            # 读取原始文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取参数值
            chamfer_width_1 = params.get("rule_chamfer_width_1")
            if chamfer_width_1 is None:
                logger.warning("未找到 rule_chamfer_width_1 参数")
                return file_path

            # 转换为整数
            chamfer_width_1_int = int(round(chamfer_width_1))

            # 执行替换
            updated_content = self._perform_rule_chamfer_replacement(
                content, chamfer_width_1_int
            )

            # 生成新文件路径
            updated_file_path = self._generate_output_path(
                file_path, "_chamfer_updated"
            )

            # 写入更新后的内容
            with open(updated_file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            logger.info(f"Rule chamfer 参数替换完成: {updated_file_path}")
            logger.info(
                f"rule_chamfer_width_1: {chamfer_width_1} -> {chamfer_width_1_int}"
            )

            return updated_file_path

        except Exception as e:
            logger.error(f"Rule chamfer 参数替换失败: {e}")
            return file_path

    def _perform_rule_chamfer_replacement(self, content: str, width_value: int) -> str:
        """执行rule_chamfer替换"""
        pattern = r"(rule_chamfer\s*=.*?width\s*=\s*0-)(\d+)(.*?treatment\s*=\s*12)"

        def replace_width(match):
            prefix = match.group(1)
            old_width = match.group(2)
            suffix = match.group(3)

            logger.info(f"替换 rule_chamfer 宽度值: {old_width} -> {width_value}")
            return f"{prefix}{width_value}{suffix}"

        return re.sub(pattern, replace_width, content, flags=re.IGNORECASE | re.DOTALL)

    def _generate_output_path(self, file_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")


class TreatmentHole2dReplacementStrategy(ParameterReplacementStrategy):
    """Treatment Hole 2D参数替换策略"""

    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含treatment_hole_2d参数"""
        return any(key.startswith("treatment_hole_2d_") for key in params.keys())

    def get_strategy_name(self) -> str:
        return "TreatmentHole2dReplacement"

    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换treatment_hole_2d参数"""
        try:
            # 读取原始文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取参数值
            treatment_params = self._extract_treatment_params(params)
            
            if not treatment_params:
                logger.warning("未找到有效的 treatment_hole_2d 参数")
                return file_path

            logger.info(f"应用 treatment_hole_2d 参数: {treatment_params}")

            # 执行替换
            updated_content = self._perform_treatment_hole_2d_replacements(
                content, treatment_params
            )

            # 生成新文件路径
            updated_file_path = self._generate_output_path(
                file_path, "_treatment_hole_2d_updated"
            )

            # 写入更新后的内容
            with open(updated_file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            logger.info(f"Treatment hole 2d 参数替换完成: {updated_file_path}")
            return updated_file_path

        except Exception as e:
            logger.error(f"Treatment hole 2d 参数替换失败: {e}")
            return file_path

    def _extract_treatment_params(self, params: Dict[str, float]) -> Dict[str, Any]:
        """提取treatment参数值"""
        treatment_params = {}
        
        # 提取各个参数
        if "treatment_hole_2d_N1" in params:
            treatment_params["N1"] = int(params["treatment_hole_2d_N1"])
        if "treatment_hole_2d_dw1" in params:
            treatment_params["dw1"] = params["treatment_hole_2d_dw1"]
        if "treatment_hole_2d_N2" in params:
            treatment_params["N2"] = int(params["treatment_hole_2d_N2"])
        if "treatment_hole_2d_dw2" in params:
            treatment_params["dw2"] = params["treatment_hole_2d_dw2"]
        if "treatment_hole_2d_dw3" in params:
            treatment_params["dw3"] = params["treatment_hole_2d_dw3"]
            
        return treatment_params

    def _perform_treatment_hole_2d_replacements(
        self, content: str, treatment_params: Dict[str, Any]
    ) -> str:
        """执行treatment_hole_2d具体替换操作"""
        updated_content = content
        
        # 第206行: treatment = 2, 替换 N=6 和 width = 2.5
        if "N1" in treatment_params or "dw1" in treatment_params:
            n1_value = treatment_params.get("N1", 6)
            dw1_value = treatment_params.get("dw1", 2.5)
            
            # 匹配treatment_hole_2d = 2行中的N值和width值
            pattern = r"(treatment_hole_2d\s*=\s*2\s*\|\|.*?number_value\s*=\s*N=)(\d+)(.*?specific_zones\s*=\s*width\s*=\s*)([\d.]+)(.*?)(\r?\n)"
            
            def replace_treatment_2(match):
                prefix = match.group(1)
                old_n = match.group(2)
                middle = match.group(3)
                old_width = match.group(4)
                suffix = match.group(5)
                newline = match.group(6)
                
                logger.debug(f"替换 treatment=2: N={old_n}->{n1_value}, width={old_width}->{dw1_value}")
                return f"{prefix}{n1_value}{middle}{dw1_value}{suffix}{newline}"
            
            updated_content = re.sub(pattern, replace_treatment_2, updated_content, flags=re.DOTALL)
        
        # 第207行: treatment = 3, 替换 N=8 和 width = 2.5
        if "N2" in treatment_params or "dw2" in treatment_params:
            n2_value = treatment_params.get("N2", 8)
            dw2_value = treatment_params.get("dw2", 2.5)
            
            pattern = r"(treatment_hole_2d\s*=\s*3\s*\|\|.*?number_value\s*=\s*N=)(\d+)(.*?specific_zones\s*=\s*width\s*=\s*)([\d.]+)(.*?)(\r?\n)"
            
            def replace_treatment_3(match):
                prefix = match.group(1)
                old_n = match.group(2)
                middle = match.group(3)
                old_width = match.group(4)
                suffix = match.group(5)
                newline = match.group(6)
                
                logger.debug(f"替换 treatment=3: N={old_n}->{n2_value}, width={old_width}->{dw2_value}")
                return f"{prefix}{n2_value}{middle}{dw2_value}{suffix}{newline}"
            
            updated_content = re.sub(pattern, replace_treatment_3, updated_content, flags=re.DOTALL)
        
        # 第208行: treatment = 4, 替换 width = 0.667*L 为纯数值
        if "dw3" in treatment_params:
            dw3_value = treatment_params["dw3"]
            
            # 修改正则表达式：将整个 "0.667*L" 作为一个整体进行匹配
            pattern = r"(treatment_hole_2d\s*=\s*4\s*\|\|.*?specific_zones\s*=\s*width\s*=\s*)([\d.]+\*L)(.*?)(\r?\n)"
            
            def replace_treatment_4(match):
                prefix = match.group(1)
                old_width_with_unit = match.group(2)  # 完整的 "0.667*L"
                suffix = match.group(3)
                newline = match.group(4)
                
                logger.debug(f"替换 treatment=4: width={old_width_with_unit}->{dw3_value}")
                return f"{prefix}{dw3_value}{suffix}{newline}"
                
            updated_content = re.sub(pattern, replace_treatment_4, updated_content, flags=re.DOTALL)

        return updated_content

    def _generate_output_path(self, file_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")


# class DistortionAngleReplacementStrategy(ParameterReplacementStrategy):
#     """Distortion Angle参数替换策略"""

#     def can_handle(self, params: Dict[str, float]) -> bool:
#         """检查是否包含distortion_angle参数"""
#         return 'distortion_angle' in params

#     def get_strategy_name(self) -> str:
#         return "DistortionAngleReplacement"

#     def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
#         """替换distortion_angle参数"""
#         try:
#             # 读取原始文件
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()

#             # 提取参数值
#             distortion_angle = params.get('distortion_angle')
#             if distortion_angle is None:
#                 logger.warning("未找到 distortion_angle 参数")
#                 return file_path

#             # 执行替换
#             updated_content = self._perform_distortion_replacement(content, distortion_angle)

#             # 生成新文件路径
#             updated_file_path = self._generate_output_path(file_path, "_distortion_angle_updated")

#             # 写入更新后的内容
#             with open(updated_file_path, 'w', encoding='utf-8') as f:
#                 f.write(updated_content)

#             logger.info(f"Distortion angle 参数替换完成: {updated_file_path}")
#             logger.info(f"distortion_angle: {distortion_angle}")

#             return updated_file_path

#         except Exception as e:
#             logger.error(f"Distortion angle 参数替换失败: {e}")
#             return file_path

#     def _perform_distortion_replacement(self, content: str, angle_value: float) -> str:
#         """执行distortion angle替换"""
#         pattern = r'(distortion-angle\s*=\s*)(\d+(?:\.\d+)?)(\.?)'

#         def replace_angle(match):
#             prefix = match.group(1)
#             old_angle = match.group(2)

#             logger.info(f"替换 distortion-angle 值: {old_angle} -> {angle_value}")
#             return f"{prefix}{angle_value}."

#         return re.sub(pattern, replace_angle, content, flags=re.IGNORECASE)

#     def _generate_output_path(self, file_path: str, suffix: str) -> str:
#         """生成输出文件路径"""
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         dir_name = os.path.dirname(file_path)
#         return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")

# class PerimeterDistanceReplacementStrategy(ParameterReplacementStrategy):
#     """Perimeter Distance参数替换策略"""

#     def can_handle(self, params: Dict[str, float]) -> bool:
#         """检查是否包含perimeter_distance参数"""
#         return 'perimeter_distance' in params

#     def get_strategy_name(self) -> str:
#         return "PerimeterDistanceReplacement"

#     def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
#         """替换perimeter_distance参数"""
#         try:
#             # 读取原始文件
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()

#             # 提取参数值
#             perimeter_distance = params.get('perimeter_distance')
#             if perimeter_distance is None:
#                 logger.warning("未找到 perimeter_distance 参数")
#                 return file_path

#             # 执行替换
#             updated_content = self._perform_perimeter_replacement(content, perimeter_distance)

#             # 生成新文件路径
#             updated_file_path = self._generate_output_path(file_path, "_perimeter_distance_updated")

#             # 写入更新后的内容
#             with open(updated_file_path, 'w', encoding='utf-8') as f:
#                 f.write(updated_content)

#             logger.info(f"Perimeter distance 参数替换完成: {updated_file_path}")
#             logger.info(f"perimeter_distance: {perimeter_distance}")

#             return updated_file_path

#         except Exception as e:
#             logger.error(f"Perimeter distance 参数替换失败: {e}")
#             return file_path

#     def _perform_perimeter_replacement(self, content: str, distance_value: float) -> str:
#         """执行perimeter distance替换"""
#         pattern = r'(remove_perimeters_with_distance\s*=\s*)(\d+(?:\.\d+)?)(\*Lmin)'

#         def replace_distance(match):
#             prefix = match.group(1)
#             old_distance = match.group(2)
#             suffix = match.group(3)

#             logger.info(f"替换 perimeter_distance 值: {old_distance} -> {distance_value}")
#             return f"{prefix}{distance_value}{suffix}"

#         return re.sub(pattern, replace_distance, content, flags=re.IGNORECASE)

#     def _generate_output_path(self, file_path: str, suffix: str) -> str:
#         """生成输出文件路径"""
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         dir_name = os.path.dirname(file_path)
#         return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")


class ParameterReplacementManager:
    """参数替换管理器 - 协调所有替换策略"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.strategies = [
            SimpleParameterReplacementStrategy(config_manager),
            RuleFilletReplacementStrategy(),
            # RecognizeChampersReplacementStrategy(),
            RuleChamferReplacementStrategy(),
            TreatmentHole2dReplacementStrategy(),
            # DistortionAngleReplacementStrategy(),
            # PerimeterDistanceReplacementStrategy()
        ]

    def process_parameter_replacements(
        self, file_path: str, params: Dict[str, float]
    ) -> str:
        """处理所有适用的参数替换"""
        current_file = file_path
        applied_strategies = []

        for strategy in self.strategies:
            if strategy.can_handle(params):
                logger.info(f"应用策略: {strategy.get_strategy_name()}")
                current_file = strategy.replace_parameters(current_file, params)
                applied_strategies.append(strategy.get_strategy_name())

        if applied_strategies:
            logger.info(f"已应用的替换策略: {', '.join(applied_strategies)}")
        else:
            logger.info("未找到适用的参数替换策略")

        return current_file

    def get_available_strategies(self) -> list:
        """获取所有可用的策略名称"""
        return [strategy.get_strategy_name() for strategy in self.strategies]

    def register_strategy(self, strategy: ParameterReplacementStrategy) -> None:
        """注册新的替换策略

        Args:
            strategy: 策略实例，必须是ParameterReplacementStrategy的子类实例

        Raises:
            ValueError: 如果策略名称已存在或策略不是ParameterReplacementStrategy的实例
        """
        if not isinstance(strategy, ParameterReplacementStrategy):
            raise ValueError(
                "Strategy must be an instance of ParameterReplacementStrategy"
            )

        strategy_name = strategy.get_strategy_name()

        # 检查策略名称是否已存在
        existing_names = self.get_available_strategies()
        if strategy_name in existing_names:
            raise ValueError(f"Strategy '{strategy_name}' already exists")

        self.strategies.append(strategy)
        logger.info(f"已注册新的替换策略: {strategy_name}")

    def unregister_strategy(self, strategy_name: str) -> None:
        """取消注册替换策略

        Args:
            strategy_name: 策略名称

        Raises:
            ValueError: 如果策略名称不存在
        """
        strategy_to_remove = None
        for strategy in self.strategies:
            if strategy.get_strategy_name() == strategy_name:
                strategy_to_remove = strategy
                break

        if strategy_to_remove is None:
            raise ValueError(f"Strategy '{strategy_name}' does not exist")

        self.strategies.remove(strategy_to_remove)
        logger.info(f"已取消注册替换策略: {strategy_name}")

    def get_strategy_by_name(self, strategy_name: str) -> ParameterReplacementStrategy:
        """根据名称获取策略实例

        Args:
            strategy_name: 策略名称

        Returns:
            ParameterReplacementStrategy: 策略实例

        Raises:
            ValueError: 如果策略名称不存在
        """
        for strategy in self.strategies:
            if strategy.get_strategy_name() == strategy_name:
                return strategy

        raise ValueError(f"Strategy '{strategy_name}' does not exist")


# 用于mesh_evaluator调用
format_mpar_parameter_value = (
    SimpleParameterReplacementStrategy()._format_mpar_parameter_value
)
