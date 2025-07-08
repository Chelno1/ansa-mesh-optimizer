"""
参数替换策略模块 - 使用策略模式重构参数替换逻辑
"""

import re
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

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

class RuleFilletReplacementStrategy(ParameterReplacementStrategy):
    """Rule Fillet参数替换策略"""
    
    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含rule_fillet_width参数"""
        return any(key.startswith('rule_fillet_width_') for key in params.keys())
    
    def get_strategy_name(self) -> str:
        return "RuleFilletReplacement"
    
    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换rule_fillet参数"""
        output_file = file_path + "_fillet_updated"
        
        try:
            # 提取参数值
            width_values = self._extract_width_values(params)
            
            logger.info(f"应用 rule_fillet width 参数: {width_values}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 执行替换
            updated_content = self._perform_replacements(content, width_values)
            
            # 写入更新后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"rule_fillet 参数替换完成，结果保存至: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"rule_fillet 参数替换失败: {e}")
            return file_path
    
    def _extract_width_values(self, params: Dict[str, float]) -> Dict[str, float]:
        """提取width参数值"""
        return {
            'width_1': params.get('rule_fillet_width_1', 3.0),
            'width_2': params.get('rule_fillet_width_2', 10.0),
            'width_3': params.get('rule_fillet_width_3', 20.0),
            'width_4': params.get('rule_fillet_width_4', 30.0)
        }
    
    def _perform_replacements(self, content: str, width_values: Dict[str, float]) -> str:
        """执行具体的替换操作"""
        # 替换模式定义
        replacement_patterns = [
            {
                'pattern': r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*0-)(\d+(?:\.\d+)?)(.*?treatment\s*=\s*7)',
                'replacement': f"\\g<1>{width_values['width_1']}\\g<3>",
                'description': '第一个rule_fillet (treatment=7)'
            },
            {
                'pattern': r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*8)',
                'replacement': f"\\g<1>{width_values['width_1']}-{width_values['width_2']}\\g<4>",
                'description': '第二个rule_fillet (treatment=8)'
            },
            {
                'pattern': r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*9)',
                'replacement': f"\\g<1>{width_values['width_2']}-{width_values['width_3']}\\g<4>",
                'description': '第三个rule_fillet (treatment=9)'
            },
            {
                'pattern': r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*10)',
                'replacement': f"\\g<1>{width_values['width_3']}-{width_values['width_4']}\\g<4>",
                'description': '第四个rule_fillet (treatment=10)'
            }
        ]
        
        updated_content = content
        for pattern_info in replacement_patterns:
            updated_content = re.sub(
                pattern_info['pattern'], 
                pattern_info['replacement'], 
                updated_content
            )
            logger.debug(f"应用替换: {pattern_info['description']}")
        
        return updated_content

class RecognizeChampersReplacementStrategy(ParameterReplacementStrategy):
    """Recognize Chamfers参数替换策略"""
    
    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含recognize_chamfers参数"""
        return any(key.startswith('recognize_chamfers_') for key in params.keys())
    
    def get_strategy_name(self) -> str:
        return "RecognizeChampersReplacement"
    
    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换recognize_chamfers参数"""
<<<<<<< HEAD
=======
        output_file = file_path + "_chamfers_updated"
        
>>>>>>> main
        try:
            # 提取参数值
            chamfer_params = self._extract_chamfer_params(params)
            
            logger.info(f"应用 recognize_chamfers 参数: {chamfer_params}")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 执行替换
            updated_content = self._perform_chamfer_replacements(content, chamfer_params)
            
<<<<<<< HEAD
            # 生成新文件路径
            updated_file_path = self._generate_output_path(file_path, "_chamfer_updated")
            
            # 写入更新后的内容
            with open(updated_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
                        
            logger.info(f"recognize_chamfers 参数替换完成，结果保存至: {updated_file_path}")
            return updated_file_path
=======
            # 写入更新后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"recognize_chamfers 参数替换完成，结果保存至: {output_file}")
            return output_file
>>>>>>> main
            
        except Exception as e:
            logger.error(f"recognize_chamfers 参数替换失败: {e}")
            return file_path
    
    def _extract_chamfer_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """提取chamfer参数值"""
        return {
            'min_angle': params.get('recognize_chamfers_min_angle', 20.0),
            'max_angle': params.get('recognize_chamfers_max_angle', 70.0),
            'max_width': params.get('recognize_chamfers_max_width', 20.0)
        }
    
    def _perform_chamfer_replacements(self, content: str, chamfer_params: Dict[str, float]) -> str:
        """执行chamfer参数替换"""
        replacement_patterns = [
            {
                'pattern': r'(recognize_chamfers_min_angle\s*=\s*)(\d+(?:\.\d+)?\.?)',
                'replacement': f"\\g<1>{chamfer_params['min_angle']}.",
                'description': 'min_angle替换'
            },
            {
                'pattern': r'(recognize_chamfers_max_angle\s*=\s*)(\d+(?:\.\d+)?\.?)',
                'replacement': f"\\g<1>{chamfer_params['max_angle']}.",
                'description': 'max_angle替换'
            },
            {
                'pattern': r'(recognize_chamfers_max_width\s*=\s*)(\d+(?:\.\d+)?\.?)',
                'replacement': f"\\g<1>{chamfer_params['max_width']}.",
                'description': 'max_width替换'
            }
        ]
        
        updated_content = content
        for pattern_info in replacement_patterns:
            updated_content = re.sub(
                pattern_info['pattern'], 
                pattern_info['replacement'], 
                updated_content
            )
            logger.debug(f"应用替换: {pattern_info['description']}")
        
        return updated_content
<<<<<<< HEAD
    
    def _generate_output_path(self, file_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")
=======
>>>>>>> main

class RuleChamferReplacementStrategy(ParameterReplacementStrategy):
    """Rule Chamfer参数替换策略"""
    
    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含rule_chamfer_width参数"""
        return any(key.startswith('rule_chamfer_width_') for key in params.keys())
    
    def get_strategy_name(self) -> str:
        return "RuleChamferReplacement"
    
    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换rule_chamfer参数"""
        try:
            # 读取原始文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取参数值
            chamfer_width_1 = params.get('rule_chamfer_width_1')
            if chamfer_width_1 is None:
                logger.warning("未找到 rule_chamfer_width_1 参数")
                return file_path
            
            # 转换为整数
            chamfer_width_1_int = int(round(chamfer_width_1))
            
            # 执行替换
            updated_content = self._perform_rule_chamfer_replacement(content, chamfer_width_1_int)
            
            # 生成新文件路径
            updated_file_path = self._generate_output_path(file_path, "_chamfer_updated")
            
            # 写入更新后的内容
            with open(updated_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Rule chamfer 参数替换完成: {updated_file_path}")
            logger.info(f"rule_chamfer_width_1: {chamfer_width_1} -> {chamfer_width_1_int}")
            
            return updated_file_path
            
        except Exception as e:
            logger.error(f"Rule chamfer 参数替换失败: {e}")
            return file_path
    
    def _perform_rule_chamfer_replacement(self, content: str, width_value: int) -> str:
        """执行rule_chamfer替换"""
        pattern = r'(rule_chamfer\s*=.*?width\s*=\s*0-)(\d+)(.*?treatment\s*=\s*12)'
        
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

class DistortionAngleReplacementStrategy(ParameterReplacementStrategy):
    """Distortion Angle参数替换策略"""
    
    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含distortion_angle参数"""
        return 'distortion_angle' in params
    
    def get_strategy_name(self) -> str:
        return "DistortionAngleReplacement"
    
    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换distortion_angle参数"""
        try:
            # 读取原始文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取参数值
            distortion_angle = params.get('distortion_angle')
            if distortion_angle is None:
                logger.warning("未找到 distortion_angle 参数")
                return file_path
            
            # 执行替换
            updated_content = self._perform_distortion_replacement(content, distortion_angle)
            
            # 生成新文件路径
            updated_file_path = self._generate_output_path(file_path, "_distortion_angle_updated")
            
            # 写入更新后的内容
            with open(updated_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Distortion angle 参数替换完成: {updated_file_path}")
            logger.info(f"distortion_angle: {distortion_angle}")
            
            return updated_file_path
            
        except Exception as e:
            logger.error(f"Distortion angle 参数替换失败: {e}")
            return file_path
    
    def _perform_distortion_replacement(self, content: str, angle_value: float) -> str:
        """执行distortion angle替换"""
        pattern = r'(distortion-angle\s*=\s*)(\d+(?:\.\d+)?)(\.?)'
        
        def replace_angle(match):
            prefix = match.group(1)
            old_angle = match.group(2)
            
            logger.info(f"替换 distortion-angle 值: {old_angle} -> {angle_value}")
            return f"{prefix}{angle_value}."
        
        return re.sub(pattern, replace_angle, content, flags=re.IGNORECASE)
    
    def _generate_output_path(self, file_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")

class PerimeterDistanceReplacementStrategy(ParameterReplacementStrategy):
    """Perimeter Distance参数替换策略"""
    
    def can_handle(self, params: Dict[str, float]) -> bool:
        """检查是否包含perimeter_distance参数"""
        return 'perimeter_distance' in params
    
    def get_strategy_name(self) -> str:
        return "PerimeterDistanceReplacement"
    
    def replace_parameters(self, file_path: str, params: Dict[str, float]) -> str:
        """替换perimeter_distance参数"""
        try:
            # 读取原始文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取参数值
            perimeter_distance = params.get('perimeter_distance')
            if perimeter_distance is None:
                logger.warning("未找到 perimeter_distance 参数")
                return file_path
            
            # 执行替换
            updated_content = self._perform_perimeter_replacement(content, perimeter_distance)
            
            # 生成新文件路径
            updated_file_path = self._generate_output_path(file_path, "_perimeter_distance_updated")
            
            # 写入更新后的内容
            with open(updated_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Perimeter distance 参数替换完成: {updated_file_path}")
            logger.info(f"perimeter_distance: {perimeter_distance}")
            
            return updated_file_path
            
        except Exception as e:
            logger.error(f"Perimeter distance 参数替换失败: {e}")
            return file_path
    
    def _perform_perimeter_replacement(self, content: str, distance_value: float) -> str:
        """执行perimeter distance替换"""
        pattern = r'(remove_perimeters_with_distance\s*=\s*)(\d+(?:\.\d+)?)(\*Lmin)'
        
        def replace_distance(match):
            prefix = match.group(1)
            old_distance = match.group(2)
            suffix = match.group(3)
            
            logger.info(f"替换 perimeter_distance 值: {old_distance} -> {distance_value}")
            return f"{prefix}{distance_value}{suffix}"
        
        return re.sub(pattern, replace_distance, content, flags=re.IGNORECASE)
    
    def _generate_output_path(self, file_path: str, suffix: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        return os.path.join(dir_name, f"{base_name}{suffix}.ansa_mpar")

class ParameterReplacementManager:
    """参数替换管理器 - 协调所有替换策略"""
    
    def __init__(self):
        self.strategies = [
            RuleFilletReplacementStrategy(),
            RecognizeChampersReplacementStrategy(),
            RuleChamferReplacementStrategy(),
            DistortionAngleReplacementStrategy(),
            PerimeterDistanceReplacementStrategy()
        ]
    
    def process_parameter_replacements(self, file_path: str, params: Dict[str, float]) -> str:
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