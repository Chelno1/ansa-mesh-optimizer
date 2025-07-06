#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估接口模块 - 改进版本

作者: Chel
创建日期: 2025-06-19
版本: 1.2.0
更新日期: 2025-06-20
修复: 错误处理，参数验证，Mock评估器
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple, Any, Union
import subprocess
import os
import re
import logging
from pathlib import Path
import tempfile
import glob
import time
import random

try:
    from src.config.config_refactored import unified_config_manager as config_manager
except ImportError:
    from config.config_refactored import unified_config_manager as config_manager

logger = logging.getLogger(__name__)

def normalize_params(params: Dict[str, Any]) -> Dict[str, float]:
    """
    标准化参数字典，确保类型正确
    
    Args:
        params: 参数字典
        
    Returns:
        标准化后的参数字典
    """
    normalized = {}
    
    for key, value in params.items():
        if hasattr(value, 'item'):  # numpy类型
            normalized[key] = float(value.item())
        elif hasattr(value, 'dtype'):  # numpy数组等
            if hasattr(value, 'size') and value.size == 1:
                normalized[key] = float(value.item())
            else:
                normalized[key] = float(value.tolist()[0]) if hasattr(value, 'tolist') else float(value)
        else:
            normalized[key] = float(value)
    
    return normalized

class MeshEvaluator(ABC):
    """网格评估器抽象基类"""
    
    @abstractmethod
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        评估网格质量
        
        Args:
            params: 网格参数字典
            
        Returns:
            网格质量评分（越小越好）
        """
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, float]) -> bool:
        """
        验证参数有效性
        
        Args:
            params: 网格参数字典
            
        Returns:
            参数是否有效
        """
        pass

class ParameterValidator:
    """参数验证器"""
    
    def __init__(self, param_space):
        self.param_space = param_space
        self.bounds = param_space.get_bounds()
        self.param_names = param_space.get_parameter_names()
        self.param_types = param_space.get_parameter_types()
    
    def validate_comprehensive(self, params: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        全面的参数验证
        
        Returns:
            (is_valid, error_message, cleaned_params)
        """
        errors = []
        cleaned_params = {}
        
        # 检查必需参数
        for name in self.param_names:
            if name not in params:
                errors.append(f"缺少必需参数: {name}")
                continue
            
            value = params[name]
            
            # 类型转换和验证
            try:
                cleaned_value = self._clean_and_validate_param(name, value)
                cleaned_params[name] = cleaned_value
            except ValueError as e:
                errors.append(f"参数 {name} 验证失败: {e}")
        
        # 检查额外参数
        extra_params = set(params.keys()) - set(self.param_names)
        if extra_params:
            logger.warning(f"忽略额外参数: {extra_params}")
        
        # 返回结果
        is_valid = len(errors) == 0
        error_message = "; ".join(errors) if errors else "验证通过"
        
        return is_valid, error_message, cleaned_params
    
    def _clean_and_validate_param(self, name: str, value: Any) -> Union[int, float]:
        """清理和验证单个参数"""
        param_index = self.param_names.index(name)
        expected_type = self.param_types[param_index]
        low, high = self.bounds[param_index]
        
        # 转换numpy类型
        if hasattr(value, 'item'):
            value = value.item()
        
        # 类型转换
        try:
            if expected_type == int:
                cleaned_value: Union[int, float] = int(round(float(value)))
            else:
                cleaned_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"无法转换为{expected_type.__name__}: {value}")
        
        # 边界检查
        if not (low <= cleaned_value <= high):
            raise ValueError(f"值 {cleaned_value} 超出范围 [{low}, {high}]")
        
        return cleaned_value

class AnsaMeshEvaluator(MeshEvaluator):
    """Ansa网格评估器 - 改进版本"""
    
    def __init__(self):
        self.config = config_manager.ansa_config
        self.param_mapping = config_manager.parameter_space.get_ansa_mapping()
        self.validator = ParameterValidator(config_manager.parameter_space)
        self._validate_environment()
    
    def _has_rule_fillet_parameters(self, params: Dict[str, float]) -> bool:
        """检查参数中是否包含 rule_fillet_width 参数"""
        return any(key.startswith('rule_fillet_width_') for key in params.keys())
    
    def _has_recognize_chamfers_parameters(self, params: Dict[str, float]) -> bool:
        """检查参数中是否包含 recognize_chamfers 参数"""
        return any(key.startswith('recognize_chamfers_') for key in params.keys())
    
    def _has_rule_chamfer_parameters(self, params: Dict[str, float]) -> bool:
        """检查参数中是否包含 rule_chamfer_width 参数"""
        return any(key.startswith('rule_chamfer_width_') for key in params.keys())
    
    def _has_distortion_angle_parameter(self, params: Dict[str, float]) -> bool:
        """检查参数中是否包含 distortion_angle 参数"""
        return 'distortion_angle' in params
    
    def _has_perimeter_distance_parameter(self, params: Dict[str, float]) -> bool:
        """检查参数中是否包含 perimeter_distance 参数"""
        return 'perimeter_distance' in params
    
    def _validate_environment(self) -> None:
        """验证Ansa环境"""
        try:
            # 检查Ansa可执行文件
            result = subprocess.run(
                [self.config.ansa_executable, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Ansa可执行文件无法运行")
            
            logger.info("Ansa环境验证成功")
            
        except subprocess.TimeoutExpired:
            logger.warning("Ansa版本检查超时")
        except FileNotFoundError:
            logger.warning("Ansa可执行文件未找到")
        except RuntimeError as e:
            logger.warning(f"Ansa环境验证失败: {e}")
        except Exception as e:
            logger.warning(f"Ansa环境验证异常: {e}")
        
        logger.info("将在需要时使用模拟模式")
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数有效性"""
        try:
            is_valid, _, _ = self.validator.validate_comprehensive(params)
            return is_valid
        except Exception as e:
            logger.error(f"参数验证异常: {e}")
            return False
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        评估网格质量 - 改进版本
        
        Args:
            params: 网格参数字典
            
        Returns:
            不合格网格单元数量
        """
        # 标准化和验证参数
        try:
            normalized_params = normalize_params(params)
            is_valid, error_msg, cleaned_params = self.validator.validate_comprehensive(normalized_params)
            
            if not is_valid:
                logger.error(f"参数验证失败: {error_msg}")
                return float('inf')
            
        except Exception as e:
            logger.error(f"参数处理失败: {e}")
            return float('inf')
        
        try:
            # 创建临时配置文件
            config_file = self._create_temp_config(cleaned_params)
            
            # 处理参数文件替换
            final_config_file = self._process_parameter_files(config_file, cleaned_params)
            
            # 运行Ansa批处理
            bad_elements_count = self._run_ansa_batch(final_config_file)
            
            logger.info(f"网格评估完成: {bad_elements_count} 个不合格单元")
            return float(bad_elements_count)
            
        except Exception as e:
            logger.error(f"网格评估失败: {e}")
            return float('inf')
        
        finally:
            # 清理临时文件
            self._cleanup_temp_files([
                locals().get('config_file'),
                locals().get('final_config_file')
            ])
    
    def _format_parameter_value(self, param_name: str, value: Any) -> str:
        """
        格式化参数值
        
        Args:
            param_name: 参数名
            value: 参数值
            
        Returns:
            格式化后的参数值字符串
        """
        # 定义需要特殊格式化的参数
        special_formatting = {
            'distortion_distance': lambda v: f"{v}.%",
            # 可以添加更多特殊格式化规则
            # 'another_param': lambda v: f"{v}mm",
        }
        
        if param_name in special_formatting:
            return special_formatting[param_name](value)
        else:
            return str(value)

    def _create_temp_config(self, params: Dict[str, float]) -> str:
        """创建临时配置文件"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for key, value in params.items():
                    formatted_value = self._format_parameter_value(key, value)
                    f.write(f"{key} = {formatted_value}\n")
                temp_file = f.name
            
            logger.debug(f"创建临时配置文件: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"创建临时配置文件失败: {e}")
            raise
    
    def _process_parameter_files(self, config_file: str, params: Dict[str, float]) -> str:
        """处理参数文件替换"""
        # 查找mpar文件
        mpar_files = list(Path('.').glob(self.config.mpar_file_pattern))
        
        if not mpar_files:
            logger.info("未找到mpar文件，使用原始配置")
            return config_file
        
        mpar_file = mpar_files[0]
        logger.info(f"使用mpar文件: {mpar_file}")
        
        try:
            # 检查是否有特殊参数需要处理
            has_fillet_params = self._has_rule_fillet_parameters(params)
            has_chamfers_params = self._has_recognize_chamfers_parameters(params)
            has_chamfer_params = self._has_rule_chamfer_parameters(params)
            has_distortion_angle_param = self._has_distortion_angle_parameter(params)
            has_perimeter_distance_param = self._has_perimeter_distance_parameter(params)
            
            if has_fillet_params:
                logger.info("检测到 rule_fillet_width 参数，进行特殊处理")
                # 先处理 rule_fillet 参数替换
                updated_mpar_file = self._replace_rule_fillet_parameters(str(mpar_file), params)
                # 使用更新后的 mpar 文件
                mpar_file = Path(updated_mpar_file)
            
            if has_chamfers_params:
                logger.info("检测到 recognize_chamfers 参数，进行特殊处理")
                # 处理 recognize_chamfers 参数替换
                updated_mpar_file = self._replace_recognize_chamfers_parameters(str(mpar_file), params)
                # 使用更新后的 mpar 文件
                mpar_file = Path(updated_mpar_file)
            
            if has_chamfer_params:
                logger.info("检测到 rule_chamfer_width 参数，进行特殊处理")
                # 处理 rule_chamfer 参数替换
                updated_mpar_file = self._replace_rule_chamfer_parameters(str(mpar_file), params)
                # 使用更新后的 mpar 文件
                mpar_file = Path(updated_mpar_file)
            
            if has_distortion_angle_param:
                logger.info("检测到 distortion_angle 参数，进行特殊处理")
                # 处理 distortion_angle 参数替换
                updated_mpar_file = self._replace_distortion_angle_parameter(str(mpar_file), params)
                # 使用更新后的 mpar 文件
                mpar_file = Path(updated_mpar_file)
            
            if has_perimeter_distance_param:
                logger.info("检测到 perimeter_distance 参数，进行特殊处理")
                # 处理 perimeter_distance 参数替换
                updated_mpar_file = self._replace_perimeter_distance_parameter(str(mpar_file), params)
                # 使用更新后的 mpar 文件
                mpar_file = Path(updated_mpar_file)
            
            # 解析mpar参数
            mpar_params = self._parse_mpar_file(mpar_file)
            
            if not mpar_params:
                logger.warning("mpar文件解析为空，使用原始配置")
                return config_file
            
            # 替换其他参数
            final_config_file = self._replace_parameters(config_file, mpar_params)
            
            return final_config_file
            
        except Exception as e:
            logger.error(f"处理参数文件失败: {e}")
            return config_file
    
    def _parse_mpar_file(self, mpar_file: Path) -> Dict[str, str]:
        """解析mpar文件"""
        params = {}
        
        try:
            with open(mpar_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 跳过空行和注释行
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue
                    
                    if '=' in line:
                        try:
                            key, value = line.split('=', 1)
                            params[key.strip()] = value.strip()
                        except ValueError:
                            logger.warning(f"无法解析第{line_num}行: {line}")
            
            logger.info(f"成功解析{mpar_file}，提取{len(params)}个参数")
            return params
            
        except FileNotFoundError:
            logger.error(f"mpar文件不存在: {mpar_file}")
            return {}
        except UnicodeDecodeError:
            logger.error(f"mpar文件编码错误: {mpar_file}")
            return {}
        except Exception as e:
            logger.error(f"解析mpar文件失败: {e}")
            return {}
    
    def _replace_parameters(self, temp_file: str, mpar_params: Dict[str, str]) -> str:
        """根据映射关系替换参数值"""
        output_file = temp_file + "_updated"
        
        try:
            updated_lines = []
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line:
                        key, current_value = line.split('=', 1)
                        key = key.strip()
                        
                        # 查找映射关系
                        if key in self.param_mapping:
                            mpar_key = self.param_mapping[key]
                            if mpar_key in mpar_params:
                                updated_lines.append(f"{key} = {mpar_params[mpar_key]}\n")
                                continue
                    
                    updated_lines.append(line)
            
            # 写入更新后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            logger.info(f"参数替换完成，结果保存至: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"参数替换失败: {e}")
            return temp_file
    
    def _replace_rule_fillet_parameters(self, mpar_file_path: str, params: Dict[str, float]) -> str:
        """
        专门处理 rule_fillet 参数替换
        
        Args:
            mpar_file_path: mpar文件路径
            params: 包含 rule_fillet_width 参数的字典
            
        Returns:
            更新后的文件路径
        """
        output_file = mpar_file_path + "_fillet_updated"
        
        try:
            # 提取 rule_fillet_width 参数
            width_1 = params.get('rule_fillet_width_1', 3.0)
            width_2 = params.get('rule_fillet_width_2', 10.0)
            width_3 = params.get('rule_fillet_width_3', 20.0)
            width_4 = params.get('rule_fillet_width_4', 30.0)
            
            logger.info(f"应用 rule_fillet width 参数: {width_1}, {width_2}, {width_3}, {width_4}")
            
            # 读取并替换文件内容
            with open(mpar_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换 rule_fillet 行中的 width 值
            import re
            
            # 替换第一个 rule_fillet (width = 0-3 -> width = 0-width_1)
            pattern1 = r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*0-)(\d+(?:\.\d+)?)(.*?treatment\s*=\s*7)'
            replacement1 = f'\\g<1>{width_1}\\g<3>'
            content = re.sub(pattern1, replacement1, content)
            
            # 替换第二个 rule_fillet (width = 3-10 -> width = width_1-width_2)
            pattern2 = r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*8)'
            replacement2 = f'\\g<1>{width_1}-{width_2}\\g<4>'
            content = re.sub(pattern2, replacement2, content)
            
            # 替换第三个 rule_fillet (width = 10-20 -> width = width_2-width_3)
            pattern3 = r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*9)'
            replacement3 = f'\\g<1>{width_2}-{width_3}\\g<4>'
            content = re.sub(pattern3, replacement3, content)
            
            # 替换第四个 rule_fillet (width = 20-30 -> width = width_3-width_4)
            pattern4 = r'(rule_fillet\s*=\s*default\s*=\s*false.*?width\s*=\s*)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.*?treatment\s*=\s*10)'
            replacement4 = f'\\g<1>{width_3}-{width_4}\\g<4>'
            content = re.sub(pattern4, replacement4, content)
            
            # 写入更新后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"rule_fillet 参数替换完成，结果保存至: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"rule_fillet 参数替换失败: {e}")
            return mpar_file_path
    
    def _replace_recognize_chamfers_parameters(self, mpar_file_path: str, params: Dict[str, float]) -> str:
        """
        专门处理 recognize_chamfers 参数替换
        
        Args:
            mpar_file_path: mpar文件路径
            params: 包含 recognize_chamfers 参数的字典
            
        Returns:
            更新后的文件路径
        """
        output_file = mpar_file_path + "_chamfers_updated"
        
        try:
            # 提取 recognize_chamfers 参数
            min_angle = params.get('recognize_chamfers_min_angle', 20.0)
            max_angle = params.get('recognize_chamfers_max_angle', 70.0)
            max_width = params.get('recognize_chamfers_max_width', 20.0)
            
            logger.info(f"应用 recognize_chamfers 参数: min_angle={min_angle}, max_angle={max_angle}, max_width={max_width}")
            
            # 读取并替换文件内容
            with open(mpar_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换 recognize_chamfers 行中的参数值
            import re
            
            # 替换 recognize_chamfers_min_angle
            pattern_min_angle = r'(recognize_chamfers_min_angle\s*=\s*)(\d+(?:\.\d+)?\.?)'
            replacement_min_angle = f'\\g<1>{min_angle}.'
            content = re.sub(pattern_min_angle, replacement_min_angle, content)
            
            # 替换 recognize_chamfers_max_angle
            pattern_max_angle = r'(recognize_chamfers_max_angle\s*=\s*)(\d+(?:\.\d+)?\.?)'
            replacement_max_angle = f'\\g<1>{max_angle}.'
            content = re.sub(pattern_max_angle, replacement_max_angle, content)
            
            # 替换 recognize_chamfers_max_width
            pattern_max_width = r'(recognize_chamfers_max_width\s*=\s*)(\d+(?:\.\d+)?\.?)'
            replacement_max_width = f'\\g<1>{max_width}.'
            content = re.sub(pattern_max_width, replacement_max_width, content)
            
            # 写入更新后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"recognize_chamfers 参数替换完成，结果保存至: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"recognize_chamfers 参数替换失败: {e}")
            return mpar_file_path
    
    def _replace_rule_chamfer_parameters(self, mpar_file_path: str, params: Dict[str, float]) -> str:
        """
        替换 .ansa_mpar 文件中的 rule_chamfer 参数
        
        处理复杂的 rule_chamfer 行格式：
        rule_chamfer = default = false || active = true || name = none || width = 0-10 || angle = 40-50 || treatment = 12
        
        Args:
            mpar_file_path: .ansa_mpar 文件路径
            params: 包含 rule_chamfer_width_1 参数的字典
            
        Returns:
            str: 更新后的文件路径
        """
        try:
            # 读取原始文件
            with open(mpar_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取 rule_chamfer_width_1 参数值
            chamfer_width_1 = params.get('rule_chamfer_width_1')
            if chamfer_width_1 is None:
                logger.warning("未找到 rule_chamfer_width_1 参数")
                return mpar_file_path
            
            # 转换为整数（根据原始格式，宽度值应该是整数）
            chamfer_width_1_int = int(round(chamfer_width_1))
            
            # 定义正则表达式模式来匹配 rule_chamfer 行
            # 匹配: rule_chamfer = ... || width = 0-数字 || ... || treatment = 12
            pattern = r'(rule_chamfer\s*=.*?width\s*=\s*0-)(\d+)(.*?treatment\s*=\s*12)'
            
            # 查找匹配项
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if not matches:
                logger.warning("未找到匹配的 rule_chamfer 行格式")
                return mpar_file_path
            
            # 执行替换
            def replace_width(match):
                prefix = match.group(1)  # rule_chamfer = ... || width = 0-
                old_width = match.group(2)  # 原始宽度值
                suffix = match.group(3)   # ... || treatment = 12
                
                logger.info(f"替换 rule_chamfer 宽度值: {old_width} -> {chamfer_width_1_int}")
                return f"{prefix}{chamfer_width_1_int}{suffix}"
            
            # 执行替换
            updated_content = re.sub(pattern, replace_width, content, flags=re.IGNORECASE | re.DOTALL)
            
            # 生成新的文件名
            base_name = os.path.splitext(os.path.basename(mpar_file_path))[0]
            dir_name = os.path.dirname(mpar_file_path)
            updated_file_path = os.path.join(dir_name, f"{base_name}_chamfer_updated.ansa_mpar")
            
            # 写入更新后的内容
            with open(updated_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Rule chamfer 参数替换完成: {updated_file_path}")
            logger.info(f"rule_chamfer_width_1: {chamfer_width_1} -> {chamfer_width_1_int}")
            
            return updated_file_path
            
        except Exception as e:
            logger.error(f"Rule chamfer 参数替换失败: {e}")
            return mpar_file_path
    
    def _replace_distortion_angle_parameter(self, mpar_file_path: str, params: Dict[str, float]) -> str:
        """
        替换 .ansa_mpar 文件中的 distortion-angle 参数
        
        处理 distortion-angle 行格式：
        distortion-angle = 0.
        
        Args:
            mpar_file_path: .ansa_mpar 文件路径
            params: 包含 distortion_angle 参数的字典
            
        Returns:
            str: 更新后的文件路径
        """
        try:
            # 读取原始文件
            with open(mpar_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取 distortion_angle 参数值
            distortion_angle = params.get('distortion_angle')
            if distortion_angle is None:
                logger.warning("未找到 distortion_angle 参数")
                return mpar_file_path
            
            # 定义正则表达式模式来匹配 distortion-angle 行
            # 匹配: distortion-angle = 数字.
            pattern = r'(distortion-angle\s*=\s*)(\d+(?:\.\d+)?)(\.?)'
            
            # 查找匹配项
            matches = re.findall(pattern, content, re.IGNORECASE)
            if not matches:
                logger.warning("未找到匹配的 distortion-angle 行格式")
                return mpar_file_path
            
            # 执行替换
            def replace_angle(match):
                prefix = match.group(1)  # distortion-angle =
                old_angle = match.group(2)  # 原始角度值
                suffix = match.group(3)   # 可能的小数点
                
                logger.info(f"替换 distortion-angle 值: {old_angle} -> {distortion_angle}")
                return f"{prefix}{distortion_angle}."
            
            # 执行替换
            updated_content = re.sub(pattern, replace_angle, content, flags=re.IGNORECASE)
            
            # 生成新的文件名
            base_name = os.path.splitext(os.path.basename(mpar_file_path))[0]
            dir_name = os.path.dirname(mpar_file_path)
            updated_file_path = os.path.join(dir_name, f"{base_name}_distortion_angle_updated.ansa_mpar")
            
            # 写入更新后的内容
            with open(updated_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Distortion angle 参数替换完成: {updated_file_path}")
            logger.info(f"distortion_angle: {distortion_angle}")
            
            return updated_file_path
            
        except Exception as e:
            logger.error(f"Distortion angle 参数替换失败: {e}")
            return mpar_file_path
    
    def _replace_perimeter_distance_parameter(self, mpar_file_path: str, params: Dict[str, float]) -> str:
        """
        替换 .ansa_mpar 文件中的 perimeter_distance 参数
        
        处理 remove_perimeters_with_distance 行格式：
        remove_perimeters_with_distance = 0.667*Lmin
        
        Args:
            mpar_file_path: .ansa_mpar 文件路径
            params: 包含 perimeter_distance 参数的字典
            
        Returns:
            str: 更新后的文件路径
        """
        try:
            # 读取原始文件
            with open(mpar_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取 perimeter_distance 参数值
            perimeter_distance = params.get('perimeter_distance')
            if perimeter_distance is None:
                logger.warning("未找到 perimeter_distance 参数")
                return mpar_file_path
            
            # 定义正则表达式模式来匹配 remove_perimeters_with_distance 行
            # 匹配: remove_perimeters_with_distance = 数字*Lmin
            pattern = r'(remove_perimeters_with_distance\s*=\s*)(\d+(?:\.\d+)?)(\*Lmin)'
            
            # 查找匹配项
            matches = re.findall(pattern, content, re.IGNORECASE)
            if not matches:
                logger.warning("未找到匹配的 remove_perimeters_with_distance 行格式")
                return mpar_file_path
            
            # 执行替换
            def replace_distance(match):
                prefix = match.group(1)  # remove_perimeters_with_distance =
                old_distance = match.group(2)  # 原始距离值
                suffix = match.group(3)   # *Lmin
                
                logger.info(f"替换 perimeter_distance 值: {old_distance} -> {perimeter_distance}")
                return f"{prefix}{perimeter_distance}{suffix}"
            
            # 执行替换
            updated_content = re.sub(pattern, replace_distance, content, flags=re.IGNORECASE)
            
            # 生成新的文件名
            base_name = os.path.splitext(os.path.basename(mpar_file_path))[0]
            dir_name = os.path.dirname(mpar_file_path)
            updated_file_path = os.path.join(dir_name, f"{base_name}_perimeter_distance_updated.ansa_mpar")
            
            # 写入更新后的内容
            with open(updated_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Perimeter distance 参数替换完成: {updated_file_path}")
            logger.info(f"perimeter_distance: {perimeter_distance}")
            
            return updated_file_path
            
        except Exception as e:
            logger.error(f"Perimeter distance 参数替换失败: {e}")
            return mpar_file_path
    
    def _run_ansa_batch(self, config_file: str) -> float:
        """运行Ansa批处理 - 改进的错误处理"""
        try:
            # 构建Ansa命令
            ansa_command = [
                self.config.ansa_executable,
                '-b',
                '-exec', str(self.config.script_dir / self.config.batch_script),
                '-i', self.config.input_model
            ]
            
            logger.info(f"执行Ansa命令: {' '.join(ansa_command)}")
            
            # 验证输入文件存在
            input_model_path = Path(self.config.input_model)
            if not input_model_path.exists():
                logger.error(f"输入模型文件不存在: {input_model_path}")
                return self._simulate_evaluation()
            
            # 验证批处理脚本存在
            batch_script_path = self.config.script_dir / self.config.batch_script
            if not batch_script_path.exists():
                logger.error(f"批处理脚本不存在: {batch_script_path}")
                return self._simulate_evaluation()
            
            # 执行命令
            result = subprocess.run(
                ansa_command,
                capture_output=True,
                text=True,
                timeout=self.config.execution_timeout,
                cwd=self.config.script_dir
            )
            
            # 具体的错误处理
            if result.returncode != 0:
                if result.returncode == 1:
                    logger.warning("Ansa返回代码1 - 可能有警告但继续执行")
                    # 尝试解析输出
                    return self._parse_ansa_output(result.stdout)
                elif result.returncode == 2:
                    logger.error("Ansa返回代码2 - 致命错误")
                    logger.error(f"错误输出: {result.stderr}")
                    return self._simulate_evaluation()
                else:
                    logger.error(f"Ansa执行失败，返回代码: {result.returncode}")
                    logger.error(f"错误输出: {result.stderr}")
                    return self._simulate_evaluation()
            
            # 解析成功的输出
            bad_elements_count = self._parse_ansa_output(result.stdout)
            return bad_elements_count
            
        except subprocess.TimeoutExpired:
            logger.error(f"Ansa执行超时({self.config.execution_timeout}秒)")
            return self._simulate_evaluation()
        except FileNotFoundError:
            logger.error(f"Ansa可执行文件未找到: {self.config.ansa_executable}")
            logger.info("使用模拟模式")
            return self._simulate_evaluation()
        except PermissionError:
            logger.error("没有权限执行Ansa")
            return self._simulate_evaluation()
        except Exception as e:
            logger.exception(f"Ansa执行时发生意外错误: {e}")
            return self._simulate_evaluation()
    
    def _parse_ansa_output(self, output: str) -> float:
        """解析Ansa输出 - 增强版本"""
        try:
            # 查找不合格网格数量的多种模式
            patterns = [
                r'bad elements:\s*(\d+)',
                r'failed elements:\s*(\d+)',
                r'poor quality elements:\s*(\d+)',
                r'质量不合格元素:\s*(\d+)',
                r'不合格单元:\s*(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    count = int(match.group(1))
                    logger.info(f"找到不合格网格数量: {count}")
                    return float(count)
            
            # 如果没有找到，尝试从最后几行提取数字
            lines = output.strip().split('\n')
            for line in reversed(lines[-10:]):  # 检查最后10行
                # 查找数字
                numbers = re.findall(r'\d+', line)
                if numbers:
                    # 取最大的数字（通常是元素数量）
                    max_number = max(int(n) for n in numbers)
                    if max_number > 0:
                        logger.info(f"从输出行解析得到数字: {max_number}")
                        return float(max_number)
            
            logger.warning("无法从输出中解析不合格网格数量")
            logger.debug(f"Ansa输出: {output}")
            return 99999.0
            
        except Exception as e:
            logger.error(f"解析Ansa输出失败: {e}")
            return 99999.0
    
    def _simulate_evaluation(self) -> float:
        """模拟评估（用于测试和备用）"""
        # 基于参数生成模拟结果
        base_score = random.uniform(50, 500)
        logger.info(f"使用模拟评估，返回结果: {base_score}")
        return base_score
    
    def _cleanup_temp_files(self, files: List[Optional[str]]) -> None:
        """清理临时文件"""
        for file_path in files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"删除临时文件: {file_path}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {e}")

class MockMeshEvaluator(MeshEvaluator):
    """模拟网格评估器（用于测试）- 改进版本"""
    
    def __init__(self, landscape_type: str = 'rosenbrock', add_noise: bool = True):
        self.landscape_type = landscape_type
        self.add_noise = add_noise
        self.evaluation_count = 0
        self.validator = ParameterValidator(config_manager.parameter_space)
        
        # 设置随机种子以便可重现
        random.seed(42)
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数有效性 - 修复版本"""
        try:
            is_valid, _, _ = self.validator.validate_comprehensive(params)
            return is_valid
        except Exception as e:
            logger.warning(f"Mock evaluator parameter validation failed: {e}")
            return False
        
            # 备用验证：检查基本必需参数
            required_params = [
                'element_size', 'perimeter_length', 'distortion_distance',
                'general_min_target_len', 'general_max_target_len'
            ]
            
            for param in required_params:
                if param not in params:
                    logger.error(f"Missing required parameter: {param}")
                    return False
            
            # 基本范围检查
            try:
                if not (0.1 <= params.get('element_size', 0) <= 10.0):
                    return False
                if not (0.1 <= params.get('perimeter_length', 0) <= 20.0):
                    return False
                if not (1 <= params.get('distortion_distance', 0) <= 100):
                    return False
                return True
            except (ValueError, TypeError):
                return False
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        模拟评估网格质量 - 增强版本
        
        使用多种数学函数模拟复杂的优化景观
        """
        self.evaluation_count += 1
        
        # 标准化参数
        try:
            normalized_params = normalize_params(params)
            is_valid, error_msg, cleaned_params = self.validator.validate_comprehensive(normalized_params)
            
            if not is_valid:
                logger.warning(f"Mock evaluator: invalid parameters - {error_msg}")
                # 对于测试，如果参数不完整，使用默认值
                cleaned_params = self._fill_missing_params(normalized_params)
            
        except Exception as e:
            logger.error(f"Mock evaluator parameter processing failed: {e}")
            # 使用默认参数
            cleaned_params = self._get_default_params()
        
        # 添加现实的延迟模拟
        if self.add_noise:
            time.sleep(random.uniform(0.05, 0.2))
        
        # 根据景观类型计算结果
        try:
            if self.landscape_type == 'rosenbrock':
                result = self._rosenbrock_function(cleaned_params)
            elif self.landscape_type == 'ackley':
                result = self._ackley_function(cleaned_params)
            elif self.landscape_type == 'rastrigin':
                result = self._rastrigin_function(cleaned_params)
            elif self.landscape_type == 'mesh_realistic':
                result = self._mesh_realistic_function(cleaned_params)
            else:
                result = self._rosenbrock_function(cleaned_params)
            
            # 添加噪声
            if self.add_noise:
                noise = random.gauss(0, 0.1 * result)
                result = max(0, result + noise)
            
            logger.debug(f"Mock evaluation #{self.evaluation_count}: {result:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"Mock evaluation function failed: {e}")
            return float('inf')
        
    def _fill_missing_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """填充缺失的参数"""
        default_params = self._get_default_params()

        # 使用提供的参数，缺失的用默认值填充
        filled_params = default_params.copy()
        filled_params.update(params)

        return filled_params
    
    def _get_default_params(self) -> Dict[str, float]:
        """获取默认参数"""
        return {
            'element_size': 1.0,
            'perimeter_length': 2.0,
            'distortion_distance': 25,
            'general_min_target_len': 1.5,
            'general_max_target_len': 9.0,
            'mesh_density': 4.0,
            'mesh_quality_threshold': 0.5,
            'smoothing_iterations': 50,
            'mesh_growth_rate': 1.0,
            'mesh_topology': 2
        }
    
    def _rosenbrock_function(self, params: Dict[str, float]) -> float:
        """Rosenbrock函数的变形（适合网格优化）"""
        x1 = params['element_size']
        x2 = params['perimeter_length']
        x3 = params['mesh_quality_threshold']
        
        # 标准化到[-2, 2]范围
        x1_norm = (x1 - 1.25) * 2.0  # element_size center at 1.25
        x2_norm = (x2 - 4.25) / 2.0  # perimeter_length center at 4.25
        x3_norm = (x3 - 0.6) * 10.0  # mesh_quality_threshold center at 0.6
        
        result = (
            100 * (x2_norm - x1_norm**2)**2 + 
            (1 - x1_norm)**2 +
            50 * (x3_norm - 0.5)**2 +
            params.get('smoothing_iterations', 50) / 20
        )
        
        return max(0, result)
    
    def _ackley_function(self, params: Dict[str, float]) -> float:
        """Ackley函数（多峰值景观）"""
        import math
        
        x1 = params['element_size']
        x2 = params['perimeter_length']
        x3 = params['mesh_quality_threshold']
        
        # 标准化
        x = [x1 - 1.25, x2 - 4.25, x3 - 0.6]
        n = len(x)
        
        sum_sq = sum(xi**2 for xi in x)
        sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x)
        
        result = (
            -20 * math.exp(-0.2 * math.sqrt(sum_sq / n)) -
            math.exp(sum_cos / n) + 20 + math.e
        ) * 10
        
        return max(0, result)
    
    def _rastrigin_function(self, params: Dict[str, float]) -> float:
        """Rastrigin函数（高度多峰值）"""
        import math
        
        x1 = params['element_size'] - 1.25
        x2 = params['perimeter_length'] - 4.25
        x3 = params['mesh_quality_threshold'] - 0.6
        
        A = 10
        result = A * 3 + sum(
            xi**2 - A * math.cos(2 * math.pi * xi)
            for xi in [x1, x2, x3]
        )
        
        return max(0, result * 5)
    
    def _mesh_realistic_function(self, params: Dict[str, float]) -> float:
        """模拟真实网格优化函数"""
        x1 = params['element_size']
        x2 = params['perimeter_length']
        x3 = params['mesh_quality_threshold']
        x4 = params.get('smoothing_iterations', 50)
        x5 = params.get('mesh_growth_rate', 1.0)
        
        # 模拟网格质量与参数的非线性关系
        # 元素尺寸太小或太大都不好
        size_penalty = abs(x1 - 1.0)**2 * 100
        
        # 网格密度的最优区间
        density_penalty = max(0, (x2 - 6.0)**2 - 4.0) * 50
        
        # 质量阈值的影响
        quality_penalty = (1.0 - x3)**2 * 200
        
        # 平滑迭代次数的边际效应
        smooth_effect = max(0, 100 - x4) + max(0, x4 - 80) * 2
        
        # 增长率的非线性效应
        growth_penalty = abs(x5 - 1.0)**1.5 * 150
        
        result = (
            size_penalty + 
            density_penalty + 
            quality_penalty + 
            smooth_effect + 
            growth_penalty +
            random.uniform(10, 50)  # 基础偏移
        )
        
        return float(max(1, result))
    
    def get_optimal_params(self) -> Dict[str, float]:
        """获取当前景观的最优参数（用于测试）"""
        optimal_params = {
            'rosenbrock': {
                'element_size': 1.25,
                'perimeter_length': 4.25,
                'mesh_quality_threshold': 0.6,
                'smoothing_iterations': 50,
                'mesh_growth_rate': 1.0,
                'mesh_topology': 2
            },
            'ackley': {
                'element_size': 1.25,
                'perimeter_length': 4.25,
                'mesh_quality_threshold': 0.6,
                'smoothing_iterations': 40,
                'mesh_growth_rate': 1.0,
                'mesh_topology': 2
            },
            'mesh_realistic': {
                'element_size': 1.0,
                'perimeter_length': 6.0,
                'mesh_quality_threshold': 1.0,
                'smoothing_iterations': 60,
                'mesh_growth_rate': 1.0,
                'mesh_topology': 2
            }
        }
        
        return optimal_params.get(self.landscape_type, optimal_params['rosenbrock'])

def create_mesh_evaluator(evaluator_type: str = 'ansa') -> MeshEvaluator:
    """
    创建网格评估器
    
    Args:
        evaluator_type: 评估器类型 ('ansa' 或 'mock')
        
    Returns:
        网格评估器实例
    """
    if evaluator_type.lower() == 'ansa':
        return AnsaMeshEvaluator()
    elif evaluator_type.lower() == 'mock':
        return MockMeshEvaluator()
    elif evaluator_type.lower().startswith('mock_'):
        # 支持不同的mock类型，例如 'mock_ackley'
        landscape = evaluator_type[5:]  # 去掉 'mock_' 前缀
        return MockMeshEvaluator(landscape_type=landscape)
    else:
        raise ValueError(f"不支持的评估器类型: {evaluator_type}")

# 工具函数
def test_evaluator(evaluator: MeshEvaluator, n_tests: int = 5) -> None:
    """测试评估器功能"""
    print(f"Testing {evaluator.__class__.__name__}...")
    
    # 获取最优参数（如果是Mock评估器）
    if hasattr(evaluator, 'get_optimal_params'):
        test_params = evaluator.get_optimal_params()
    else:
        # 使用默认测试参数
        test_params = {
            'element_size': 1.0,
            'perimeter_length': 2.0,
            'mesh_quality_threshold': 0.5,
            'smoothing_iterations': 40,
            'mesh_growth_rate': 1.0,
            'mesh_topology': 2
        }
    
    print(f"Test parameters: {test_params}")
    print(f"Parameter validation: {evaluator.validate_params(test_params)}")
    
    results = []
    for i in range(n_tests):
        result = evaluator.evaluate_mesh(test_params)
        results.append(result)
        print(f"Test {i+1}: {result:.6f}")
    
    if results:
        avg_result = sum(results) / len(results)
        print(f"Average result: {avg_result:.6f}")
    
    print("Testing completed!\n")

if __name__ == "__main__":
    # 测试评估器
    print("=== Mesh Evaluator Testing ===")
    
    # 测试Mock评估器
    mock_evaluator = create_mesh_evaluator('mock')
    test_evaluator(mock_evaluator, n_tests=3)
    
    # 测试不同景观的Mock评估器
    for landscape in ['ackley', 'mesh_realistic']:
        mock_eval = create_mesh_evaluator(f'mock_{landscape}')
        test_evaluator(mock_eval, n_tests=2)
    
    # 测试Ansa评估器（如果可用）
    try:
        ansa_evaluator = create_mesh_evaluator('ansa')
        print("Ansa evaluator created successfully")
        print(f"Parameter validation test: {ansa_evaluator.validate_params({'element_size': 1.0, 'mesh_density': 2.0, 'mesh_quality_threshold': 0.5})}")
    except Exception as e:
        print(f"Ansa evaluator test skipped: {e}")
    
    print("All evaluator tests completed!")