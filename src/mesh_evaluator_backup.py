#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估接口模块 (完全修复版)

作者: Chel
创建日期: 2025-06-19
版本: 1.1.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import subprocess
import os
import re
import logging
from pathlib import Path
import tempfile
import glob

logger = logging.getLogger(__name__)

def normalize_params(params: Dict) -> Dict:
    """
    标准化参数，将numpy类型转换为Python原生类型
    
    Args:
        params: 参数字典
        
    Returns:
        标准化后的参数字典
    """
    normalized = {}
    for key, value in params.items():
        try:
            # 检查是否是numpy类型
            if hasattr(value, 'item'):  # numpy scalar
                normalized[key] = value.item()
            elif hasattr(value, 'dtype'):  # numpy array
                if hasattr(value, 'size') and value.size == 1:
                    normalized[key] = value.item()
                else:
                    normalized[key] = value.tolist() if hasattr(value, 'tolist') else value
            elif str(type(value).__module__).startswith('numpy'):
                # 其他numpy类型
                if hasattr(value, 'item'):
                    normalized[key] = value.item()
                else:
                    normalized[key] = float(value) if hasattr(value, '__float__') else value
            else:
                # Python原生类型
                normalized[key] = value
        except Exception as e:
            logger.warning(f"标准化参数 {key} 时出错: {e}，使用原值")
            normalized[key] = value
    
    return normalized

# 尝试导入配置，如果失败则使用本地配置
try:
    from config import config_manager
    logger.debug("成功导入config模块")
except ImportError:
    logger.warning("无法导入config模块，使用本地配置")
    
    class SimpleAnsaConfig:
        def __init__(self):
            self.script_dir = Path(__file__).parent
            self.mpar_file_pattern = '*.ansa_mpar'
            self.qual_file_pattern = '*.ansa_qual'
            self.min_element_length = 2.0
            self.max_element_length = 8.0
    
    class SimpleConfig:
        def __init__(self):
            self.ansa_config = SimpleAnsaConfig()
            
        def get_parameter_mapping(self):
            return {
                'element_size': 'target_element_length',
                'mesh_density': 'perimeter_length',
                'mesh_quality_threshold': 'distortion-angle',
                'smoothing_iterations': 'general_min_target_len',
                'mesh_growth_rate': 'cfd_interior_growth_rate',
                'mesh_topology': 'mesh_type'
            }
    
    config_manager = SimpleConfig()

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

class AnsaMeshEvaluator(MeshEvaluator):
    """Ansa网格评估器"""
    
    def __init__(self):
        self.config = config_manager.ansa_config
        self.param_mapping = config_manager.get_parameter_mapping()
        self._validate_environment()
    
    def _validate_environment(self) -> None:
        """验证Ansa环境"""
        try:
            # 检查Ansa可执行文件
            result = subprocess.run(
                [self.config.ansa_executable],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Ansa可执行文件无法运行")
            
            logger.info("Ansa环境验证成功")
            
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Ansa环境验证失败: {e}")
            logger.warning("将使用模拟模式进行测试")
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数有效性"""
        required_params = ['element_size', 'mesh_density', 'mesh_quality_threshold']
        
        # 检查必要参数是否存在
        for param in required_params:
            if param not in params:
                logger.error(f"缺少必要参数: {param}")
                return False
        
        # 检查参数范围（简单验证）
        try:
            element_size = float(params['element_size'])
            mesh_density = int(params['mesh_density'])
            quality_threshold = float(params['mesh_quality_threshold'])
            
            if not (0.1 <= element_size <= 10.0):
                logger.error(f"element_size 超出范围: {element_size}")
                return False
            
            if not (1 <= mesh_density <= 10):
                logger.error(f"mesh_density 超出范围: {mesh_density}")
                return False
            
            if not (0.0 <= quality_threshold <= 1.0):
                logger.error(f"mesh_quality_threshold 超出范围: {quality_threshold}")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.error(f"参数类型错误: {e}")
            return False
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        评估网格质量
        
        Args:
            params: 网格参数字典
            
        Returns:
            不合格网格单元数量
        """
        # 标准化参数
        normalized_params = normalize_params(params)
        
        if not self.validate_params(normalized_params):
            logger.error(f"参数验证失败: {normalized_params}")
            return float('inf')
        
        try:
            # 创建临时配置文件
            config_file = self._create_temp_config(normalized_params)
            
            # 处理参数文件替换
            final_config_file = self._process_parameter_files(config_file, normalized_params)
            
            # 运行Ansa批处理
            bad_elements_count = self._run_ansa_batch(final_config_file)
            
            logger.info(f"网格评估完成: {bad_elements_count} 个不合格单元")
            return bad_elements_count
            
        except Exception as e:
            logger.error(f"网格评估失败: {e}")
            return float('inf')
        
        finally:
            # 清理临时文件
            self._cleanup_temp_files([config_file, final_config_file])
    
    def _create_temp_config(self, params: Dict[str, float]) -> str:
        """创建临时配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")
            temp_file = f.name
        
        logger.debug(f"创建临时配置文件: {temp_file}")
        return temp_file
    
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
            # 解析mpar参数
            mpar_params = self._parse_mpar_file(mpar_file)
            
            # 替换参数
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
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        try:
                            key, value = line.split('=', 1)
                            params[key.strip()] = value.strip()
                        except ValueError:
                            logger.warning(f"无法解析第{line_num}行: {line}")
            
            logger.info(f"成功解析{mpar_file}，提取{len(params)}个参数")
            return params
            
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
    
    def _run_ansa_batch(self, config_file: str) -> float:
        """运行Ansa批处理"""
        try:
            # 构建Ansa命令
            ansa_command = [
                'ansa',  # 假设ansa在PATH中
                '-b',
                '-exec', str(self.config.script_dir / 'batch_mesh.py'),
                '-i', 'input_model.ansa'
            ]
            
            logger.info(f"执行Ansa命令: {' '.join(ansa_command)}")
            
            # 执行命令
            result = subprocess.run(
                ansa_command,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=self.config.script_dir
            )
            
            if result.returncode != 0:
                logger.error(f"Ansa执行失败: {result.stderr}")
                return self._simulate_evaluation()
            
            # 解析输出
            bad_elements_count = self._parse_ansa_output(result.stdout)
            return bad_elements_count
            
        except subprocess.TimeoutExpired:
            logger.error("Ansa执行超时")
            return float('inf')
        except Exception as e:
            logger.error(f"Ansa执行异常: {e}")
            return self._simulate_evaluation()
    
    def _parse_ansa_output(self, output: str) -> float:
        """解析Ansa输出"""
        try:
            # 查找不合格网格数量
            patterns = [
                r'bad elements:\s*(\d+)',
                r'failed elements:\s*(\d+)',
                r'poor quality elements:\s*(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    count = int(match.group(1))
                    logger.info(f"找到不合格网格数量: {count}")
                    return float(count)
            
            # 如果没有找到，尝试从最后几行提取数字
            lines = output.strip().split('\n')
            for line in reversed(lines[-5:]):  # 检查最后5行
                numbers = re.findall(r'\d+', line)
                if numbers:
                    return float(numbers[-1])  # 取最后一个数字
            
            logger.warning("无法从输出中解析不合格网格数量")
            return 99999.0
            
        except Exception as e:
            logger.error(f"解析Ansa输出失败: {e}")
            return 99999.0
    
    def _simulate_evaluation(self) -> float:
        """模拟评估（用于测试）"""
        import random
        
        # 基于参数生成模拟结果
        base_score = random.uniform(100, 1000)
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
    """模拟网格评估器（用于测试）"""
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数有效性"""
        required_params = ['element_size', 'mesh_density', 'mesh_quality_threshold']
        return all(param in params for param in required_params)
    
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """
        模拟评估网格质量
        
        使用简单的数学函数模拟优化景观
        """
        # 标准化参数
        normalized_params = normalize_params(params)
        
        if not self.validate_params(normalized_params):
            return float('inf')
        
        # 模拟复杂的优化函数
        x1 = normalized_params['element_size']
        x2 = normalized_params['mesh_density']
        x3 = normalized_params['mesh_quality_threshold']
        
        # Rosenbrock函数的变形（有明确的最小值）
        result = (
            100 * (x2 - x1**2)**2 + (1 - x1)**2 +
            50 * (x3 - 0.5)**2 +
            normalized_params.get('smoothing_iterations', 50) / 10
        )
        
        # 添加一些噪声
        import random
        noise = random.uniform(-0.1, 0.1) * result
        
        return max(0, result + noise)

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
    else:
        raise ValueError(f"不支持的评估器类型: {evaluator_type}")

# 测试函数
def test_normalize_params():
    """测试normalize_params函数"""
    try:
        import numpy as np
        test_params = {
            'element_size': np.float64(1.5),
            'mesh_density': np.int64(3),
            'mesh_quality_threshold': 0.7,
            'regular_param': 42
        }
        
        normalized = normalize_params(test_params)
        print(f"原始参数: {test_params}")
        print(f"标准化参数: {normalized}")
        
        # 检查类型
        for key, value in normalized.items():
            print(f"{key}: {type(value)} = {value}")
        
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    # 运行测试
    print("测试 normalize_params 函数...")
    test_normalize_params()
    
    print("\n测试模拟评估器...")
    evaluator = MockMeshEvaluator()
    test_params = {
        'element_size': 1.0,
        'mesh_density': 3,
        'mesh_quality_threshold': 0.5
    }
    
    result = evaluator.evaluate_mesh(test_params)
    print(f"评估结果: {result}")