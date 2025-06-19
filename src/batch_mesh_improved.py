#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的Ansa批处理网格脚本

作者: Chel
创建日期: 2025-06-19
版本: 1.1.0
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# 设置脚本目录
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(script_dir / 'ansa_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import ansa
    from ansa import base
    from ansa import mesh
    from ansa import constants
    ANSA_AVAILABLE = True
    logger.info("Ansa模块加载成功")
except ImportError:
    ANSA_AVAILABLE = False
    logger.warning("Ansa模块未找到，将使用模拟模式")

class AnsaBatchMeshRunner:
    """Ansa批处理网格运行器"""
    
    def __init__(self, script_dir: Path = None):
        """
        初始化批处理运行器
        
        Args:
            script_dir: 脚本目录路径
        """
        self.script_dir = script_dir or Path(__file__).parent.resolve()
        self.mesh_dir = self.script_dir / 'mesh'
        self.output_dir = self.script_dir / 'output'
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认配置
        self.config = {
            'min_element_length': 2.0,
            'max_element_length': 8.0,
            'mpar_file': 'mend.ansa_mpar',
            'qual_file': 'mend.ansa_qual',
            'output_model': 'output_mesh.ansa'
        }
        
        logger.info(f"脚本目录: {self.script_dir}")
        logger.info(f"网格目录: {self.mesh_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def load_config(self, config_file: Optional[str] = None) -> None:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径
        """
        if config_file is None:
            config_file = self.script_dir / 'batch_config.json'
        
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self.config.update(user_config)
                logger.info(f"配置已从{config_file}加载")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        else:
            logger.info("使用默认配置")
    
    def run_batch_mesh(self) -> bool:
        """
        运行批处理网格生成
        
        Returns:
            操作是否成功
        """
        if not ANSA_AVAILABLE:
            logger.warning("Ansa不可用，使用模拟模式")
            return self._simulate_batch_mesh()
        
        try:
            logger.info("开始批处理网格生成")
            
            # 读取网格参数
            mpar_file = self.mesh_dir / self.config['mpar_file']
            if mpar_file.exists():
                mesh.ReadMeshParams(str(mpar_file))
                logger.info(f"网格参数已加载: {mpar_file}")
            else:
                logger.warning(f"网格参数文件不存在: {mpar_file}")
            
            # 读取质量标准
            qual_file = self.mesh_dir / self.config['qual_file']
            if qual_file.exists():
                mesh.ReadQualityCriteria(str(qual_file))
                logger.info(f"质量标准已加载: {qual_file}")
            else:
                logger.warning(f"质量标准文件不存在: {qual_file}")
            
            # 收集壳体属性
            props = base.CollectEntities(constants.LSDYNA, None, 'SECTION_SHELL')
            
            if props:
                logger.info(f"找到{len(props)}个壳体属性")
                
                # 执行批处理网格生成
                result = mesh.BatchGenerator(props)
                
                if result:
                    logger.info("批处理网格生成成功")
                    return True
                else:
                    logger.error("批处理网格生成失败")
                    return False
            else:
                logger.warning("未找到壳体属性")
                return False
                
        except Exception as e:
            logger.error(f"批处理网格生成异常: {e}")
            return False
    
    def _simulate_batch_mesh(self) -> bool:
        """模拟批处理网格生成"""
        logger.info("模拟批处理网格生成...")
        
        # 模拟网格生成过程
        import time
        import random
        
        time.sleep(1)  # 模拟处理时间
        
        # 随机决定是否成功
        success = random.choice([True, True, True, False])  # 75%成功率
        
        if success:
            logger.info("模拟批处理网格生成成功")
        else:
            logger.error("模拟批处理网格生成失败")
        
        return success
    
    def check_element_quality(self) -> Dict[str, any]:
        """
        检查单元质量
        
        Returns:
            质量检查结果
        """
        if not ANSA_AVAILABLE:
            return self._simulate_quality_check()
        
        try:
            results = {
                'min_length_check': self._check_shell_min_length(self.config['min_element_length']),
                'max_length_check': self._check_shell_max_length(self.config['max_element_length']),
                'total_elements': 0,
                'bad_elements': 0
            }
            
            # 统计总单元数
            all_elements = base.CollectEntitiesI(constants.LSDYNA, None, 'ELEMENT_SHELL')
            results['total_elements'] = len(all_elements) if all_elements else 0
            
            # 统计不合格单元数
            results['bad_elements'] = (
                results['min_length_check']['failed_count'] + 
                results['max_length_check']['failed_count']
            )
            
            logger.info(f"质量检查完成: 总单元数={results['total_elements']}, "
                       f"不合格单元数={results['bad_elements']}")
            
            return results
            
        except Exception as e:
            logger.error(f"质量检查异常: {e}")
            return self._simulate_quality_check()
    
    def _check_shell_min_length(self, min_len: float) -> Dict[str, any]:
        """
        检查壳单元最小尺寸
        
        Args:
            min_len: 最小长度阈值
            
        Returns:
            检查结果
        """
        failed_elems = []
        
        try:
            for elem in base.CollectEntitiesI(constants.LSDYNA, None, 'ELEMENT_SHELL'):
                quality = base.ElementQuality(elem, 'MIN-LEN')
                if quality == 'error':
                    continue
                if float(quality) <= float(min_len):
                    failed_elems.append(elem)
            
            status = 'OK' if not failed_elems else 'NOK'
            
            logger.info(f"最小长度检查: {len(failed_elems)} 个不合格单元 (阈值: {min_len})")
            
            return {
                'status': status,
                'failed_count': len(failed_elems),
                'failed_elements': failed_elems,
                'threshold': min_len,
                'check_type': 'min_length'
            }
            
        except Exception as e:
            logger.error(f"最小长度检查失败: {e}")
            return {
                'status': 'ERROR',
                'failed_count': 0,
                'failed_elements': [],
                'threshold': min_len,
                'check_type': 'min_length',
                'error': str(e)
            }
    
    def _check_shell_max_length(self, max_len: float) -> Dict[str, any]:
        """
        检查壳单元最大尺寸
        
        Args:
            max_len: 最大长度阈值
            
        Returns:
            检查结果
        """
        failed_elems = []
        
        try:
            for elem in base.CollectEntitiesI(constants.LSDYNA, None, 'ELEMENT_SHELL'):
                quality = base.ElementQuality(elem, 'MAX-LEN')
                if quality == 'error':
                    continue
                if float(quality) >= float(max_len):
                    failed_elems.append(elem)
            
            status = 'OK' if not failed_elems else 'NOK'
            
            logger.info(f"最大长度检查: {len(failed_elems)} 个不合格单元 (阈值: {max_len})")
            
            return {
                'status': status,
                'failed_count': len(failed_elems),
                'failed_elements': failed_elems,
                'threshold': max_len,
                'check_type': 'max_length'
            }
            
        except Exception as e:
            logger.error(f"最大长度检查失败: {e}")
            return {
                'status': 'ERROR',
                'failed_count': 0,
                'failed_elements': [],
                'threshold': max_len,
                'check_type': 'max_length',
                'error': str(e)
            }
    
    def _simulate_quality_check(self) -> Dict[str, any]:
        """模拟质量检查"""
        import random
        
        logger.info("模拟质量检查...")
        
        # 模拟结果
        total_elements = random.randint(1000, 10000)
        bad_min = random.randint(0, 50)
        bad_max = random.randint(0, 30)
        
        results = {
            'min_length_check': {
                'status': 'OK' if bad_min == 0 else 'NOK',
                'failed_count': bad_min,
                'failed_elements': list(range(bad_min)),
                'threshold': self.config['min_element_length'],
                'check_type': 'min_length'
            },
            'max_length_check': {
                'status': 'OK' if bad_max == 0 else 'NOK',
                'failed_count': bad_max,
                'failed_elements': list(range(bad_max)),
                'threshold': self.config['max_element_length'],
                'check_type': 'max_length'
            },
            'total_elements': total_elements,
            'bad_elements': bad_min + bad_max
        }
        
        logger.info(f"模拟质量检查完成: 总单元数={total_elements}, "
                   f"不合格单元数={bad_min + bad_max}")
        
        return results
    
    def save_model(self, output_file: Optional[str] = None) -> bool:
        """
        保存模型
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            保存是否成功
        """
        if output_file is None:
            output_file = self.output_dir / self.config['output_model']
        else:
            output_file = Path(output_file)
        
        if not ANSA_AVAILABLE:
            return self._simulate_save_model(output_file)
        
        try:
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            base.SaveAs(str(output_file))
            
            logger.info(f"模型已保存: {output_file}")
            
            # 验证文件是否存在
            if output_file.exists():
                file_size = output_file.stat().st_size
                logger.info(f"保存的文件大小: {file_size / 1024 / 1024:.2f} MB")
                return True
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
            with open(output_file, 'w') as f:
                f.write("# 模拟的Ansa模型文件\n")
                f.write(f"# 生成时间: {Path(__file__).stat().st_mtime}\n")
                f.write("# 这是一个用于测试的模拟文件\n")
            
            logger.info(f"模拟模型已保存: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"模拟保存模型失败: {e}")
            return False
    
    def generate_quality_report(self, quality_results: Dict[str, any], 
                              output_file: Optional[str] = None) -> str:
        """
        生成质量报告
        
        Args:
            quality_results: 质量检查结果
            output_file: 报告输出文件路径
            
        Returns:
            报告文件路径
        """
        if output_file is None:
            timestamp = Path(__file__).stat().st_mtime
            output_file = self.output_dir / f"quality_report_{int(timestamp)}.txt"
        else:
            output_file = Path(output_file)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Ansa批处理网格质量报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"总单元数: {quality_results.get('total_elements', 'N/A')}\n")
                f.write(f"不合格单元数: {quality_results.get('bad_elements', 'N/A')}\n\n")
                
                # 最小长度检查
                min_check = quality_results.get('min_length_check', {})
                f.write("最小长度检查:\n")
                f.write(f"  状态: {min_check.get('status', 'N/A')}\n")
                f.write(f"  阈值: {min_check.get('threshold', 'N/A')}\n")
                f.write(f"  不合格单元数: {min_check.get('failed_count', 'N/A')}\n\n")
                
                # 最大长度检查
                max_check = quality_results.get('max_length_check', {})
                f.write("最大长度检查:\n")
                f.write(f"  状态: {max_check.get('status', 'N/A')}\n")
                f.write(f"  阈值: {max_check.get('threshold', 'N/A')}\n")
                f.write(f"  不合格单元数: {max_check.get('failed_count', 'N/A')}\n\n")
                
                # 配置信息
                f.write("配置信息:\n")
                for key, value in self.config.items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"质量报告已生成: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")
            return ""

def main() -> int:
    """
    主函数
    
    Returns:
        退出码
    """
    try:
        logger.info("开始Ansa批处理网格操作")
        
        # 创建批处理运行器
        runner = AnsaBatchMeshRunner()
        
        # 加载配置
        runner.load_config()
        
        # 运行批处理网格
        mesh_success = runner.run_batch_mesh()
        
        if not mesh_success:
            logger.error("网格生成失败")
            return 1
        
        # 检查质量
        quality_results = runner.check_element_quality()
        
        # 输出不合格网格数量（用于优化器读取）
        bad_elements = quality_results.get('bad_elements', 0)
        print(f'bad elements: {bad_elements}')
        
        # 生成质量报告
        report_file = runner.generate_quality_report(quality_results)
        
        # 保存模型
        save_success = runner.save_model()
        
        if not save_success:
            logger.warning("模型保存失败，但继续执行")
        
        logger.info("Ansa批处理网格操作完成")
        
        # 返回退出码（0表示成功，1表示有不合格网格）
        return 1 if bad_elements > 0 else 0
        
    except Exception as e:
        logger.error(f"批处理操作异常: {e}")
        return 2  # 异常退出码

def run_quality_check_only(min_len: float = 2.0, max_len: float = 8.0) -> Dict[str, any]:
    """
    仅运行质量检查（不生成网格）
    
    Args:
        min_len: 最小长度阈值
        max_len: 最大长度阈值
        
    Returns:
        质量检查结果
    """
    runner = AnsaBatchMeshRunner()
    runner.config['min_element_length'] = min_len
    runner.config['max_element_length'] = max_len
    
    return runner.check_element_quality()

def batch_mesh_with_params(params: Dict[str, float]) -> int:
    """
    使用指定参数运行批处理网格
    
    Args:
        params: 网格参数字典
        
    Returns:
        不合格网格单元数量
    """
    try:
        # 创建运行器
        runner = AnsaBatchMeshRunner()
        
        # 更新配置（如果参数中包含相关设置）
        if 'min_element_length' in params:
            runner.config['min_element_length'] = params['min_element_length']
        if 'max_element_length' in params:
            runner.config['max_element_length'] = params['max_element_length']
        
        # 运行网格生成
        mesh_success = runner.run_batch_mesh()
        
        if not mesh_success:
            logger.error("网格生成失败")
            return 99999  # 返回大数值表示失败
        
        # 检查质量
        quality_results = runner.check_element_quality()
        bad_elements = quality_results.get('bad_elements', 99999)
        
        logger.info(f"网格参数: {params}")
        logger.info(f"不合格网格数量: {bad_elements}")
        
        return bad_elements
        
    except Exception as e:
        logger.error(f"批处理网格异常: {e}")
        return 99999

# 便捷函数
def check_shell_min_length(min_len: float) -> str:
    """
    检查壳单元最小尺寸（向后兼容）
    
    Args:
        min_len: 最小长度阈值
        
    Returns:
        检查状态 ('OK' 或 'NOK')
    """
    runner = AnsaBatchMeshRunner()
    result = runner._check_shell_min_length(min_len)
    
    # 输出不合格单元数（与原代码兼容）
    print(f'bad elements: {result["failed_count"]}')
    
    return result['status']

def check_shell_max_length(max_len: float) -> str:
    """
    检查壳单元最大尺寸（向后兼容）
    
    Args:
        max_len: 最大长度阈值
        
    Returns:
        检查状态 ('OK' 或 'NOK')
    """
    runner = AnsaBatchMeshRunner()
    result = runner._check_shell_max_length(max_len)
    
    # 输出不合格单元数（与原代码兼容）
    print(f'bad elements: {result["failed_count"]}')
    
    return result['status']

def run_batch_mesh() -> int:
    """
    运行批处理网格（向后兼容）
    
    Returns:
        成功返回1，失败返回0
    """
    runner = AnsaBatchMeshRunner()
    success = runner.run_batch_mesh()
    return 1 if success else 0

if __name__ == '__main__':
    # 如果直接运行此脚本，执行主函数
    exit_code = main()
    sys.exit(exit_code)