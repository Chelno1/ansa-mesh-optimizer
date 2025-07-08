#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数流向调试脚本
用于验证用户配置参数是否正确传递到优化器并更新mapr文件
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parameter_flow_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def debug_config_loading():
    """调试配置加载过程"""
    logger.info("=== 调试配置加载过程 ===")
    
    try:
        from src.config.config_refactored import unified_config_manager
        
        # 检查默认配置
        logger.info("默认配置参数空间:")
        param_names = unified_config_manager.parameter_space.get_parameter_names()
        logger.info(f"参数数量: {len(param_names)}")
        logger.info(f"参数列表: {param_names}")
        
        # 检查ANSA映射
        ansa_mapping = unified_config_manager.parameter_space.get_ansa_mapping()
        logger.info(f"ANSA映射数量: {len(ansa_mapping)}")
        for param, ansa_param in ansa_mapping.items():
            logger.info(f"  {param} -> {ansa_param}")
        
        # 检查默认值
        default_values = unified_config_manager.parameter_space.get_default_values()
        logger.info(f"默认值: {default_values}")
        
        return True
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return False

def debug_parameter_replacement():
    """调试参数替换策略"""
    logger.info("=== 调试参数替换策略 ===")
    
    try:
        from src.evaluators.parameter_replacement_strategies import ParameterReplacementManager
        
        # 创建替换管理器
        manager = ParameterReplacementManager()
        available_strategies = manager.get_available_strategies()
        logger.info(f"可用替换策略: {available_strategies}")
        
        # 测试参数
        test_params = {
            'rule_fillet_width_1': 3.5,
            'rule_fillet_width_2': 8.0,
            'recognize_chamfers_min_angle': 25.0,
            'distortion_angle': 15.0,
            'perimeter_distance': 0.8
        }
        
        logger.info(f"测试参数: {test_params}")
        
        # 检查每个策略是否能处理测试参数
        for strategy in manager.strategies:
            can_handle = strategy.can_handle(test_params)
            logger.info(f"策略 {strategy.get_strategy_name()} 可处理: {can_handle}")
        
        return True
        
    except Exception as e:
        logger.error(f"参数替换策略调试失败: {e}")
        return False

def debug_mpar_file_analysis():
    """调试mpar文件分析"""
    logger.info("=== 调试mpar文件分析 ===")
    
    mpar_files = [
        Path("data/mesh/8mm_V23.ansa_mpar"),
        Path("data/mesh/mend.ansa_mpar")
    ]
    
    for mpar_file in mpar_files:
        if not mpar_file.exists():
            logger.warning(f"mpar文件不存在: {mpar_file}")
            continue
            
        logger.info(f"分析文件: {mpar_file}")
        
        try:
            # 解析文件中的关键参数
            key_params = {}
            with open(mpar_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 检查关键参数
                        if any(keyword in key.lower() for keyword in [
                            'target_element_length', 'perimeter_length', 'distortion',
                            'rule_fillet', 'recognize_chamfers', 'remove_perimeters'
                        ]):
                            key_params[key] = value
                            logger.info(f"  第{line_num}行: {key} = {value}")
            
            logger.info(f"文件 {mpar_file.name} 中找到 {len(key_params)} 个关键参数")
            
        except Exception as e:
            logger.error(f"分析mpar文件失败: {e}")

def debug_evaluator_parameter_flow():
    """调试评估器中的参数流向"""
    logger.info("=== 调试评估器参数流向 ===")
    
    try:
        from src.evaluators.mesh_evaluator import AnsaMeshEvaluator, normalize_params
        
        # 创建评估器
        evaluator = AnsaMeshEvaluator()
        
        # 测试参数
        test_params = {
            'element_size': 1.5,
            'perimeter_length': 6.0,
            'distortion_distance': 25,
            'rule_fillet_width_1': 3.0,
            'rule_fillet_width_2': 8.0,
            'recognize_chamfers_min_angle': 20.0,
            'distortion_angle': 10.0,
            'perimeter_distance': 0.75
        }
        
        logger.info(f"原始测试参数: {test_params}")
        
        # 标准化参数
        normalized_params = normalize_params(test_params)
        logger.info(f"标准化后参数: {normalized_params}")
        
        # 验证参数
        try:
            is_valid = evaluator.validate_params(normalized_params)
            logger.info(f"参数验证结果: {is_valid}")
        except Exception as e:
            logger.error(f"参数验证失败: {e}")
        
        # 检查参数映射
        param_mapping = evaluator.param_mapping
        logger.info(f"评估器参数映射: {param_mapping}")
        
        # 模拟参数处理流程
        logger.info("模拟参数处理流程...")
        
        # 1. 创建临时配置文件
        temp_config = evaluator._create_temp_config(normalized_params)
        logger.info(f"临时配置文件: {temp_config}")
        
        # 读取临时配置文件内容
        if Path(temp_config).exists():
            with open(temp_config, 'r') as f:
                content = f.read()
                logger.info(f"临时配置文件内容:\n{content}")
        
        return True
        
    except Exception as e:
        logger.error(f"评估器参数流向调试失败: {e}")
        return False

def debug_parameter_replacement_execution():
    """调试参数替换执行过程"""
    logger.info("=== 调试参数替换执行过程 ===")
    
    # 使用实际的mpar文件进行测试
    mpar_file = Path("data/mesh/8mm_V23.ansa_mpar")
    if not mpar_file.exists():
        logger.error(f"测试mpar文件不存在: {mpar_file}")
        return False
    
    try:
        from src.evaluators.parameter_replacement_strategies import ParameterReplacementManager
        
        # 测试参数
        test_params = {
            'rule_fillet_width_1': 2.5,
            'rule_fillet_width_2': 7.5,
            'rule_fillet_width_3': 15.0,
            'rule_fillet_width_4': 25.0,
            'recognize_chamfers_min_angle': 25.0,
            'recognize_chamfers_max_angle': 75.0,
            'recognize_chamfers_max_width': 15.0,
            'rule_chamfer_width_1': 12.0,
            'distortion_angle': 5.0,
            'perimeter_distance': 0.8
        }
        
        logger.info(f"测试参数: {test_params}")
        
        # 创建替换管理器
        manager = ParameterReplacementManager()
        
        # 执行参数替换
        updated_file = manager.process_parameter_replacements(str(mpar_file), test_params)
        logger.info(f"更新后的文件: {updated_file}")
        
        # 检查更新后的文件是否存在
        if Path(updated_file).exists():
            logger.info("更新后的文件已创建")
            
            # 比较原文件和更新后文件的差异
            with open(mpar_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            with open(updated_file, 'r', encoding='utf-8') as f:
                updated_content = f.read()
            
            # 查找差异
            original_lines = original_content.split('\n')
            updated_lines = updated_content.split('\n')
            
            differences = []
            for i, (orig, upd) in enumerate(zip(original_lines, updated_lines)):
                if orig != upd:
                    differences.append(f"第{i+1}行: '{orig}' -> '{upd}'")
            
            if differences:
                logger.info(f"发现 {len(differences)} 处差异:")
                for diff in differences[:10]:  # 只显示前10个差异
                    logger.info(f"  {diff}")
            else:
                logger.warning("未发现任何差异 - 可能参数替换未生效")
        else:
            logger.error("更新后的文件未创建")
        
        return True
        
    except Exception as e:
        logger.error(f"参数替换执行调试失败: {e}")
        return False

def debug_ansa_subprocess():
    """调试ANSA子进程执行"""
    logger.info("=== 调试ANSA子进程执行 ===")
    
    try:
        from src.config.config_refactored import unified_config_manager
        
        ansa_config = unified_config_manager.ansa_config
        
        # 检查ANSA配置
        logger.info(f"ANSA可执行文件: {ansa_config.ansa_executable}")
        logger.info(f"脚本目录: {ansa_config.script_dir}")
        logger.info(f"批处理脚本: {ansa_config.batch_script}")
        logger.info(f"输入模型: {ansa_config.input_model}")
        logger.info(f"执行超时: {ansa_config.execution_timeout}")
        
        # 检查关键文件是否存在
        batch_script_path = ansa_config.script_dir / ansa_config.batch_script
        logger.info(f"批处理脚本路径: {batch_script_path}")
        logger.info(f"批处理脚本存在: {batch_script_path.exists()}")
        
        input_model_path = Path(ansa_config.input_model)
        logger.info(f"输入模型路径: {input_model_path}")
        logger.info(f"输入模型存在: {input_model_path.exists()}")
        
        # 检查ANSA可执行文件
        import subprocess
        try:
            result = subprocess.run(
                [ansa_config.ansa_executable, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            logger.info(f"ANSA版本检查返回码: {result.returncode}")
            if result.stdout:
                logger.info(f"ANSA版本输出: {result.stdout[:200]}")
            if result.stderr:
                logger.info(f"ANSA版本错误: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logger.warning("ANSA版本检查超时")
        except FileNotFoundError:
            logger.error("ANSA可执行文件未找到")
        except Exception as e:
            logger.error(f"ANSA版本检查失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"ANSA子进程调试失败: {e}")
        return False

def main():
    """主调试函数"""
    logger.info("开始参数流向调试...")
    
    # 执行各项调试检查
    checks = [
        ("配置加载", debug_config_loading),
        ("参数替换策略", debug_parameter_replacement),
        ("mpar文件分析", debug_mpar_file_analysis),
        ("评估器参数流向", debug_evaluator_parameter_flow),
        ("参数替换执行", debug_parameter_replacement_execution),
        ("ANSA子进程", debug_ansa_subprocess)
    ]
    
    results = {}
    for check_name, check_func in checks:
        logger.info(f"\n{'='*50}")
        logger.info(f"执行检查: {check_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = check_func()
            results[check_name] = result
            logger.info(f"检查 '{check_name}' 完成: {'成功' if result else '失败'}")
        except Exception as e:
            logger.error(f"检查 '{check_name}' 异常: {e}")
            results[check_name] = False
    
    # 总结结果
    logger.info(f"\n{'='*50}")
    logger.info("调试结果总结:")
    logger.info(f"{'='*50}")
    
    for check_name, result in results.items():
        status = "✓ 成功" if result else "✗ 失败"
        logger.info(f"{check_name}: {status}")
    
    # 生成诊断报告
    failed_checks = [name for name, result in results.items() if not result]
    if failed_checks:
        logger.error(f"\n发现问题的检查项: {', '.join(failed_checks)}")
        logger.error("建议检查这些模块的实现和配置")
    else:
        logger.info("\n所有检查项都通过了！")
    
    logger.info("参数流向调试完成")

if __name__ == "__main__":
    main()