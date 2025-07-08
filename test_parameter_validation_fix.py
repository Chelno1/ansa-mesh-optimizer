#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试参数验证修复
验证部分参数更新是否正常工作
"""

import logging
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluators.mesh_evaluator import AnsaMeshEvaluator, MockMeshEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_partial_parameter_validation():
    """测试部分参数验证"""
    logger.info("=== 测试部分参数验证修复 ===")
    
    # 测试参数 - 只包含部分参数
    partial_params = {
        'element_size': 1.5,
        'perimeter_length': 6.0,
        'rule_fillet_width_1': 3.0,
        'rule_fillet_width_2': 8.0,
        'recognize_chamfers_min_angle': 25.0,
        'distortion_angle': 10.0
    }
    
    logger.info(f"测试参数（部分）: {partial_params}")
    logger.info(f"参数数量: {len(partial_params)}")
    
    try:
        # 测试 AnsaMeshEvaluator
        logger.info("\n--- 测试 AnsaMeshEvaluator ---")
        ansa_evaluator = AnsaMeshEvaluator()
        
        # 测试参数验证
        is_valid = ansa_evaluator.validate_params(partial_params)
        logger.info(f"AnsaMeshEvaluator 参数验证结果: {is_valid}")
        
        if is_valid:
            logger.info("✅ AnsaMeshEvaluator 参数验证成功！")
            
            # 测试评估
            try:
                result = ansa_evaluator.evaluate_mesh(partial_params)
                logger.info(f"AnsaMeshEvaluator 评估结果: {result}")
                logger.info("✅ AnsaMeshEvaluator 评估成功！")
            except Exception as e:
                logger.warning(f"AnsaMeshEvaluator 评估失败（预期，因为ANSA不可用）: {e}")
        else:
            logger.error("❌ AnsaMeshEvaluator 参数验证失败")
            
    except Exception as e:
        logger.error(f"❌ AnsaMeshEvaluator 测试失败: {e}")
    
    try:
        # 测试 MockMeshEvaluator
        logger.info("\n--- 测试 MockMeshEvaluator ---")
        mock_evaluator = MockMeshEvaluator()
        
        # 测试参数验证
        is_valid = mock_evaluator.validate_params(partial_params)
        logger.info(f"MockMeshEvaluator 参数验证结果: {is_valid}")
        
        if is_valid:
            logger.info("✅ MockMeshEvaluator 参数验证成功！")
            
            # 测试评估
            result = mock_evaluator.evaluate_mesh(partial_params)
            logger.info(f"MockMeshEvaluator 评估结果: {result}")
            logger.info("✅ MockMeshEvaluator 评估成功！")
        else:
            logger.error("❌ MockMeshEvaluator 参数验证失败")
            
    except Exception as e:
        logger.error(f"❌ MockMeshEvaluator 测试失败: {e}")

def test_full_parameter_validation():
    """测试完整参数验证（确保向后兼容）"""
    logger.info("\n=== 测试完整参数验证（向后兼容性） ===")
    
    # 完整参数集
    full_params = {
        'element_size': 1.0,
        'perimeter_length': 2.0,
        'min_target_length': 1.5,
        'max_target_length': 9.0,
        'distortion_distance': 20,
        'quality_threshold': 0.6,
        'smoothing_iterations': 50,
        'rule_fillet_width_1': 3.0,
        'rule_fillet_width_2': 10.0,
        'rule_fillet_width_3': 20.0,
        'rule_fillet_width_4': 30.0,
        'recognize_chamfers_min_angle': 20.0,
        'recognize_chamfers_max_angle': 70.0,
        'recognize_chamfers_max_width': 20.0,
        'rule_chamfer_width_1': 10.0,
        'distortion_angle': 0.0,
        'perimeter_distance': 0.667,
        'mesh_density': 5.0,
        'growth_rate': 1.0,
        'mesh_topology': 2
    }
    
    logger.info(f"完整参数数量: {len(full_params)}")
    
    try:
        mock_evaluator = MockMeshEvaluator()
        is_valid = mock_evaluator.validate_params(full_params)
        logger.info(f"完整参数验证结果: {is_valid}")
        
        if is_valid:
            result = mock_evaluator.evaluate_mesh(full_params)
            logger.info(f"完整参数评估结果: {result}")
            logger.info("✅ 完整参数测试成功！")
        else:
            logger.error("❌ 完整参数验证失败")
            
    except Exception as e:
        logger.error(f"❌ 完整参数测试失败: {e}")

def test_parameter_flow_integration():
    """测试参数流向集成"""
    logger.info("\n=== 测试参数流向集成 ===")
    
    # 模拟用户配置中的部分参数
    user_config_params = {
        'element_size': 2.0,
        'rule_fillet_width_1': 4.0,
        'rule_fillet_width_2': 12.0,
        'recognize_chamfers_min_angle': 30.0,
        'distortion_angle': 15.0
    }
    
    logger.info(f"用户配置参数: {user_config_params}")
    
    try:
        # 测试参数替换策略
        from src.evaluators.parameter_replacement_strategies import ParameterReplacementManager
        
        replacement_manager = ParameterReplacementManager()
        
        # 查找mpar文件
        import glob
        mpar_files = glob.glob("data/mesh/*.ansa_mpar")
        
        if mpar_files:
            mpar_file = mpar_files[0]
            logger.info(f"使用mpar文件: {mpar_file}")
            
            # 应用参数替换
            updated_file = replacement_manager.process_parameter_replacements(mpar_file, user_config_params)
            logger.info(f"参数替换完成: {updated_file}")
            
            # 验证文件是否创建
            if os.path.exists(updated_file):
                logger.info("✅ 参数替换文件创建成功！")
                
                # 检查文件内容变化
                with open(mpar_file, 'r') as f:
                    original_content = f.read()
                
                with open(updated_file, 'r') as f:
                    updated_content = f.read()
                
                if original_content != updated_content:
                    logger.info("✅ 参数替换确实修改了文件内容！")
                    
                    # 计算差异行数
                    original_lines = original_content.split('\n')
                    updated_lines = updated_content.split('\n')
                    
                    differences = 0
                    for i, (orig, upd) in enumerate(zip(original_lines, updated_lines)):
                        if orig != upd:
                            differences += 1
                            logger.info(f"  差异第{i+1}行: '{orig}' -> '{upd}'")
                    
                    logger.info(f"总共发现 {differences} 处差异")
                else:
                    logger.warning("⚠️ 参数替换没有修改文件内容")
            else:
                logger.error("❌ 参数替换文件未创建")
        else:
            logger.warning("未找到mpar文件，跳过参数替换测试")
            
    except Exception as e:
        logger.error(f"❌ 参数流向集成测试失败: {e}")

if __name__ == "__main__":
    logger.info("开始参数验证修复测试...")
    
    test_partial_parameter_validation()
    test_full_parameter_validation()
    test_parameter_flow_integration()
    
    logger.info("\n参数验证修复测试完成！")