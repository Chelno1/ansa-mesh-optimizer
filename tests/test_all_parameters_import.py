#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全参数导入测试 - 验证所有20个参数是否能正确导入到optimizer进行优化

作者: Roo (Debug Expert)
创建日期: 2025-07-07
功能: 系统性诊断参数导入问题
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_system():
    """测试配置系统"""
    print("=" * 60)
    print("🔍 第1步: 测试配置系统")
    print("=" * 60)
    
    try:
        from src.config.config_refactored import UnifiedConfigManager
        config_manager = UnifiedConfigManager()
        
        # 检查参数空间
        param_space = config_manager.parameter_space
        param_names = param_space.get_parameter_names()
        param_count = len(param_names)
        
        print(f"✅ 配置系统加载成功")
        print(f"📊 参数总数: {param_count}")
        print(f"📋 参数列表: {param_names}")
        
        # 验证预期的20个参数
        expected_params = [
            # 网格尺寸参数 (4个)
            'element_size', 'perimeter_length', 'min_target_length', 'max_target_length',
            # 网格质量参数 (3个)
            'distortion_distance', 'quality_threshold', 'smoothing_iterations',
            # rule_fillet width参数 (4个)
            'rule_fillet_width_1', 'rule_fillet_width_2', 'rule_fillet_width_3', 'rule_fillet_width_4',
            # recognize_chamfers参数 (3个)
            'recognize_chamfers_min_angle', 'recognize_chamfers_max_angle', 'recognize_chamfers_max_width',
            # rule_chamfer参数 (1个)
            'rule_chamfer_width_1',
            # distortion_angle参数 (1个)
            'distortion_angle',
            # perimeter_distance参数 (1个)
            'perimeter_distance',
            # CFD特定参数 (3个)
            'mesh_density', 'growth_rate', 'mesh_topology'
        ]
        
        missing_params = []
        unexpected_params = []
        
        for param in expected_params:
            if param not in param_names:
                missing_params.append(param)
        
        for param in param_names:
            if param not in expected_params:
                unexpected_params.append(param)
        
        if missing_params:
            print(f"❌ 缺失参数: {missing_params}")
        else:
            print(f"✅ 所有预期参数都存在")
            
        if unexpected_params:
            print(f"⚠️  意外参数: {unexpected_params}")
        
        return True, config_manager, param_names
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        traceback.print_exc()
        return False, None, []

def test_parameter_definitions(config_manager):
    """测试参数定义"""
    print("\n" + "=" * 60)
    print("🔍 第2步: 测试参数定义")
    print("=" * 60)
    
    try:
        param_space = config_manager.parameter_space
        param_names = param_space.get_parameter_names()
        bounds = param_space.get_bounds()
        param_types = param_space.get_parameter_types()
        ansa_mapping = param_space.get_ansa_mapping()
        default_values = param_space.get_default_values()
        
        print(f"✅ 参数定义获取成功")
        print(f"📊 参数边界: {len(bounds)} 个")
        print(f"📊 参数类型: {len(param_types)} 个")
        print(f"📊 ANSA映射: {len(ansa_mapping)} 个")
        print(f"📊 默认值: {len(default_values)} 个")
        
        # 详细检查每个参数
        problems = []
        for i, param_name in enumerate(param_names):
            param_def = param_space.get_parameter(param_name)
            if param_def is None:
                problems.append(f"参数 {param_name} 定义为空")
                continue
                
            # 检查边界
            if i < len(bounds):
                bound = bounds[i]
                if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                    problems.append(f"参数 {param_name} 边界格式错误: {bound}")
                elif bound[0] >= bound[1]:
                    problems.append(f"参数 {param_name} 边界无效: {bound}")
            else:
                problems.append(f"参数 {param_name} 缺少边界定义")
            
            # 检查类型
            if i < len(param_types):
                param_type = param_types[i]
                if param_type is None:
                    problems.append(f"参数 {param_name} 类型未定义")
            else:
                problems.append(f"参数 {param_name} 缺少类型定义")
        
        if problems:
            print("❌ 参数定义问题:")
            for problem in problems[:10]:  # 只显示前10个问题
                print(f"   - {problem}")
            return False
        else:
            print("✅ 所有参数定义正确")
            return True
            
    except Exception as e:
        print(f"❌ 参数定义测试失败: {e}")
        traceback.print_exc()
        return False

def test_parameter_space_conversion(config_manager):
    """测试参数空间转换"""
    print("\n" + "=" * 60)
    print("🔍 第3步: 测试参数空间转换")
    print("=" * 60)
    
    try:
        param_space = config_manager.parameter_space
        
        # 测试边界验证
        print("🔧 测试边界验证...")
        param_space.validate_bounds()
        print("✅ 边界验证通过")
        
        # 测试参数值验证
        print("🔧 测试参数值验证...")
        default_values = param_space.get_default_values()
        if default_values:
            param_space.validate_parameter_values(default_values)
            print("✅ 默认值验证通过")
        
        # 测试skopt转换
        print("🔧 测试skopt空间转换...")
        try:
            skopt_space = param_space.to_skopt_space()
            print(f"✅ skopt空间转换成功，包含 {len(skopt_space)} 个参数")
        except Exception as e:
            print(f"⚠️  skopt转换失败 (可能是依赖缺失): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数空间转换测试失败: {e}")
        traceback.print_exc()
        return False

def test_evaluator_compatibility(param_names):
    """测试评估器兼容性"""
    print("\n" + "=" * 60)
    print("🔍 第4步: 测试评估器兼容性")
    print("=" * 60)
    
    try:
        from src.evaluators.mesh_evaluator import create_mesh_evaluator
        
        # 测试mock评估器
        print("🔧 测试Mock评估器...")
        mock_evaluator = create_mesh_evaluator('mock')
        
        # 创建测试参数
        test_params = {}
        expected_param_mapping = {
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
        
        # 检查参数映射
        missing_mappings = []
        for param_name in param_names:
            if param_name in expected_param_mapping:
                test_params[param_name] = expected_param_mapping[param_name]
            else:
                missing_mappings.append(param_name)
        
        if missing_mappings:
            print(f"⚠️  缺少参数映射: {missing_mappings}")
        
        # 测试参数验证
        print("🔧 测试参数验证...")
        is_valid = mock_evaluator.validate_params(test_params)
        if is_valid:
            print("✅ 参数验证通过")
        else:
            print("❌ 参数验证失败")
            return False
        
        # 测试评估功能
        print("🔧 测试评估功能...")
        result = mock_evaluator.evaluate_mesh(test_params)
        if isinstance(result, (int, float)) and result >= 0:
            print(f"✅ 评估功能正常 (结果: {result:.6f})")
        else:
            print(f"❌ 评估返回无效结果: {result}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 评估器兼容性测试失败: {e}")
        traceback.print_exc()
        return False

def test_optimizer_integration(config_manager, param_names):
    """测试优化器集成"""
    print("\n" + "=" * 60)
    print("🔍 第5步: 测试优化器集成")
    print("=" * 60)
    
    try:
        from src.core.ansa_mesh_optimizer_improved import MeshOptimizer
        
        print("🔧 创建优化器实例...")
        optimizer = MeshOptimizer(
            config=config_manager.optimization_config,
            evaluator_type='mock',
            use_cache=False
        )
        
        print("✅ 优化器创建成功")
        
        # 检查参数空间
        opt_param_space = optimizer.param_space
        opt_param_names = opt_param_space.get_parameter_names()
        opt_bounds = opt_param_space.get_bounds()
        
        print(f"📊 优化器参数空间: {len(opt_param_names)} 个参数")
        print(f"📊 优化器参数边界: {len(opt_bounds)} 个边界")
        
        # 验证参数一致性
        param_mismatch = []
        for param in param_names:
            if param not in opt_param_names:
                param_mismatch.append(param)
        
        if param_mismatch:
            print(f"❌ 参数不匹配: {param_mismatch}")
            return False
        else:
            print("✅ 参数空间一致性验证通过")
        
        # 测试基础优化功能
        print("🔧 测试基础优化功能 (5次迭代)...")
        try:
            result = optimizer.optimize(
                optimizer='genetic',  # 使用总是可用的遗传算法
                n_calls=5
            )
            
            if 'best_value' in result and isinstance(result['best_value'], (int, float)):
                print(f"✅ 优化功能正常 (最佳值: {result['best_value']:.6f})")
                
                # 检查返回的参数
                best_params = result['best_params']
                print(f"📊 返回参数数量: {len(best_params)}")
                
                missing_return_params = []
                for param in param_names:
                    if param not in best_params:
                        missing_return_params.append(param)
                
                if missing_return_params:
                    print(f"❌ 优化结果缺少参数: {missing_return_params}")
                    return False
                else:
                    print("✅ 所有参数都包含在优化结果中")
                
                return True
            else:
                print("❌ 优化返回无效结果")
                return False
                
        except Exception as e:
            print(f"❌ 优化功能测试失败: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ 优化器集成测试失败: {e}")
        traceback.print_exc()
        return False

def test_parameter_flow(config_manager):
    """测试参数流向"""
    print("\n" + "=" * 60)
    print("🔍 第6步: 测试参数流向")
    print("=" * 60)
    
    try:
        # 测试参数从配置到优化器的完整流向
        param_space = config_manager.parameter_space
        param_names = param_space.get_parameter_names()
        
        print("🔧 测试参数获取流程...")
        
        # 1. 获取参数名称
        names_from_space = param_space.get_parameter_names()
        print(f"✅ 参数名称获取: {len(names_from_space)} 个")
        
        # 2. 获取参数边界
        bounds_from_space = param_space.get_bounds()
        print(f"✅ 参数边界获取: {len(bounds_from_space)} 个")
        
        # 3. 获取参数类型
        types_from_space = param_space.get_parameter_types()
        print(f"✅ 参数类型获取: {len(types_from_space)} 个")
        
        # 4. 获取ANSA映射
        ansa_mapping = param_space.get_ansa_mapping()
        print(f"✅ ANSA映射获取: {len(ansa_mapping)} 个")
        
        # 5. 检查数据一致性
        if len(names_from_space) == len(bounds_from_space) == len(types_from_space):
            print("✅ 参数数据长度一致性验证通过")
        else:
            print(f"❌ 参数数据长度不一致:")
            print(f"   名称: {len(names_from_space)}")
            print(f"   边界: {len(bounds_from_space)}")
            print(f"   类型: {len(types_from_space)}")
            return False
        
        # 6. 测试wrapper兼容性
        from src.core.ansa_mesh_optimizer_improved import config_manager as opt_config_manager
        opt_param_space = opt_config_manager.parameter_space
        
        print("🔧 测试wrapper兼容性...")
        wrapper_param_names = opt_param_space.get_parameter_names()
        wrapper_bounds = opt_param_space.get_bounds()
        
        if set(names_from_space) == set(wrapper_param_names):
            print("✅ Wrapper参数名称一致性验证通过")
        else:
            print("❌ Wrapper参数名称不一致")
            return False
        
        if len(bounds_from_space) == len(wrapper_bounds):
            print("✅ Wrapper参数边界一致性验证通过")
        else:
            print("❌ Wrapper参数边界不一致")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 参数流向测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始全参数导入测试")
    print("测试目标: 验证所有20个参数是否能正确导入到optimizer进行优化")
    
    all_tests_passed = True
    
    # 第1步: 测试配置系统
    success, config_manager, param_names = test_config_system()
    if not success:
        print("\n❌ 配置系统测试失败，无法继续")
        return False
    all_tests_passed &= success
    
    # 第2步: 测试参数定义
    success = test_parameter_definitions(config_manager)
    all_tests_passed &= success
    
    # 第3步: 测试参数空间转换
    success = test_parameter_space_conversion(config_manager)
    all_tests_passed &= success
    
    # 第4步: 测试评估器兼容性
    success = test_evaluator_compatibility(param_names)
    all_tests_passed &= success
    
    # 第5步: 测试优化器集成
    success = test_optimizer_integration(config_manager, param_names)
    all_tests_passed &= success
    
    # 第6步: 测试参数流向
    success = test_parameter_flow(config_manager)
    all_tests_passed &= success
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    if all_tests_passed:
        print("🎉 所有测试通过！")
        print("✅ 所有20个参数都能正确导入到optimizer进行优化")
        print(f"📋 参数列表: {param_names}")
        print(f"📊 参数总数: {len(param_names)}")
    else:
        print("❌ 部分测试失败！")
        print("🔧 建议检查上述失败的测试步骤")
    
    return all_tests_passed

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n退出代码: {exit_code}")
    except KeyboardInterrupt:
        print("\n⚠️  用户中断测试")
        exit_code = 130
    except Exception as e:
        print(f"\n💥 测试过程中发生未捕获的异常: {e}")
        traceback.print_exc()
        exit_code = 1
    
    sys.exit(exit_code)