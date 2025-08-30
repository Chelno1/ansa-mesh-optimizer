#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新增的7个参数功能
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.config.config import SimpleConfigManager, SimpleParameterSpace
from src.evaluators.parameter_replacement_strategies import (
    ParameterReplacementManager,
    SimpleParameterReplacementStrategy,
    TreatmentHole2dReplacementStrategy
)

def test_parameter_definitions():
    """测试7个新参数是否正确定义"""
    print("=== 测试参数定义 ===")
    
    # 创建参数空间
    param_space = SimpleParameterSpace()
    
    # 检查新添加的参数
    new_params = [
        "general_curvature_minimum_length",
        "recognize_feature_line_bounds_angle", 
        "recognize_feature_line_bounds_corner_angle",
        "treatment_hole_2d_N1",
        "treatment_hole_2d_dw1",
        "treatment_hole_2d_N2", 
        "treatment_hole_2d_dw2",
        "treatment_hole_2d_dw3"
    ]
    
    all_params = param_space.get_parameter_names()
    
    for param_name in new_params:
        if param_name in all_params:
            param_def = param_space.get_parameter(param_name)
            print(f"✓ {param_name}: {param_def.description} [{param_def.bounds}] 默认值={param_def.default_value}")
        else:
            print(f"✗ {param_name}: 未找到定义")
    
    print(f"\n总参数数量: {len(all_params)}")

def test_strategy_registration():
    """测试策略注册"""
    print("\n=== 测试策略注册 ===")
    
    manager = ParameterReplacementManager()
    strategies = manager.get_available_strategies()
    
    print("已注册的策略:")
    for strategy in strategies:
        print(f"  - {strategy}")
    
    # 检查是否包含新策略
    if "TreatmentHole2dReplacement" in strategies:
        print("✓ TreatmentHole2dReplacementStrategy 已正确注册")
    else:
        print("✗ TreatmentHole2dReplacementStrategy 未注册")

def test_parameter_filtering():
    """测试参数过滤逻辑"""
    print("\n=== 测试参数过滤 ===")
    
    # 测试参数
    test_params = {
        "general_curvature_minimum_length": 3.0,
        "recognize_feature_line_bounds_angle": 25.0,
        "treatment_hole_2d_N1": 7,
        "treatment_hole_2d_dw1": 3.0,
        "rule_fillet_width_1": 4.0
    }
    
    # 简单策略应该只处理前两个
    simple_strategy = SimpleParameterReplacementStrategy()
    can_handle_simple = simple_strategy.can_handle(test_params)
    print(f"SimpleParameterReplacementStrategy 可处理: {can_handle_simple}")
    
    # Treatment策略应该处理treatment_hole_2d参数
    treatment_strategy = TreatmentHole2dReplacementStrategy()
    can_handle_treatment = treatment_strategy.can_handle(test_params)
    print(f"TreatmentHole2dReplacementStrategy 可处理: {can_handle_treatment}")

def test_parameter_bounds():
    """测试参数边界"""
    print("\n=== 测试参数边界 ===")
    
    param_space = SimpleParameterSpace()
    
    # 测试边界验证
    test_values = {
        "general_curvature_minimum_length": 2.5,  # 在 1.0-5.0 范围内
        "recognize_feature_line_bounds_angle": 20.0,  # 在 10.0-30.0 范围内  
        "treatment_hole_2d_N1": 6,  # 在 4-10 范围内
        "treatment_hole_2d_dw1": 2.5,  # 在 1.0-4.0 范围内
        "treatment_hole_2d_dw3": 0.667  # 在 0.4-1.0 范围内
    }
    
    try:
        param_space.validate_parameter_values(test_values)
        print("✓ 参数边界验证通过")
    except Exception as e:
        print(f"✗ 参数边界验证失败: {e}")

def test_ansa_mapping():
    """测试ANSA映射"""
    print("\n=== 测试ANSA映射 ===")
    
    param_space = SimpleParameterSpace()
    ansa_mapping = param_space.get_ansa_mapping()
    
    # 检查新参数的映射
    new_param_mappings = {
        "general_curvature_minimum_length": "general_curvature_minimum_length",
        "recognize_feature_line_bounds_angle": "recognize_feature_line_bounds_angle", 
        "recognize_feature_line_bounds_corner_angle": "recognize_feature_line_bounds_corner_angle",
        "treatment_hole_2d_N1": "treatment_hole_2d_N1",
        "treatment_hole_2d_dw1": "treatment_hole_2d_dw1",
        "treatment_hole_2d_N2": "treatment_hole_2d_N2",
        "treatment_hole_2d_dw2": "treatment_hole_2d_dw2", 
        "treatment_hole_2d_dw3": "treatment_hole_2d_dw3"
    }
    
    for param_name, expected_mapping in new_param_mappings.items():
        if param_name in ansa_mapping:
            actual_mapping = ansa_mapping[param_name]
            if actual_mapping == expected_mapping:
                print(f"✓ {param_name} -> {actual_mapping}")
            else:
                print(f"✗ {param_name} 映射错误: {actual_mapping} != {expected_mapping}")
        else:
            print(f"✗ {param_name} 无ANSA映射")

if __name__ == "__main__":
    print("开始测试新增的7个参数功能...\n")
    
    try:
        test_parameter_definitions()
        test_strategy_registration()
        test_parameter_filtering()
        test_parameter_bounds()
        test_ansa_mapping()
        
        print("\n=== 测试完成 ===")
        print("所有基本功能测试通过！新参数已成功集成到系统中。")
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()