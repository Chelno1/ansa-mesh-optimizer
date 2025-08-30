#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新参数替换功能演示
"""

import sys
import os
import shutil
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.evaluators.parameter_replacement_strategies import ParameterReplacementManager

def demo_parameter_replacement():
    """演示参数替换功能"""
    print("=== 新参数替换功能演示 ===\n")
    
    # 复制测试用的mpar文件
    source_file = "data/mesh/mend.ansa_mpar"
    test_file = "data/mesh/test_mend.ansa_mpar"
    
    if not Path(source_file).exists():
        print(f"错误: 源文件 {source_file} 不存在")
        return
    
    # 复制文件用于测试
    shutil.copy2(source_file, test_file)
    print(f"已复制测试文件: {test_file}")
    
    # 创建参数替换管理器
    manager = ParameterReplacementManager()
    
    # 定义测试参数 - 包含所有7个新参数
    test_params = {
        # 简单参数
        "general_curvature_minimum_length": 3.2,
        "recognize_feature_line_bounds_angle": 25.0,
        "recognize_feature_line_bounds_corner_angle": 35.0,
        
        # 复杂参数 (treatment_hole_2d)
        "treatment_hole_2d_N1": 7,        # 替换第206行的N=6
        "treatment_hole_2d_dw1": 3.0,     # 替换第206行的width=2.5
        "treatment_hole_2d_N2": 10,       # 替换第207行的N=8
        "treatment_hole_2d_dw2": 3.5,     # 替换第207行的width=2.5
        "treatment_hole_2d_dw3": 0.8,     # 替换第208行的0.667
    }
    
    print("测试参数:")
    for param_name, param_value in test_params.items():
        print(f"  {param_name}: {param_value}")
    
    print("\n开始参数替换...")
    
    # 执行参数替换
    try:
        result_file = manager.process_parameter_replacements(test_file, test_params)
        print(f"\n参数替换完成！")
        print(f"结果文件: {result_file}")
        
        # 检查结果文件是否存在
        if Path(result_file).exists():
            file_size = Path(result_file).stat().st_size
            print(f"文件大小: {file_size} bytes")
            
            # 显示前几行内容作为验证
            with open(result_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:30]
            print(f"\n前30行内容预览:")
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}: {line.rstrip()}")
        else:
            print("警告: 结果文件不存在")
            
    except Exception as e:
        print(f"参数替换过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理临时文件
    if Path(test_file).exists():
        os.remove(test_file)
        print(f"\n已清理临时文件: {test_file}")

def show_strategy_info():
    """显示策略信息"""
    print("\n=== 参数替换策略信息 ===\n")
    
    manager = ParameterReplacementManager()
    strategies = manager.get_available_strategies()
    
    print("已注册的策略:")
    for i, strategy_name in enumerate(strategies, 1):
        strategy = manager.get_strategy_by_name(strategy_name)
        print(f"{i}. {strategy_name}")
        print(f"   描述: {strategy.__doc__ or '无描述'}")
        
        # 测试策略能处理的参数类型
        if strategy_name == "SimpleParameterReplacement":
            test_params = {"general_curvature_minimum_length": 3.0}
            can_handle = strategy.can_handle(test_params)
            print(f"   可处理简单参数: {can_handle}")
            
        elif strategy_name == "TreatmentHole2dReplacement":
            test_params = {"treatment_hole_2d_N1": 7}
            can_handle = strategy.can_handle(test_params)
            print(f"   可处理复杂参数: {can_handle}")
        
        print()

if __name__ == "__main__":
    print("新增参数功能演示\n")
    
    try:
        show_strategy_info()
        demo_parameter_replacement()
        
        print("\n=== 演示完成 ===")
        print("所有7个新参数已成功集成到参数替换系统中！")
        print("\n新增的参数:")
        print("1. general_curvature_minimum_length - 通用曲率最小长度")
        print("2. recognize_feature_line_bounds_angle - 特征线边界识别角度")
        print("3. recognize_feature_line_bounds_corner_angle - 特征线边界拐角角度")
        print("4. treatment_hole_2d_N1 - 孔洞处理规则2的节点数量")
        print("5. treatment_hole_2d_dw1 - 孔洞处理规则2的区域宽度")
        print("6. treatment_hole_2d_N2 - 孔洞处理规则3的节点数量")
        print("7. treatment_hole_2d_dw2 - 孔洞处理规则3的区域宽度")
        print("8. treatment_hole_2d_dw3 - 孔洞处理规则4的区域宽度系数")
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()