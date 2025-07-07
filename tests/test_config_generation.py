#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件生成测试脚本

验证config generate命令是否包含所有参数

作者: Chel
创建日期: 2025-07-07
"""

import sys
import json
from pathlib import Path

def test_config_file_completeness():
    """测试配置文件完整性"""
    print("🔍 测试配置文件生成是否包含所有参数")
    print("=" * 60)
    
    # 期望的20个参数列表
    expected_parameters = [
        'element_size', 'perimeter_length', 'min_target_length', 'max_target_length', 
        'distortion_distance', 'quality_threshold', 'smoothing_iterations', 
        'rule_fillet_width_1', 'rule_fillet_width_2', 'rule_fillet_width_3', 'rule_fillet_width_4',
        'recognize_chamfers_min_angle', 'recognize_chamfers_max_angle', 'recognize_chamfers_max_width',
        'rule_chamfer_width_1', 'distortion_angle', 'perimeter_distance', 
        'mesh_density', 'growth_rate', 'mesh_topology'
    ]
    
    # 检查配置文件是否存在
    config_file = Path("default_config.json")
    if not config_file.exists():
        print("❌ 配置文件不存在: default_config.json")
        return False
    
    try:
        # 读取配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        print(f"✅ 配置文件读取成功")
        
        # 检查基本结构
        if 'parameters' not in config_data:
            print("❌ 配置文件缺少 'parameters' 部分")
            return False
        
        parameters = config_data['parameters']
        found_parameters = list(parameters.keys())
        
        print(f"📊 配置文件包含参数数量: {len(found_parameters)}")
        print(f"📊 期望参数数量: {len(expected_parameters)}")
        
        # 检查参数完整性
        missing_parameters = []
        extra_parameters = []
        
        for param in expected_parameters:
            if param not in found_parameters:
                missing_parameters.append(param)
        
        for param in found_parameters:
            if param not in expected_parameters:
                extra_parameters.append(param)
        
        # 输出详细结果
        print("\n🔍 参数完整性检查:")
        
        if not missing_parameters and not extra_parameters:
            print("✅ 所有参数都正确包含在配置文件中!")
            
            # 验证每个参数的结构
            print("\n🔍 参数结构验证:")
            valid_structure = True
            
            for param in expected_parameters:
                param_config = parameters[param]
                
                # 检查必需字段
                required_fields = ['param_type', 'bounds', 'description', 'ansa_mapping', 'default_value']
                missing_fields = []
                
                for field in required_fields:
                    if field not in param_config:
                        missing_fields.append(field)
                
                if missing_fields:
                    print(f"  ❌ {param}: 缺少字段 {missing_fields}")
                    valid_structure = False
                else:
                    print(f"  ✅ {param}: 结构完整")
            
            if valid_structure:
                print("\n🎉 配置文件结构验证通过!")
                
                # 输出参数分类统计
                print("\n📊 参数分类统计:")
                type_counts = {}
                for param, config in parameters.items():
                    param_type = config.get('param_type', 'unknown')
                    type_counts[param_type] = type_counts.get(param_type, 0) + 1
                
                for param_type, count in type_counts.items():
                    print(f"  {param_type}: {count} 个参数")
                
                return True
            else:
                print("\n❌ 配置文件结构验证失败!")
                return False
        else:
            if missing_parameters:
                print(f"❌ 缺少参数 ({len(missing_parameters)}个):")
                for param in missing_parameters:
                    print(f"  - {param}")
            
            if extra_parameters:
                print(f"⚠️ 额外参数 ({len(extra_parameters)}个):")
                for param in extra_parameters:
                    print(f"  + {param}")
            
            return len(missing_parameters) == 0
        
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 读取配置文件时出错: {e}")
        return False

def test_parameter_mappings():
    """测试参数映射一致性"""
    print("\n🔍 测试参数映射一致性")
    print("=" * 60)
    
    config_file = Path("default_config.json")
    if not config_file.exists():
        print("❌ 配置文件不存在")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        parameters = config_data.get('parameters', {})
        
        # 检查ansa_mapping的唯一性
        mappings = {}
        duplicate_mappings = []
        
        for param_name, param_config in parameters.items():
            ansa_mapping = param_config.get('ansa_mapping', '')
            if ansa_mapping:
                if ansa_mapping in mappings:
                    duplicate_mappings.append((param_name, mappings[ansa_mapping], ansa_mapping))
                else:
                    mappings[ansa_mapping] = param_name
        
        if duplicate_mappings:
            print("⚠️ 发现重复的ANSA映射:")
            for param1, param2, mapping in duplicate_mappings:
                print(f"  {param1} 和 {param2} 都映射到: {mapping}")
        else:
            print("✅ 所有ANSA映射都是唯一的")
        
        # 检查参数边界的合理性
        print("\n🔍 参数边界检查:")
        boundary_issues = []
        
        for param_name, param_config in parameters.items():
            bounds = param_config.get('bounds', [])
            default_value = param_config.get('default_value')
            
            if len(bounds) == 2:
                min_val, max_val = bounds
                
                # 检查边界顺序
                if min_val >= max_val:
                    boundary_issues.append(f"{param_name}: 最小值({min_val}) >= 最大值({max_val})")
                
                # 检查默认值是否在边界内
                if default_value is not None:
                    if not (min_val <= default_value <= max_val):
                        boundary_issues.append(f"{param_name}: 默认值({default_value}) 不在边界[{min_val}, {max_val}]内")
            else:
                boundary_issues.append(f"{param_name}: 边界格式错误")
        
        if boundary_issues:
            print("⚠️ 发现边界问题:")
            for issue in boundary_issues:
                print(f"  - {issue}")
        else:
            print("✅ 所有参数边界都合理")
        
        return len(duplicate_mappings) == 0 and len(boundary_issues) == 0
        
    except Exception as e:
        print(f"❌ 测试参数映射时出错: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 配置文件生成完整性测试")
    print("=" * 70)
    
    results = {}
    
    # 测试1: 配置文件完整性
    print("\n📋 测试1: 配置文件参数完整性")
    results['completeness'] = test_config_file_completeness()
    
    # 测试2: 参数映射一致性
    print("\n📋 测试2: 参数映射一致性")
    results['mappings'] = test_parameter_mappings()
    
    # 输出总结
    print("\n" + "=" * 70)
    print("📊 测试结果总结")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! 配置文件生成功能正常!")
        print("\n✅ 确认结果:")
        print("  - 配置文件包含所有20个参数")
        print("  - 参数结构完整")
        print("  - 参数映射一致")
        print("  - 参数边界合理")
        
        return True
    else:
        print(f"\n❌ {total - passed} 个测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)