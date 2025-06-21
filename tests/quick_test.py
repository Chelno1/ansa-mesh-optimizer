#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证JSON序列化修复

作者: Chel
创建日期: 2025-06-19
"""

import sys
import json
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_numpy_serialization():
    """测试numpy类型序列化问题"""
    print("测试numpy类型JSON序列化...")
    
    # 模拟优化器可能遇到的参数类型
    test_params = {
        'element_size': np.float64(1.5),
        'mesh_density': np.int64(3),
        'mesh_quality_threshold': np.float32(0.7),
        'smoothing_iterations': np.int32(50),
        'mesh_growth_rate': 1.2,
        'mesh_topology': 2
    }
    
    print(f"原始参数: {test_params}")
    print("参数类型:")
    for key, value in test_params.items():
        print(f"  {key}: {type(value)} = {value}")
    
    # 测试原始JSON序列化（应该失败）
    try:
        json.dumps(test_params)
        print("❌ 原始JSON序列化意外成功")
    except TypeError as e:
        print(f"✓ 原始JSON序列化失败（预期）: {e}")
    
    # 测试修复后的序列化
    def normalize_for_json(obj):
        """标准化对象以便JSON序列化"""
        if isinstance(obj, dict):
            return {key: normalize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [normalize_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # 其他numpy类型
            return obj.item()
        else:
            return obj
    
    try:
        normalized_params = normalize_for_json(test_params)
        json_str = json.dumps(normalized_params, sort_keys=True)
        print(f"✓ 修复后的序列化成功: {json_str}")
        
        # 验证反序列化
        deserialized = json.loads(json_str)
        print(f"✓ 反序列化成功: {deserialized}")
        
        # 检查类型转换
        print("标准化后的参数类型:")
        for key, value in normalized_params.items():
            print(f"  {key}: {type(value)} = {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复后的序列化失败: {e}")
        return False

def test_quick_optimization():
    """快速测试优化功能"""
    print("\n" + "="*50)
    print("快速测试优化功能")
    print("="*50)
    
    try:
        # 尝试导入必要模块
        from utils import normalize_params, safe_json_serialize
        from mesh_evaluator import MockMeshEvaluator
        
        print("✓ 成功导入模块")
        
        # 创建测试参数
        test_params = {
            'element_size': np.float64(1.0),
            'mesh_density': np.int64(2),
            'mesh_quality_threshold': 0.5
        }
        
        # 测试参数标准化
        normalized = normalize_params(test_params)
        print(f"✓ 参数标准化成功: {normalized}")
        
        # 测试JSON序列化
        json_str = safe_json_serialize(normalized)
        print(f"✓ JSON序列化成功，长度: {len(json_str)}")
        
        # 测试评估器
        evaluator = MockMeshEvaluator()
        result = evaluator.evaluate_mesh(test_params)
        print(f"✓ 模拟评估成功: {result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_minimal_optimizer():
    """创建最小化的优化器测试"""
    print("\n" + "="*50)
    print("创建最小化优化器测试")
    print("="*50)
    
    try:
        # 只导入必要的模块
        from utils import normalize_params
        
        # 定义一个简单的优化函数
        def simple_objective(params):
            """简单的目标函数"""
            normalized = normalize_params(params)
            x1 = normalized.get('element_size', 1.0)
            x2 = normalized.get('mesh_density', 3)
            x3 = normalized.get('mesh_quality_threshold', 0.5)
            
            # 简单的二次函数
            return (x1 - 1.0)**2 + (x2 - 3)**2 + (x3 - 0.5)**2
        
        # 测试不同参数组合
        test_cases = [
            {
                'element_size': np.float64(1.0),
                'mesh_density': np.int64(3),
                'mesh_quality_threshold': 0.5
            },
            {
                'element_size': 1.5,
                'mesh_density': 2,
                'mesh_quality_threshold': 0.7
            },
            {
                'element_size': np.float32(0.8),
                'mesh_density': np.int32(4),
                'mesh_quality_threshold': np.float64(0.3)
            }
        ]
        
        print("测试目标函数:")
        for i, params in enumerate(test_cases):
            result = simple_objective(params)
            print(f"  测试 {i+1}: {params} -> {result}")
        
        print("✓ 最小化优化器测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 最小化优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("快速测试脚本 - 验证JSON序列化修复")
    print("=" * 60)
    
    tests = [
        ("Numpy序列化", test_numpy_serialization),
        ("快速优化", test_quick_optimization),
        ("最小化优化器", create_minimal_optimizer)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✓ 通过" if success else "❌ 失败"
        print(f"{test_name:<20}: {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过! 现在可以尝试运行完整的优化器了。")
        print("\n建议执行:")
        print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
    else:
        print(f"\n⚠️  仍有 {failed} 个测试失败，建议先解决这些问题。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)