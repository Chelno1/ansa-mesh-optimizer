#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复后的测试脚本

作者: Chel
创建日期: 2025-06-19
"""

import sys
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_optimization():
    """测试基本优化功能"""
    print("=" * 50)
    print("测试基本优化功能")
    print("=" * 50)
    
    try:
        from ansa_mesh_optimizer_improved import optimize_mesh_parameters
        
        # 使用模拟评估器进行快速测试
        result = optimize_mesh_parameters(
            n_calls=10,  # 减少迭代次数以便快速测试
            optimizer='bayesian',
            evaluator_type='mock'
        )
        
        print("\n优化结果:")
        print(f"最佳参数: {result['best_params']}")
        print(f"最佳值: {result['best_value']:.6f}")
        print(f"执行时间: {result['execution_time']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_functionality():
    """测试缓存功能"""
    print("\n" + "=" * 50)
    print("测试缓存功能")
    print("=" * 50)
    
    try:
        from optimization_cache import OptimizationCache
        from utils import normalize_params
        import numpy as np
        
        # 创建缓存
        cache = OptimizationCache('test_cache.pkl')
        
        # 测试参数（包含numpy类型）
        test_params = {
            'element_size': np.float64(1.5),
            'mesh_density': np.int64(3),
            'mesh_quality_threshold': 0.7
        }
        
        # 标准化参数
        normalized_params = normalize_params(test_params)
        print(f"原始参数: {test_params}")
        print(f"标准化参数: {normalized_params}")
        
        # 测试缓存设置和获取
        cache.set(normalized_params, 123.45)
        cached_result = cache.get(normalized_params)
        
        print(f"缓存结果: {cached_result}")
        
        # 清理测试文件
        cache_file = Path('test_cache.pkl')
        if cache_file.exists():
            cache_file.unlink()
        
        return cached_result == 123.45
        
    except Exception as e:
        print(f"缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_evaluator():
    """测试模拟评估器"""
    print("\n" + "=" * 50)
    print("测试模拟评估器")
    print("=" * 50)
    
    try:
        from mesh_evaluator import MockMeshEvaluator
        from utils import normalize_params
        import numpy as np
        
        evaluator = MockMeshEvaluator()
        
        # 测试参数（包含numpy类型）
        test_params = {
            'element_size': np.float64(1.0),
            'mesh_density': np.int64(2),
            'mesh_quality_threshold': np.float32(0.5),
            'smoothing_iterations': 40,
            'mesh_growth_rate': 1.0,
            'mesh_topology': 2
        }
        
        print(f"测试参数: {test_params}")
        
        # 标准化参数
        normalized_params = normalize_params(test_params)
        print(f"标准化参数: {normalized_params}")
        
        # 评估
        result = evaluator.evaluate_mesh(test_params)
        print(f"评估结果: {result}")
        
        return isinstance(result, (int, float)) and result >= 0
        
    except Exception as e:
        print(f"模拟评估器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_validation():
    """测试参数验证"""
    print("\n" + "=" * 50)
    print("测试参数验证")
    print("=" * 50)
    
    try:
        from config import config_manager
        from utils import validate_param_types, normalize_params
        import numpy as np
        
        param_space = config_manager.parameter_space
        
        # 测试不同类型的参数
        test_cases = [
            {
                'element_size': np.float64(1.5),
                'mesh_density': np.int32(3),
                'mesh_quality_threshold': 0.7,
                'smoothing_iterations': np.int64(50),
                'mesh_growth_rate': np.float32(1.2),
                'mesh_topology': 2
            },
            {
                'element_size': 1.0,
                'mesh_density': 4,
                'mesh_quality_threshold': 0.8,
                'smoothing_iterations': 30,
                'mesh_growth_rate': 0.9,
                'mesh_topology': 1
            }
        ]
        
        for i, test_params in enumerate(test_cases):
            print(f"\n测试案例 {i+1}:")
            print(f"原始参数: {test_params}")
            
            # 标准化
            normalized = normalize_params(test_params)
            print(f"标准化参数: {normalized}")
            
            # 验证类型
            validated = validate_param_types(normalized, param_space)
            print(f"验证后参数: {validated}")
            
            # 检查类型
            param_types = param_space.get_param_types()
            param_names = param_space.get_param_names()
            
            for j, name in enumerate(param_names):
                if name in validated:
                    expected_type = param_types[j]
                    actual_type = type(validated[name])
                    print(f"  {name}: {actual_type.__name__} (期望: {expected_type.__name__})")
                    
                    if expected_type == int:
                        assert isinstance(validated[name], int), f"{name} 应该是int类型"
                    elif expected_type == float:
                        assert isinstance(validated[name], (int, float)), f"{name} 应该是float类型"
        
        print("\n参数验证测试通过!")
        return True
        
    except Exception as e:
        print(f"参数验证测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_serialization():
    """测试JSON序列化"""
    print("\n" + "=" * 50)
    print("测试JSON序列化")
    print("=" * 50)
    
    try:
        from utils import safe_json_serialize, normalize_params
        import numpy as np
        import json
        
        # 创建包含各种numpy类型的测试数据
        test_data = {
            'float64': np.float64(1.23456),
            'int64': np.int64(42),
            'float32': np.float32(3.14159),
            'int32': np.int32(100),
            'array': np.array([1, 2, 3]),
            'scalar_array': np.array(5.0),
            'nested': {
                'value': np.float64(2.718),
                'list': [np.int64(1), np.int64(2), np.int64(3)]
            },
            'regular_types': {
                'string': 'test',
                'int': 123,
                'float': 456.789,
                'list': [1, 2, 3],
                'bool': True
            }
        }
        
        print("原始数据类型:")
        for key, value in test_data.items():
            if hasattr(value, 'dtype'):
                print(f"  {key}: {type(value).__name__} (dtype: {value.dtype})")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # 测试安全序列化
        json_str = safe_json_serialize(test_data)
        print(f"\nJSON序列化成功，长度: {len(json_str)}")
        
        # 测试反序列化
        deserialized = json.loads(json_str)
        print("反序列化成功")
        
        # 验证数据
        print("\n反序列化后的数据:")
        for key, value in deserialized.items():
            print(f"  {key}: {value} ({type(value).__name__})")
        
        return True
        
    except Exception as e:
        print(f"JSON序列化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_with_different_types():
    """测试优化器处理不同参数类型"""
    print("\n" + "=" * 50)
    print("测试优化器处理不同参数类型")
    print("=" * 50)
    
    try:
        from ansa_mesh_optimizer_improved import MeshOptimizer
        from config import config_manager
        
        # 创建优化器
        optimizer = MeshOptimizer(
            evaluator_type='mock',
            use_cache=False  # 禁用缓存以避免干扰
        )
        
        # 运行短期优化测试
        result = optimizer.optimize(
            optimizer='random',  # 使用随机搜索，更快
            n_calls=5  # 只运行5次迭代
        )
        
        print(f"优化结果: {result['best_params']}")
        print(f"最佳值: {result['best_value']}")
        
        # 检查参数类型
        param_space = config_manager.parameter_space
        param_types = param_space.get_param_types()
        param_names = param_space.get_param_names()
        
        print("\n参数类型检查:")
        for i, name in enumerate(param_names):
            if name in result['best_params']:
                value = result['best_params'][name]
                expected_type = param_types[i]
                actual_type = type(value)
                
                print(f"  {name}: {value} ({actual_type.__name__}, 期望: {expected_type.__name__})")
                
                # 验证类型正确性
                if expected_type == int:
                    assert isinstance(value, int), f"{name} 应该是int类型，实际是 {actual_type}"
                elif expected_type == float:
                    assert isinstance(value, (int, float)), f"{name} 应该是数值类型，实际是 {actual_type}"
        
        print("\n优化器类型处理测试通过!")
        return True
        
    except Exception as e:
        print(f"优化器类型处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始运行修复验证测试...")
    print("=" * 70)
    
    tests = [
        ("JSON序列化", test_json_serialization),
        ("参数验证", test_parameter_validation),
        ("缓存功能", test_cache_functionality),
        ("模拟评估器", test_mock_evaluator),
        ("优化器类型处理", test_optimizer_with_different_types),
        ("基本优化功能", test_basic_optimization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n正在运行: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{test_name} 运行失败: {e}")
            results[test_name] = False
    
    # 输出测试结果摘要
    print("\n" + "=" * 70)
    print("测试结果摘要")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<25} : {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 个测试通过, {failed} 个测试失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过! 修复成功!")
        return True
    else:
        print(f"\n❌ 还有 {failed} 个测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    # 运行测试
    success = run_all_tests()
    
    # 如果所有测试通过，运行原始命令
    if success:
        print("\n" + "=" * 70)
        print("运行原始命令测试")
        print("=" * 70)
        
        try:
            import subprocess
            import sys
            
            # 运行原始命令
            cmd = [sys.executable, "main.py", "optimize", "--optimizer", "bayesian", "--n-calls", "5", "--evaluator", "mock"]
            print(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            print("标准输出:")
            print(result.stdout)
            
            if result.stderr:
                print("标准错误:")
                print(result.stderr)
            
            print(f"返回码: {result.returncode}")
            
            if result.returncode == 0:
                print("\n🎉 原始命令执行成功!")
            else:
                print("\n❌ 原始命令执行失败")
                
        except Exception as e:
            print(f"运行原始命令时出错: {e}")
    
    sys.exit(0 if success else 1)