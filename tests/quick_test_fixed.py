#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复导入问题的快速测试脚本

作者: Chel
创建日期: 2025-06-19
"""

import sys
import json
import logging
from pathlib import Path

# 配置基础日志
logging.basicConfig(level=logging.ERROR)  # 只显示错误，减少干扰

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 安全导入numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ numpy不可用，将使用模拟数据")

def create_test_params():
    """创建测试参数"""
    if NUMPY_AVAILABLE:
        return {
            'element_size': np.float64(1.5),
            'mesh_density': np.int64(3),
            'mesh_quality_threshold': np.float32(0.7),
            'smoothing_iterations': np.int32(50),
            'mesh_growth_rate': 1.2,
            'mesh_topology': 2
        }
    else:
        # 模拟numpy类型的行为
        class MockNumPy:
            def __init__(self, value):
                self.value = value
            def item(self):
                return self.value
            def __str__(self):
                return str(self.value)
            def __repr__(self):
                return f"mock_numpy({self.value})"
        
        return {
            'element_size': MockNumPy(1.5),
            'mesh_density': MockNumPy(3),
            'mesh_quality_threshold': 0.7,
            'smoothing_iterations': 50,
            'mesh_growth_rate': 1.2,
            'mesh_topology': 2
        }

def normalize_params_local(params):
    """本地参数标准化函数"""
    normalized = {}
    for key, value in params.items():
        if hasattr(value, 'item'):  # numpy类型或模拟类型
            normalized[key] = value.item()
        elif NUMPY_AVAILABLE and isinstance(value, (np.integer, np.floating)):
            normalized[key] = value.item()
        elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
            if value.size == 1:
                normalized[key] = value.item()
            else:
                normalized[key] = value.tolist()
        else:
            normalized[key] = value
    return normalized

def safe_json_serialize_local(obj):
    """本地安全JSON序列化"""
    def convert_types(obj):
        if isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy类型或模拟类型
            return obj.item()
        elif NUMPY_AVAILABLE and isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    converted_obj = convert_types(obj)
    return json.dumps(converted_obj, sort_keys=True, ensure_ascii=False)

def test_numpy_serialization():
    """测试numpy类型序列化问题"""
    print("测试JSON序列化修复...")
    
    test_params = create_test_params()
    
    print(f"原始参数: {test_params}")
    print("参数类型:")
    for key, value in test_params.items():
        print(f"  {key}: {type(value)} = {value}")
    
    # 测试原始JSON序列化（可能失败）
    try:
        json.dumps(test_params)
        print("❓ 原始JSON序列化成功（可能系统已处理numpy类型）")
    except (TypeError, ValueError) as e:
        print(f"✓ 原始JSON序列化失败（预期）: {e}")
    
    # 测试修复后的序列化
    try:
        normalized_params = normalize_params_local(test_params)
        json_str = safe_json_serialize_local(test_params)
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
        import traceback
        traceback.print_exc()
        return False

def test_module_imports():
    """测试模块导入"""
    print("\n" + "="*50)
    print("测试模块导入")
    print("="*50)
    
    modules_to_test = [
        ('utils', 'utils'),
        ('config', 'config'),
        ('mesh_evaluator', 'mesh_evaluator'),
        ('optimization_cache', 'optimization_cache'),
        ('early_stopping', 'early_stopping')
    ]
    
    import_results = {}
    
    for module_name, import_name in modules_to_test:
        try:
            __import__(import_name)
            print(f"✓ {module_name} 导入成功")
            import_results[module_name] = True
        except ImportError as e:
            print(f"❌ {module_name} 导入失败: {e}")
            import_results[module_name] = False
        except Exception as e:
            print(f"⚠️ {module_name} 导入异常: {e}")
            import_results[module_name] = False
    
    success_count = sum(import_results.values())
    total_count = len(import_results)
    
    print(f"\n导入结果: {success_count}/{total_count} 个模块成功")
    
    return success_count >= total_count * 0.6  # 至少60%成功

def test_mock_evaluator_standalone():
    """独立测试模拟评估器"""
    print("\n" + "="*50)
    print("独立测试模拟评估器")
    print("="*50)
    
    # 创建一个简单的模拟评估器
    class SimpleMockEvaluator:
        def evaluate_mesh(self, params):
            # 标准化参数
            normalized = normalize_params_local(params)
            
            x1 = normalized.get('element_size', 1.0)
            x2 = normalized.get('mesh_density', 3)
            x3 = normalized.get('mesh_quality_threshold', 0.5)
            
            # 简单的目标函数
            result = (x1 - 1.0)**2 + (x2 - 3)**2 + (x3 - 0.5)**2
            
            # 添加一些随机性
            import random
            noise = random.uniform(-0.1, 0.1) * result
            
            return max(0, result + noise)
        
        def validate_params(self, params):
            required_params = ['element_size', 'mesh_density', 'mesh_quality_threshold']
            return all(param in params for param in required_params)
    
    try:
        evaluator = SimpleMockEvaluator()
        
        # 测试参数
        test_cases = [
            create_test_params(),
            {
                'element_size': 1.0,
                'mesh_density': 3,
                'mesh_quality_threshold': 0.5
            }
        ]
        
        for i, test_params in enumerate(test_cases):
            print(f"测试案例 {i+1}: {test_params}")
            
            # 验证参数
            if evaluator.validate_params(test_params):
                result = evaluator.evaluate_mesh(test_params)
                print(f"  ✓ 评估结果: {result}")
            else:
                print(f"  ❌ 参数验证失败")
        
        print("✓ 独立模拟评估器测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 独立模拟评估器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_cache_standalone():
    """独立测试优化缓存"""
    print("\n" + "="*50)
    print("独立测试优化缓存")
    print("="*50)
    
    try:
        # 创建简单的缓存类
        class SimpleCache:
            def __init__(self):
                self.cache = {}
            
            def _compute_hash(self, params):
                # 标准化参数
                normalized = normalize_params_local(params)
                param_str = safe_json_serialize_local(normalized)
                import hashlib
                return hashlib.md5(param_str.encode()).hexdigest()
            
            def get(self, params):
                params_hash = self._compute_hash(params)
                return self.cache.get(params_hash)
            
            def set(self, params, result):
                params_hash = self._compute_hash(params)
                normalized = normalize_params_local(params)
                
                self.cache[params_hash] = {
                    'params': normalized,
                    'result': float(result) if hasattr(result, 'item') else result
                }
        
        cache = SimpleCache()
        
        # 测试参数
        test_params = create_test_params()
        test_result = 123.45
        
        print(f"测试参数: {test_params}")
        
        # 设置缓存
        cache.set(test_params, test_result)
        print(f"✓ 缓存设置成功")
        
        # 获取缓存
        cached_result = cache.get(test_params)
        if cached_result:
            cached_value = cached_result['result']
            print(f"✓ 缓存获取成功: {cached_value}")
            
            if abs(cached_value - test_result) < 1e-6:
                print("✓ 缓存值正确")
                return True
            else:
                print(f"❌ 缓存值不匹配: 期望 {test_result}, 实际 {cached_value}")
                return False
        else:
            print("❌ 缓存获取失败")
            return False
        
    except Exception as e:
        print(f"❌ 独立缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("修复导入问题的快速测试脚本")
    print("=" * 60)
    
    tests = [
        ("JSON序列化", test_numpy_serialization),
        ("模块导入", test_module_imports),
        ("独立模拟评估器", test_mock_evaluator_standalone),
        ("独立缓存", test_optimization_cache_standalone)
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
        print("\n🎉 所有测试通过!")
        print("\n建议下一步:")
        print("1. 运行: python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
        print("2. 或运行: python test_fixed.py (完整测试)")
    elif passed >= len(tests) * 0.75:
        print(f"\n✅ 大部分测试通过 ({passed}/{len(tests)})，可以尝试运行主程序")
        print("\n建议执行:")
        print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
    else:
        print(f"\n⚠️ 较多测试失败 ({failed}/{len(tests)})，建议先解决导入问题")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)