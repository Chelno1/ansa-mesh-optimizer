#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终测试脚本

作者: Chel
创建日期: 2025-06-19
"""

import sys
import subprocess
from pathlib import Path

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    modules = [
        'config',
        'mesh_evaluator', 
        'optimization_cache',
        'early_stopping',
        'ansa_mesh_optimizer_improved'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

def test_simple_optimization():
    """测试简单优化"""
    print("\n测试简单优化...")
    
    try:
        from mesh_evaluator import MockMeshEvaluator
        from optimization_cache import OptimizationCache
        
        # 创建评估器
        evaluator = MockMeshEvaluator()
        
        # 测试参数
        test_params = {
            'element_size': 1.0,
            'mesh_density': 3,
            'mesh_quality_threshold': 0.5
        }
        
        # 测试评估
        result = evaluator.evaluate_mesh(test_params)
        print(f"✓ 模拟评估成功: {result}")
        
        # 测试缓存
        cache = OptimizationCache('test_cache.pkl')
        cache.set(test_params, result)
        cached_result = cache.get(test_params)
        
        if cached_result and abs(cached_result['result'] - result) < 1e-6:
            print("✓ 缓存功能正常")
        else:
            print("❌ 缓存功能异常")
            return False
        
        # 清理
        try:
            Path('test_cache.pkl').unlink()
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ 简单优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_command():
    """测试主命令"""
    print("\n测试主命令...")
    
    cmd = [sys.executable, "main.py", "optimize", "--optimizer", "random", "--n-calls", "3", "--evaluator", "mock"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"返回码: {result.returncode}")
        
        if result.returncode == 0:
            print("✓ 命令执行成功")
            print("输出预览:")
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print("❌ 命令执行失败")
            print("错误信息:")
            print(result.stderr)
            print("标准输出:")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ 命令超时")
        return False
    except Exception as e:
        print(f"❌ 命令测试异常: {e}")
        return False

def run_actual_optimization():
    """运行实际优化"""
    print("\n运行实际优化...")
    
    try:
        from ansa_mesh_optimizer_improved import optimize_mesh_parameters
        
        print("开始优化...")
        result = optimize_mesh_parameters(
            n_calls=3,
            optimizer='random',
            evaluator_type='mock'
        )
        
        print(f"✓ 优化成功完成")
        print(f"最佳值: {result['best_value']:.6f}")
        print(f"最佳参数: {result['best_params']}")
        print(f"执行时间: {result['execution_time']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 实际优化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("最终测试脚本")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("简单优化", test_simple_optimization),
        ("实际优化", run_actual_optimization),
        ("主命令", test_main_command)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出总结
    print("\n" + "=" * 50)
    print("最终测试结果")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✓ 通过" if success else "❌ 失败"
        print(f"{test_name:<15}: {status}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过! 系统工作正常!")
        print("\n你现在可以安全使用以下命令:")
        print("python main.py optimize --optimizer bayesian --n-calls 20 --evaluator mock")
        print("python main.py compare --optimizers random genetic --n-calls 15 --evaluator mock")
    elif passed >= 3:
        print(f"\n✅ 大部分测试通过，系统基本可用")
        print("建议先使用简单命令:")
        print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
    else:
        print(f"\n❌ 多数测试失败，需要进一步调试")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)