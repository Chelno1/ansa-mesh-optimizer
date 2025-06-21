#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速修复测试脚本

作者: Chel
创建日期: 2025-06-19
"""

import sys
import subprocess
import shutil
from pathlib import Path

def replace_mesh_evaluator():
    """替换mesh_evaluator.py文件"""
    print("替换mesh_evaluator.py文件...")
    
    try:
        # 检查文件是否存在
        if not Path("mesh_evaluator_fixed.py").exists():
            print("❌ mesh_evaluator_fixed.py 不存在")
            return False
        
        # 备份原文件
        if Path("mesh_evaluator.py").exists():
            shutil.copy2("mesh_evaluator.py", "mesh_evaluator_backup.py")
            print("✓ 原文件已备份为 mesh_evaluator_backup.py")
        
        # 替换文件
        shutil.copy2("mesh_evaluator_fixed.py", "mesh_evaluator.py")
        print("✓ 已使用修复版本替换 mesh_evaluator.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 替换文件失败: {e}")
        return False

def test_normalize_function():
    """测试normalize_params函数"""
    print("\n测试normalize_params函数...")
    
    try:
        from mesh_evaluator import normalize_params
        
        # 创建测试数据
        try:
            import numpy as np
            test_params = {
                'element_size': np.float64(1.5),
                'mesh_density': np.int64(3),
                'mesh_quality_threshold': 0.7,
                'regular_param': 42
            }
            has_numpy = True
        except ImportError:
            test_params = {
                'element_size': 1.5,
                'mesh_density': 3,
                'mesh_quality_threshold': 0.7,
                'regular_param': 42
            }
            has_numpy = False
        
        print(f"测试参数: {test_params}")
        print(f"Numpy可用: {has_numpy}")
        
        # 测试标准化
        normalized = normalize_params(test_params)
        print(f"标准化结果: {normalized}")
        
        # 检查类型
        print("参数类型检查:")
        for key, value in normalized.items():
            print(f"  {key}: {type(value).__name__} = {value}")
        
        print("✓ normalize_params 函数测试成功")
        return True
        
    except Exception as e:
        print(f"❌ normalize_params 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_evaluator():
    """测试模拟评估器"""
    print("\n测试模拟评估器...")
    
    try:
        from mesh_evaluator import MockMeshEvaluator
        
        evaluator = MockMeshEvaluator()
        
        test_params = {
            'element_size': 1.0,
            'mesh_density': 3,
            'mesh_quality_threshold': 0.5
        }
        
        print(f"测试参数: {test_params}")
        
        # 测试参数验证
        is_valid = evaluator.validate_params(test_params)
        print(f"参数验证: {is_valid}")
        
        if is_valid:
            # 测试评估
            result = evaluator.evaluate_mesh(test_params)
            print(f"评估结果: {result}")
            
            if isinstance(result, (int, float)) and result >= 0:
                print("✓ 模拟评估器测试成功")
                return True
            else:
                print(f"❌ 评估结果异常: {result}")
                return False
        else:
            print("❌ 参数验证失败")
            return False
        
    except Exception as e:
        print(f"❌ 模拟评估器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_optimization():
    """测试简单优化"""
    print("\n测试简单优化...")
    
    try:
        from ansa_mesh_optimizer_improved import optimize_mesh_parameters
        
        print("开始优化...")
        result = optimize_mesh_parameters(
            n_calls=3,
            optimizer='random',
            evaluator_type='mock'
        )
        
        print(f"✓ 优化成功!")
        print(f"最佳值: {result['best_value']:.6f}")
        print(f"最佳参数: {result['best_params']}")
        print(f"执行时间: {result['execution_time']:.2f}秒")
        
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
        
        print(f"命令: {' '.join(cmd)}")
        print(f"返回码: {result.returncode}")
        
        if result.returncode == 0:
            print("✓ 主命令执行成功")
            print("输出预览:")
            lines = result.stdout.split('\n')[:8]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print("❌ 主命令执行失败")
            print("错误信息:")
            error_lines = result.stderr.split('\n')[:5]
            for line in error_lines:
                if line.strip():
                    print(f"  {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ 命令超时")
        return False
    except Exception as e:
        print(f"❌ 命令测试异常: {e}")
        return False

def main():
    """主函数"""
    print("快速修复测试")
    print("=" * 50)
    
    # 替换文件
    if not replace_mesh_evaluator():
        print("文件替换失败，无法继续测试")
        return False
    
    # 运行测试
    tests = [
        ("normalize_params函数", test_normalize_function),
        ("模拟评估器", test_mock_evaluator),
        ("简单优化", test_simple_optimization),
        ("主命令", test_main_command)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*30}")
        print(f"运行测试: {test_name}")
        print('='*30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出结果
    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)
    
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
        print("\n🎉 所有测试通过! 修复成功!")
        print("\n你现在可以使用以下命令:")
        print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
        print("python main.py optimize --optimizer bayesian --n-calls 20 --evaluator mock")
        print("python main.py compare --optimizers random genetic --n-calls 10 --evaluator mock")
    elif passed >= 3:
        print(f"\n✅ 大部分测试通过，系统基本可用")
        print("建议先尝试简单命令:")
        print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
    else:
        print(f"\n❌ 多数测试失败，需要进一步调试")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)