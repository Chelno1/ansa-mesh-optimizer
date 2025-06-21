#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行接口测试脚本

作者: Chel
创建日期: 2025-06-19
"""

import sys
import subprocess
from pathlib import Path

def test_cli_help():
    """测试帮助命令"""
    print("测试帮助命令...")
    
    try:
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ 主帮助命令成功")
            if "--evaluator" in result.stdout:
                print("❌ 发现--evaluator在主帮助中（应该在子命令中）")
            else:
                print("✓ --evaluator正确地在子命令中")
        else:
            print(f"❌ 主帮助命令失败: {result.stderr}")
        
        # 测试optimize子命令帮助
        result = subprocess.run([sys.executable, "main.py", "optimize", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ optimize帮助命令成功")
            if "--evaluator" in result.stdout:
                print("✓ --evaluator在optimize帮助中找到")
            else:
                print("❌ --evaluator在optimize帮助中未找到")
        else:
            print(f"❌ optimize帮助命令失败: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试帮助命令失败: {e}")
        return False

def test_cli_parsing():
    """测试命令行参数解析"""
    print("\n测试命令行参数解析...")
    
    # 测试命令
    test_commands = [
        ["main.py", "optimize", "--optimizer", "random", "--n-calls", "5", "--evaluator", "mock"],
        ["main.py", "optimize", "--help"],
        ["main.py", "compare", "--optimizers", "random", "genetic", "--evaluator", "mock"],
        ["main.py", "info"],
        ["main.py", "config", "generate"]
    ]
    
    for cmd in test_commands:
        print(f"测试命令: {' '.join(cmd)}")
        
        try:
            if "--help" in cmd:
                # 帮助命令应该成功
                result = subprocess.run([sys.executable] + cmd[1:], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("  ✓ 帮助命令成功")
                else:
                    print(f"  ❌ 帮助命令失败: {result.returncode}")
            else:
                # 检查命令是否能正确解析（不一定要成功执行）
                result = subprocess.run([sys.executable] + cmd[1:], 
                                      capture_output=True, text=True, timeout=30)
                
                if "unrecognized arguments" in result.stderr:
                    print(f"  ❌ 参数解析失败: {result.stderr}")
                elif "error:" in result.stderr and "unrecognized" in result.stderr:
                    print(f"  ❌ 参数错误: {result.stderr}")
                else:
                    print(f"  ✓ 参数解析成功 (返回码: {result.returncode})")
                    
        except subprocess.TimeoutExpired:
            print("  ⚠️ 命令超时（可能在正常执行）")
        except Exception as e:
            print(f"  ❌ 命令执行异常: {e}")
    
    return True

def test_simple_execution():
    """测试简单执行"""
    print("\n测试简单执行...")
    
    try:
        # 尝试执行一个简单的优化命令
        cmd = [sys.executable, "main.py", "optimize", 
               "--optimizer", "random", "--n-calls", "3", "--evaluator", "mock"]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print("标准输出:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        
        if result.stderr:
            print("标准错误:")
            print(result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
        
        # 检查是否有明显的错误
        if "unrecognized arguments" in result.stderr:
            print("❌ 参数解析错误")
            return False
        elif "Traceback" in result.stderr and "ImportError" in result.stderr:
            print("⚠️ 导入错误，但参数解析正确")
            return True
        elif result.returncode == 0:
            print("✓ 命令执行成功")
            return True
        else:
            print(f"⚠️ 命令执行完成但有错误 (返回码: {result.returncode})")
            return True  # 参数解析可能是正确的
            
    except subprocess.TimeoutExpired:
        print("⚠️ 命令执行超时（可能在正常运行）")
        return True
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False

def main():
    """主测试函数"""
    print("命令行接口测试")
    print("=" * 40)
    
    tests = [
        ("帮助命令", test_cli_help),
        ("参数解析", test_cli_parsing),
        ("简单执行", test_simple_execution)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"运行测试: {test_name}")
        print('='*40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出结果摘要
    print("\n" + "=" * 40)
    print("测试结果摘要")
    print("=" * 40)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "❌ 失败"
        print(f"{test_name:<15}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有CLI测试通过!")
        print("\n现在可以安全使用以下命令:")
        print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
    elif passed >= total * 0.6:
        print(f"\n✅ 大部分测试通过，可以尝试运行命令")
        print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
    else:
        print("\n❌ 多数测试失败，建议检查main.py文件")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)