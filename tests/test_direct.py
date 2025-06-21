#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接测试修复的脚本

作者: Chel
创建日期: 2025-06-19
"""

import sys
import subprocess
from pathlib import Path

def test_help_commands():
    """测试帮助命令"""
    print("=" * 50)
    print("测试帮助命令")
    print("=" * 50)
    
    commands = [
        ["python", "main.py", "--help"],
        ["python", "main.py", "optimize", "--help"],
        ["python", "main.py", "compare", "--help"]
    ]
    
    for cmd in commands:
        print(f"\n执行: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✓ 命令成功")
                # 检查--evaluator的位置
                if "optimize" in cmd:
                    if "--evaluator" in result.stdout:
                        print("✓ --evaluator在optimize帮助中找到")
                    else:
                        print("❌ --evaluator在optimize帮助中未找到")
                        print("帮助内容:")
                        print(result.stdout[:300] + "...")
            else:
                print(f"❌ 命令失败: {result.stderr}")
                
        except Exception as e:
            print(f"❌ 执行异常: {e}")

def test_argument_parsing():
    """测试参数解析"""
    print("\n" + "=" * 50)
    print("测试参数解析")
    print("=" * 50)
    
    # 创建一个最小的测试版本
    test_script = """
import sys
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Test')
    
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--config', type=str)
    
    subparsers = parser.add_subparsers(dest='command')
    
    optimize_parser = subparsers.add_parser('optimize')
    optimize_parser.add_argument('--optimizer', default='bayesian')
    optimize_parser.add_argument('--evaluator', choices=['ansa', 'mock'], default='mock')
    optimize_parser.add_argument('--n-calls', type=int, default=20)
    
    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--optimizers', nargs='+', default=['bayesian'])
    compare_parser.add_argument('--evaluator', choices=['ansa', 'mock'], default='mock')
    
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(f"成功解析参数: {args}")
"""
    
    # 写入测试文件
    with open("test_parser.py", "w") as f:
        f.write(test_script)
    
    # 测试命令
    test_commands = [
        ["python", "test_parser.py", "optimize", "--optimizer", "random", "--evaluator", "mock"],
        ["python", "test_parser.py", "compare", "--optimizers", "random", "genetic", "--evaluator", "mock"],
        ["python", "test_parser.py", "optimize", "--help"]
    ]
    
    for cmd in test_commands:
        print(f"\n测试命令: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✓ 解析成功")
                if result.stdout:
                    print(f"输出: {result.stdout.strip()}")
            else:
                print(f"❌ 解析失败: {result.stderr}")
                
        except Exception as e:
            print(f"❌ 执行异常: {e}")
    
    # 清理测试文件
    try:
        Path("test_parser.py").unlink()
    except:
        pass

def test_actual_main():
    """测试实际的main.py"""
    print("\n" + "=" * 50)
    print("测试实际的main.py")
    print("=" * 50)
    
    # 首先检查文件是否存在
    if not Path("main.py").exists():
        print("❌ main.py文件不存在")
        return
    
    # 测试基本的参数解析
    test_commands = [
        ["python", "main.py", "optimize", "--help"],
        ["python", "main.py", "optimize", "--optimizer", "random", "--n-calls", "2", "--evaluator", "mock"]
    ]
    
    for cmd in test_commands:
        print(f"\n执行: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            print(f"返回码: {result.returncode}")
            
            if "unrecognized arguments" in result.stderr:
                print("❌ 参数未识别")
                print(f"错误: {result.stderr}")
            elif result.returncode == 0 or "help" in cmd:
                print("✓ 参数解析正确")
                if result.stdout:
                    print("输出片段:")
                    lines = result.stdout.split('\n')[:10]
                    for line in lines:
                        print(f"  {line}")
            else:
                print("⚠️ 参数解析可能正确，但执行有其他问题")
                if result.stderr:
                    print("错误信息:")
                    print(result.stderr[:200] + "...")
                    
        except subprocess.TimeoutExpired:
            print("⚠️ 命令超时（可能在正常执行）")
        except Exception as e:
            print(f"❌ 执行异常: {e}")

def main():
    """主函数"""
    print("直接测试修复")
    print("=" * 60)
    
    test_help_commands()
    test_argument_parsing()
    test_actual_main()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    print("\n如果看到参数解析正确的消息，可以尝试运行:")
    print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")

if __name__ == "__main__":
    main()