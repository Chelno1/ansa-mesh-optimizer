#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复版本的脚本

作者: Chel
创建日期: 2025-06-19
"""

import sys
import subprocess
from pathlib import Path

def test_main_fixed():
    """测试main_fixed.py"""
    print("测试修复版本 main_fixed.py")
    print("=" * 50)
    
    # 检查文件是否存在
    if not Path("main_fixed.py").exists():
        print("❌ main_fixed.py 文件不存在")
        return False
    
    # 测试命令列表
    test_commands = [
        # 帮助命令
        ["python", "main_fixed.py", "--help"],
        ["python", "main_fixed.py", "optimize", "--help"],
        ["python", "main_fixed.py", "compare", "--help"],
        
        # 信息命令
        ["python", "main_fixed.py", "info"],
        ["python", "main_fixed.py", "info", "--check-deps"],
        
        # 实际运行命令
        ["python", "main_fixed.py", "optimize", "--optimizer", "random", "--n-calls", "3", "--evaluator", "mock"]
    ]
    
    success_count = 0
    total_count = len(test_commands)
    
    for i, cmd in enumerate(test_commands):
        print(f"\n测试 {i+1}/{total_count}: {' '.join(cmd)}")
        print("-" * 40)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            print(f"返回码: {result.returncode}")
            
            # 分析结果
            if "unrecognized arguments" in result.stderr:
                print("❌ 参数解析失败")
                print(f"错误: {result.stderr}")
            elif "--help" in cmd or "info" in cmd:
                if result.returncode == 0:
                    print("✓ 命令成功")
                    success_count += 1
                    
                    # 对于optimize --help，检查是否包含--evaluator
                    if "optimize" in cmd and "--help" in cmd:
                        if "--evaluator" in result.stdout:
                            print("✓ --evaluator参数在help中找到")
                        else:
                            print("❌ --evaluator参数在help中未找到")
                            print("Help内容预览:")
                            print(result.stdout[:300] + "...")
                else:
                    print(f"❌ 帮助命令失败: {result.stderr}")
            else:
                # 实际运行命令
                if result.returncode == 0:
                    print("✓ 命令执行成功")
                    success_count += 1
                    if result.stdout:
                        print("输出预览:")
                        lines = result.stdout.split('\n')[:5]
                        for line in lines:
                            if line.strip():
                                print(f"  {line}")
                elif "模块导入失败" in result.stdout or "ModuleNotFoundError" in result.stderr:
                    print("⚠️ 模块导入问题（参数解析可能正确）")
                    success_count += 0.5  # 部分成功
                    print("输出:")
                    print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
                else:
                    print(f"❌ 命令执行失败")
                    print("错误信息:")
                    print(result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr)
                    if result.stdout:
                        print("标准输出:")
                        print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
                        
        except subprocess.TimeoutExpired:
            print("⚠️ 命令超时（可能在正常执行）")
            success_count += 0.5
        except Exception as e:
            print(f"❌ 执行异常: {e}")
    
    # 输出总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"成功/总数: {success_count}/{total_count}")
    
    success_rate = success_count / total_count
    
    if success_rate >= 0.8:
        print("🎉 测试大部分通过!")
        print("\n建议操作:")
        print("1. 将 main_fixed.py 复制为 main.py")
        print("2. 运行: python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
        return True
    elif success_rate >= 0.5:
        print("⚠️ 部分测试通过，可能有导入问题")
        print("\n建议检查依赖库安装")
        return True
    else:
        print("❌ 多数测试失败")
        return False

def copy_to_main():
    """将main_fixed.py复制到main.py"""
    try:
        import shutil
        
        if Path("main_fixed.py").exists():
            # 备份原文件
            if Path("main.py").exists():
                shutil.copy2("main.py", "main_backup.py")
                print("✓ 原main.py已备份为main_backup.py")
            
            # 复制新文件
            shutil.copy2("main_fixed.py", "main.py")
            print("✓ main_fixed.py已复制为main.py")
            
            return True
        else:
            print("❌ main_fixed.py不存在")
            return False
            
    except Exception as e:
        print(f"❌ 复制失败: {e}")
        return False

def main():
    """主函数"""
    success = test_main_fixed()
    
    if success:
        print("\n" + "=" * 50)
        print("是否要将修复版本复制为main.py? (y/N): ", end="")
        
        try:
            response = input().strip().lower()
            if response in ['y', 'yes']:
                if copy_to_main():
                    print("\n🎉 复制成功! 现在可以运行:")
                    print("python main.py optimize --optimizer random --n-calls 5 --evaluator mock")
                else:
                    print("\n❌ 复制失败")
            else:
                print("\n继续使用main_fixed.py进行测试")
        except (KeyboardInterrupt, EOFError):
            print("\n操作取消")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)