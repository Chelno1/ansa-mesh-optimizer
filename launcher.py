#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyInstaller启动包装器
解决相对导入问题的统一解决方案

作者: Chel
创建日期: 2025-08-31
"""

import sys
import os


def setup_paths():
    """设置Python模块搜索路径"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller打包后的环境
        base_path = sys._MEIPASS
        print(f"PyInstaller环境检测到，基础路径: {base_path}")
        
        # 导入并使用导入修复器
        try:
            # 先添加基本路径
            src_path = os.path.join(base_path, 'src')
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            if base_path not in sys.path:
                sys.path.insert(0, base_path)
                
            # 尝试导入修复器
            import import_fixer
            import_fixer.fix_module_paths()
            print("导入修复器已启用")
        except ImportError:
            print("导入修复器不可用，使用基础路径设置")
            
    else:
        # 开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"开发环境，基础路径: {base_path}")
        
        # 添加src目录到Python路径
        src_path = os.path.join(base_path, 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            print(f"已添加src路径: {src_path}")
        
        # 添加项目根目录到Python路径
        if base_path not in sys.path:
            sys.path.insert(0, base_path)
            print(f"已添加根目录: {base_path}")
    
    return base_path, src_path


def main():
    """主函数 - 设置路径后调用原始main函数"""
    try:
        print("=== ANSA Mesh Optimizer Launcher ===")
        
        # 设置路径
        base_path, src_path = setup_paths()
        
        # 验证路径 - 检查多个可能的位置
        main_py_locations = [
            os.path.join(src_path, 'main.py'),
            os.path.join(base_path, 'src', 'main.py'),
            os.path.join(base_path, 'main.py')
        ]
        
        main_py_found = False
        for main_py_path in main_py_locations:
            if os.path.exists(main_py_path):
                print(f"找到main.py文件: {main_py_path}")
                main_py_found = True
                break
        
        if not main_py_found:
            print("错误：在以下位置都找不到main.py文件:")
            for path in main_py_locations:
                print(f"  - {path}")
            print("\n当前目录内容:")
            if hasattr(sys, '_MEIPASS'):
                for root, dirs, files in os.walk(sys._MEIPASS):
                    level = root.replace(sys._MEIPASS, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            return 1
        
        print("路径设置完成，启动主程序...")
        
        # 导入并运行原始main函数
        try:
            from src.main import main as original_main
        except ImportError:
            # 如果src.main导入失败，尝试直接导入main
            sys.path.insert(0, os.path.dirname(main_py_path))
            import main as main_module
            original_main = main_module.main
        
        return original_main()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("当前Python路径:")
        for i, path in enumerate(sys.path):
            print(f"  {i}: {path}")
        return 1
    except Exception as e:
        print(f"启动器异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n用户中断程序")
        sys.exit(130)
    except Exception as e:
        print(f"未捕获的异常: {e}")
        sys.exit(1)