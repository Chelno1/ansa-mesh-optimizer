#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyInstaller运行时钩子 - matplotlib和字体配置
"""

import os
import sys
import platform
from pathlib import Path

def setup_matplotlib_backend():
    """设置matplotlib后端为无头模式"""
    os.environ['MPLBACKEND'] = 'Agg'
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        import matplotlib.pyplot as plt
        plt.ioff()  # 关闭交互模式
        
        print("✓ matplotlib后端设置为Agg（无头模式）")
    except ImportError as e:
        print(f"⚠️ matplotlib导入失败: {e}")
    except Exception as e:
        print(f"⚠️ matplotlib配置失败: {e}")

def setup_windows_fonts():
    """Windows环境下的字体配置"""
    if platform.system() != 'Windows':
        return
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # Windows中文字体优先级
        windows_fonts = [
            'Microsoft YaHei',
            'SimHei',
            'SimSun',
            'KaiTi',
            'FangSong'
        ]
        
        # 获取可用字体
        try:
            available_fonts = [f.name for f in fm.fontManager.ttflist]
        except Exception:
            # 如果字体管理器失败，使用默认字体
            available_fonts = []
        
        # 查找可用的中文字体
        found_font = None
        for font in windows_fonts:
            if font in available_fonts:
                found_font = font
                break
        
        # 设置字体配置
        if found_font:
            font_list = [found_font] + ['DejaVu Sans', 'Arial', 'sans-serif']
        else:
            font_list = ['DejaVu Sans', 'Arial', 'sans-serif']
        
        # 安全地设置字体参数
        try:
            plt.rcParams['font.sans-serif'] = font_list
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'sans-serif'
            
            # 同时设置matplotlib全局参数
            matplotlib.rcParams['font.sans-serif'] = font_list
            matplotlib.rcParams['axes.unicode_minus'] = False
            matplotlib.rcParams['font.family'] = 'sans-serif'
        except Exception:
            pass  # 忽略字体设置错误
        
        if found_font:
            print(f"✓ Windows中文字体设置: {found_font}")
        else:
            print("⚠️ 未找到中文字体，使用默认字体")
            
    except ImportError:
        print("⚠️ matplotlib未安装，跳过字体配置")
    except Exception as e:
        print(f"⚠️ 字体设置失败: {e}")

def setup_temp_directory():
    """设置临时目录权限"""
    try:
        import tempfile
        
        # 确保临时目录可写
        temp_dir = Path(tempfile.gettempdir())
        app_temp_dir = temp_dir / 'ansa-mesh-optimizer'
        
        if not app_temp_dir.exists():
            app_temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建应用临时目录: {app_temp_dir}")
        
        # 设置环境变量
        os.environ['ANSA_OPTIMIZER_TEMP'] = str(app_temp_dir)
        
    except Exception as e:
        print(f"⚠️ 临时目录设置失败: {e}")

def main():
    """主初始化函数"""
    print("=" * 50)
    print("ANSA网格优化器 - 运行时初始化")
    print("=" * 50)
    
    # 设置matplotlib后端（如果可用）
    try:
        setup_matplotlib_backend()
    except Exception as e:
        print(f"⚠️ matplotlib后端设置跳过: {e}")
    
    # 设置Windows字体（如果可用）
    try:
        setup_windows_fonts()
    except Exception as e:
        print(f"⚠️ 字体设置跳过: {e}")
    
    # 设置临时目录
    try:
        setup_temp_directory()
    except Exception as e:
        print(f"⚠️ 临时目录设置跳过: {e}")
    
    print("=" * 50)
    print("运行时初始化完成")
    print("=" * 50)

if __name__ == '__main__':
    main()