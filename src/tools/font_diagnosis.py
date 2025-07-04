#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体问题诊断脚本
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
from pathlib import Path

def diagnose_font_issues():
    """诊断字体问题"""
    print("="*60)
    print("字体问题诊断")
    print("="*60)
    
    # 1. 系统信息
    print(f"1. 系统信息:")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   Python版本: {platform.python_version()}")
    print(f"   Matplotlib版本: {matplotlib.__version__}")
    print(f"   Matplotlib后端: {matplotlib.get_backend()}")
    
    # 2. 当前字体配置
    print(f"\n2. 当前matplotlib字体配置:")
    print(f"   font.family: {plt.rcParams['font.family']}")
    print(f"   font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"   axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")
    
    # 3. 系统可用字体检查
    print(f"\n3. 系统可用字体检查:")
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"   总字体数: {len(available_fonts)}")
    
    # 检查常见中文字体
    chinese_fonts = [
        'SimHei', 'SimSun', 'Microsoft YaHei', 'KaiTi', 'FangSong',
        'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS', 'PingFang SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
        'Source Han Sans CN', 'DejaVu Sans'
    ]
    
    print(f"   检查常见中文字体:")
    found_fonts = []
    for font in chinese_fonts:
        if font in available_fonts:
            found_fonts.append(font)
            print(f"   ✓ {font}")
        else:
            print(f"   ✗ {font}")
    
    # 4. 字体文件路径检查
    print(f"\n4. 字体文件路径:")
    font_dirs = fm.get_fontconfig_fonts()
    print(f"   字体目录数量: {len(font_dirs) if font_dirs else 0}")
    
    # matplotlib字体目录
    mpl_data_dir = matplotlib.get_data_path()
    mpl_font_dir = Path(mpl_data_dir) / "fonts" / "ttf"
    print(f"   Matplotlib字体目录: {mpl_font_dir}")
    print(f"   目录存在: {mpl_font_dir.exists()}")
    
    if mpl_font_dir.exists():
        ttf_files = list(mpl_font_dir.glob("*.ttf"))
        print(f"   TTF文件数量: {len(ttf_files)}")
        for ttf in ttf_files[:5]:  # 只显示前5个
            print(f"     - {ttf.name}")
    
    # 5. 字体缓存信息
    print(f"\n5. 字体缓存信息:")
    cache_dir = fm.get_cachedir()
    print(f"   缓存目录: {cache_dir}")
    cache_file = Path(cache_dir) / "fontlist-v330.json"  # 版本号可能不同
    cache_files = list(Path(cache_dir).glob("fontlist*.json"))
    print(f"   缓存文件: {[f.name for f in cache_files]}")
    
    return found_fonts

def test_font_rendering():
    """测试字体渲染"""
    print(f"\n6. 字体渲染测试:")
    
    # 测试不同字体设置
    test_configs = [
        ['SimHei'],
        ['Microsoft YaHei'],
        ['DejaVu Sans'],
        ['Arial Unicode MS'],
        ['sans-serif'],
    ]
    
    for i, fonts in enumerate(test_configs):
        try:
            plt.rcParams['font.sans-serif'] = fonts
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, '中文测试 Chinese Test', 
                   fontsize=16, ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'字体配置 {i+1}: {fonts[0]}')
            
            filename = f'font_test_{i+1}_{fonts[0].replace(" ", "_")}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ 配置 {i+1} ({fonts[0]}): 已生成 {filename}")
            
        except Exception as e:
            print(f"   ✗ 配置 {i+1} ({fonts[0]}): 失败 - {e}")

def force_font_setup():
    """强制字体设置"""
    print(f"\n7. 强制字体设置:")
    
    try:
        # 清理字体缓存
        cache_dir = fm.get_cachedir()
        cache_files = list(Path(cache_dir).glob("fontlist*.json"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                print(f"   已删除缓存文件: {cache_file.name}")
            except Exception as e:
                print(f"   删除缓存文件失败: {e}")
        
        # 重建字体缓存
        fm._rebuild()
        print(f"   ✓ 字体缓存已重建")
        
        # 强制设置字体
        if platform.system() == 'Windows':
            fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
        elif platform.system() == 'Darwin':  # macOS
            fonts = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']
        else:  # Linux
            fonts = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
        
        plt.rcParams['font.sans-serif'] = fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        print(f"   ✓ 强制设置字体: {fonts}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 强制字体设置失败: {e}")
        return False

if __name__ == "__main__":
    # 诊断字体问题
    found_fonts = diagnose_font_issues()
    
    # 测试字体渲染
    test_font_rendering()
    
    # 强制字体设置
    force_font_setup()
    
    print(f"\n" + "="*60)
    print("诊断完成")
    print("="*60)
    
    if found_fonts:
        print(f"✓ 找到 {len(found_fonts)} 个中文字体")
        print(f"建议使用: {found_fonts[0]}")
    else:
        print("❌ 未找到中文字体，需要安装")