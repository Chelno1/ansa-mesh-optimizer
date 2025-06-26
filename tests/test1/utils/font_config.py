#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体配置模块 - 增强版本
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import platform
import logging
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

def get_all_available_fonts():
    """获取所有可用字体"""
    try:
        return [f.name for f in fm.fontManager.ttflist]
    except Exception as e:
        logger.error(f"获取字体列表失败: {e}")
        return []

def find_chinese_fonts():
    """查找中文字体"""
    available_fonts = get_all_available_fonts()
    
    # 扩展的中文字体检测
    chinese_patterns = [
        # 完全匹配
        'SimHei', 'SimSun', 'Microsoft YaHei', 'KaiTi', 'FangSong',
        'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS', 'PingFang SC', 'Heiti SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
        'Source Han Sans CN', 'Source Han Sans SC',
        
        # 部分匹配模式
    ]
    
    found_fonts = []
    
    # 完全匹配
    for font in chinese_patterns:
        if font in available_fonts:
            found_fonts.append(font)
    
    # 模糊匹配
    chinese_keywords = ['YaHei', 'SimHei', 'SimSun', 'Hiragino', 'WenQuanYi', 'Noto', 'Source Han', 'PingFang']
    for font in available_fonts:
        for keyword in chinese_keywords:
            if keyword.lower() in font.lower() and font not in found_fonts:
                found_fonts.append(font)
                break
    
    return found_fonts

def force_set_chinese_font():
    """强制设置中文字体"""
    try:
        # 清理字体缓存
        cache_dir = fm.get_cachedir()
        cache_files = list(Path(cache_dir).glob("fontlist*.json"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"已删除字体缓存: {cache_file}")
            except:
                pass
        
        # 重建字体缓存
        fm._rebuild()
        logger.info("字体缓存已重建")
        
        # 根据系统设置字体
        system = platform.system()
        
        if system == 'Windows':
            # Windows系统
            fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        elif system == 'Darwin':
            # macOS系统
            fonts = ['Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti', 'PingFang SC']
        else:
            # Linux系统
            fonts = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
        
        # 添加通用备选字体
        fonts.extend(['sans-serif', 'Arial', 'Liberation Sans'])
        
        # 设置matplotlib参数
        plt.rcParams['font.sans-serif'] = fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 强制设置全局字体参数
        matplotlib.rcParams['font.sans-serif'] = fonts
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.family'] = 'sans-serif'
        
        logger.info(f"强制设置字体: {fonts[:3]}")
        return fonts[0]
        
    except Exception as e:
        logger.error(f"强制设置字体失败: {e}")
        return None

def setup_chinese_fonts():
    """设置中文字体 - 增强版本"""
    logger.info("开始设置中文字体...")
    
    # 首先尝试查找中文字体
    chinese_fonts = find_chinese_fonts()
    
    if chinese_fonts:
        logger.info(f"找到中文字体: {chinese_fonts}")
        selected_font = chinese_fonts[0]
        
        # 设置字体
        fonts = chinese_fonts + ['DejaVu Sans', 'Arial', 'sans-serif']
        
        plt.rcParams['font.sans-serif'] = fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        matplotlib.rcParams['font.sans-serif'] = fonts
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.family'] = 'sans-serif'
        
        logger.info(f"使用中文字体: {selected_font}")
        return selected_font
    
    else:
        logger.warning("未找到中文字体，尝试强制设置...")
        return force_set_chinese_font()

def apply_chinese_font_settings():
    """应用中文字体设置"""
    try:
        # 强制重新设置
        result = setup_chinese_fonts()
        
        # 额外设置
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 10
        
        return result is not None
        
    except Exception as e:
        logger.error(f"应用字体设置失败: {e}")
        return False

def test_chinese_display():
    """测试中文显示效果"""
    try:
        # 确保字体设置生效
        apply_chinese_font_settings()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 测试文本
        test_texts = [
            "优化器性能比较图表",
            "目标值变化曲线分析", 
            "最佳参数结果展示",
            "收敛性分析报告",
            "遗传算法进化过程",
            "参数敏感性分析",
            "Chinese Font Test 中文字体测试"
        ]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'black']
        
        for i, (text, color) in enumerate(zip(test_texts, colors)):
            ax.text(0.1, 0.9 - i*0.11, text, fontsize=14, color=color,
                   transform=ax.transAxes, weight='bold')
        
        # 测试坐标轴标签
        ax.set_title("中文字体显示测试 - 增强版", fontsize=18, weight='bold')
        ax.set_xlabel("X轴标签（参数值）", fontsize=14)
        ax.set_ylabel("Y轴标签（目标函数值）", fontsize=14)
        
        # 添加一些测试数据
        import numpy as np
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2, label='正弦曲线示例')
        ax.legend()
        
        # 移除多余的坐标刻度
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("chinese_font_test_enhanced.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("中文字体测试完成，请检查 chinese_font_test_enhanced.png 文件")
        print("✓ 增强版中文字体测试图片已生成: chinese_font_test_enhanced.png")
        
        # 输出当前字体设置
        print(f"当前字体设置: {plt.rcParams['font.sans-serif']}")
        return True
        
    except Exception as e:
        logger.error(f"中文字体测试失败: {e}")
        print(f"❌ 中文字体测试失败: {e}")
        return False

# 自动设置中文字体
try:
    CHINESE_FONT = setup_chinese_fonts()
    FONT_CONFIGURED = CHINESE_FONT is not None
    if FONT_CONFIGURED:
        print(f"✓ 字体配置成功: {CHINESE_FONT}")
    else:
        print("⚠️ 字体配置可能存在问题")
except Exception as e:
    logger.error(f"字体初始化失败: {e}")
    CHINESE_FONT = None
    FONT_CONFIGURED = False
    print(f"❌ 字体配置失败: {e}")