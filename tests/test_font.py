#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体配置测试脚本
"""

def test_font_configuration():
    """测试字体配置"""
    print("="*50)
    print("字体配置测试")
    print("="*50)
    
    try:
        # 导入字体配置模块
        from font_config import (
            CHINESE_FONT, FONT_CONFIGURED, 
            test_chinese_display,
            get_available_chinese_fonts,
            apply_chinese_font_settings
        )
        
        print(f"1. 字体配置状态: {'成功' if FONT_CONFIGURED else '失败'}")
        print(f"2. 选用的中文字体: {CHINESE_FONT}")
        
        # 显示可用的中文字体
        available_fonts = get_available_chinese_fonts()
        print(f"3. 系统可用中文字体: {len(available_fonts)} 个")
        for font in available_fonts[:5]:  # 只显示前5个
            print(f"   - {font}")
        if len(available_fonts) > 5:
            print(f"   ... 还有 {len(available_fonts) - 5} 个字体")
        
        # 测试中文显示
        print("\n4. 测试中文图表生成...")
        if test_chinese_display():
            print("   ✓ 中文图表测试成功")
        else:
            print("   ❌ 中文图表测试失败")
        
        # 测试matplotlib配置
        print("\n5. matplotlib配置检查:")
        import matplotlib.pyplot as plt
        current_font = plt.rcParams['font.sans-serif']
        print(f"   当前字体设置: {current_font}")
        print(f"   负号显示设置: {plt.rcParams['axes.unicode_minus']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 无法导入字体配置模块: {e}")
        return False
    except Exception as e:
        print(f"❌ 字体配置测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_font_configuration()
    if success:
        print("\n🎉 字体配置测试完成！")
    else:
        print("\n💥 字体配置存在问题，请检查配置。")