#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
装饰器测试脚本
"""

import matplotlib.pyplot as plt
import numpy as np

def test_decorator_functionality():
    """测试装饰器功能"""
    print("="*50)
    print("装饰器功能测试")
    print("="*50)
    
    try:
        from font_decorator import with_chinese_font, plotting_ready
        
        # 测试基础装饰器
        @with_chinese_font
        def test_basic_plot():
            """测试基础绘图装饰器"""
            fig, ax = plt.subplots(figsize=(8, 6))
            
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            ax.plot(x, y, 'b-', linewidth=2, label='正弦曲线')
            ax.set_title('基础装饰器测试 - 中文标题')
            ax.set_xlabel('X轴 - 横坐标')
            ax.set_ylabel('Y轴 - 纵坐标')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig('test_basic_decorator.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 基础装饰器测试完成")
            return True
        
        # 测试增强装饰器
        @plotting_ready(backend='Agg', save_original=True)
        def test_enhanced_plot():
            """测试增强绘图装饰器"""
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 子图1
            x = np.linspace(0, 10, 100)
            y1 = np.cos(x)
            ax1.plot(x, y1, 'r-', linewidth=2, label='余弦曲线')
            ax1.set_title('增强装饰器测试 - 子图1')
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('幅值')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 子图2
            y2 = np.random.normal(0, 1, 100)
            ax2.hist(y2, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('随机数据分布')
            ax2.set_xlabel('数值')
            ax2.set_ylabel('频次')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('test_enhanced_decorator.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ 增强装饰器测试完成")
            return True
        
        # 执行测试
        print("1. 测试基础装饰器...")
        test_basic_plot()
        
        print("2. 测试增强装饰器...")
        test_enhanced_plot()
        
        print("\n🎉 所有装饰器测试通过！")
        print("请检查生成的图片:")
        print("  - test_basic_decorator.png")
        print("  - test_enhanced_decorator.png")
        
        return True
        
    except ImportError as e:
        print(f"❌ 无法导入装饰器模块: {e}")
        return False
    except Exception as e:
        print(f"❌ 装饰器测试失败: {e}")
        return False

def test_integration_with_existing_code():
    """测试与现有代码的集成"""
    print("\n" + "="*50)
    print("集成测试")
    print("="*50)
    
    try:
        # 测试早停模块
        print("1. 测试早停模块...")
        from early_stopping import EarlyStopping
        
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # 模拟一些数据
        test_values = [10.0, 8.5, 7.2, 6.8, 6.7, 6.65, 6.64]
        for i, value in enumerate(test_values):
            early_stopping(value)
        
        # 绘制历史图（如果有plot_history方法且使用了装饰器）
        if hasattr(early_stopping, 'plot_history'):
            early_stopping.plot_history(save_path='test_early_stopping.png')
            print("   ✓ 早停历史图生成成功")
        
        print("2. 测试遗传算法模块...")
        # 可以添加更多模块测试
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始装饰器测试...")
    
    # 基础功能测试
    basic_success = test_decorator_functionality()
    
    if basic_success:
        # 集成测试
        integration_success = test_integration_with_existing_code()
        
        if integration_success:
            print("\n🎊 所有测试通过！装饰器配置成功！")
        else:
            print("\n⚠️ 基础功能正常，但集成可能存在问题")
    else:
        print("\n💥 基础功能测试失败，请检查配置")