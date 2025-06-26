#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体装饰器 - 自动应用中文字体设置

作者: Chel
创建日期: 2025-06-25
版本: 1.0.0
"""

import functools
import logging

logger = logging.getLogger(__name__)

def with_chinese_font(func):
    """
    装饰器：自动应用中文字体设置
    
    用法：
        @with_chinese_font
        def plot_some_chart():
            # 绘图代码，中文字体会自动应用
            plt.title("中文标题")
            plt.xlabel("中文X轴")
            plt.ylabel("中文Y轴")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from utils.font_config import apply_chinese_font_settings, FONT_CONFIGURED
            if FONT_CONFIGURED:
                apply_chinese_font_settings()
                logger.debug(f"已为函数 {func.__name__} 应用中文字体设置")
            else:
                logger.warning(f"字体未正确配置，函数 {func.__name__} 可能无法正确显示中文")
        except ImportError:
            logger.warning(f"字体配置模块未找到，函数 {func.__name__} 使用备用字体设置")
            # 备用字体设置
            try:
                import matplotlib.pyplot as plt
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as e:
                logger.error(f"备用字体设置失败: {e}")
        except Exception as e:
            logger.warning(f"应用字体设置失败: {e}")
        
        return func(*args, **kwargs)
    
    return wrapper

def with_chinese_font_context(save_original=True):
    """
    增强版装饰器：支持保存和恢复原始字体设置
    
    Args:
        save_original: 是否保存并恢复原始字体设置
    
    用法：
        @with_chinese_font_context(save_original=True)
        def plot_chart():
            # 绘图代码
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_settings = {}
            
            try:
                import matplotlib.pyplot as plt
                
                # 保存原始设置
                if save_original:
                    original_settings = {
                        'font.sans-serif': plt.rcParams['font.sans-serif'].copy(),
                        'axes.unicode_minus': plt.rcParams['axes.unicode_minus']
                    }
                
                # 应用中文字体设置
                from utils.font_config import apply_chinese_font_settings
                apply_chinese_font_settings()
                
                # 执行原函数
                result = func(*args, **kwargs)
                
                return result
                
            except ImportError:
                logger.warning("字体配置模块未找到，使用备用设置")
                try:
                    import matplotlib.pyplot as plt
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"备用字体设置失败: {e}")
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"字体设置失败: {e}")
                return func(*args, **kwargs)
            finally:
                # 恢复原始设置
                if save_original and original_settings:
                    try:
                        import matplotlib.pyplot as plt
                        for key, value in original_settings.items():
                            plt.rcParams[key] = value
                    except Exception as e:
                        logger.warning(f"恢复原始字体设置失败: {e}")
        
        return wrapper
    return decorator

def ensure_matplotlib_backend(backend='Agg'):
    """
    装饰器：确保matplotlib使用指定的后端
    
    Args:
        backend: matplotlib后端名称
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import matplotlib
                original_backend = matplotlib.get_backend()
                
                if original_backend != backend:
                    matplotlib.use(backend)
                    logger.debug(f"切换matplotlib后端从 {original_backend} 到 {backend}")
                
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.warning(f"设置matplotlib后端失败: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# 组合装饰器：同时应用中文字体和后端设置
def plotting_ready(backend='Agg', save_original=True):
    """
    组合装饰器：为绘图函数做好完整准备
    
    Args:
        backend: matplotlib后端
        save_original: 是否保存原始字体设置
    """
    def decorator(func):
        # 组合多个装饰器
        func = ensure_matplotlib_backend(backend)(func)
        func = with_chinese_font_context(save_original)(func)
        return func
    return decorator