"""
显示配置模块 - 统一管理matplotlib显示设置
"""

import os
import logging

logger = logging.getLogger(__name__)

# 全局无头模式标志
_NO_DISPLAY_MODE = False

def set_no_display_mode(enabled: bool = True):
    """
    设置无头模式
    
    Args:
        enabled: 是否启用无头模式
    """
    global _NO_DISPLAY_MODE
    _NO_DISPLAY_MODE = enabled
    
    if enabled:
        # 设置环境变量
        os.environ['MPLBACKEND'] = 'Agg'
        
        # 设置matplotlib后端
        try:
            import matplotlib
            matplotlib.use('Agg')
            logger.info("已设置matplotlib为无头模式 (Agg后端)")
        except ImportError:
            logger.warning("matplotlib未安装，跳过后端设置")

def is_no_display_mode() -> bool:
    """
    检查是否为无头模式
    
    Returns:
        bool: 是否为无头模式
    """
    return _NO_DISPLAY_MODE

def configure_matplotlib_for_display():
    """
    根据当前模式配置matplotlib
    """
    if _NO_DISPLAY_MODE:
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            import matplotlib.pyplot as plt
            # 禁用交互模式
            plt.ioff()
            
            logger.debug("matplotlib已配置为无头模式")
        except ImportError:
            logger.warning("matplotlib未安装，跳过配置")

def safe_show():
    """
    安全的显示函数 - 在无头模式下不显示
    """
    if not _NO_DISPLAY_MODE:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            logger.warning("matplotlib未安装，无法显示图表")
    else:
        logger.debug("无头模式下跳过图表显示")

def safe_close():
    """
    安全的关闭函数
    """
    try:
        import matplotlib.pyplot as plt
        plt.close()
    except ImportError:
        pass