"""
显示配置模块 - 统一管理matplotlib显示设置
"""

import os
import logging
from typing import Optional
from contextlib import contextmanager
from threading import local

logger = logging.getLogger(__name__)

# 线程本地存储，避免全局状态问题
_thread_local = local()


class DisplayConfig:
    """
    显示配置类 - 管理matplotlib显示模式
    
    使用上下文管理器模式来控制显示状态，避免全局变量带来的问题
    """
    
    def __init__(self, no_display: bool = False):
        """
        初始化显示配置
        
        Args:
            no_display: 是否启用无头模式
        """
        self._no_display = no_display
        self._original_backend = None
        self._original_env = None
        self._matplotlib_configured = False
        
    @property
    def no_display_mode(self) -> bool:
        """获取当前无头模式状态"""
        return self._no_display
        
    def __enter__(self):
        """进入上下文管理器"""
        # 保存当前状态
        self._save_current_state()
        
        if self._no_display:
            self._activate_no_display_mode()
        else:
            self._configure_display_mode()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        # 恢复原始状态
        self._restore_state()
        
    def _save_current_state(self):
        """保存当前matplotlib状态"""
        try:
            import matplotlib
            self._original_backend = matplotlib.get_backend()
        except ImportError:
            pass
            
        # 保存环境变量
        self._original_env = os.environ.get('MPLBACKEND')
        
    def _restore_state(self):
        """恢复原始状态"""
        if self._original_env is not None:
            os.environ['MPLBACKEND'] = self._original_env
        elif 'MPLBACKEND' in os.environ:
            del os.environ['MPLBACKEND']
            
        if self._original_backend:
            try:
                import matplotlib
                matplotlib.use(self._original_backend)
            except ImportError:
                pass
        
    def _activate_no_display_mode(self):
        """激活无头模式"""
        # 设置环境变量
        os.environ['MPLBACKEND'] = 'Agg'
        
        # 设置matplotlib后端
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            # 设置线程本地状态
            _thread_local.no_display = True
            self._matplotlib_configured = True
            
            logger.info("已设置matplotlib为无头模式 (Agg后端)")
        except ImportError:
            logger.warning("matplotlib未安装，跳过后端设置")
            
    def _configure_display_mode(self):
        """配置显示模式"""
        # 设置线程本地状态
        _thread_local.no_display = False
        
        try:
            import matplotlib
            
            # 自动检测是否需要无头模式
            if _should_use_headless_mode():
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.ioff()
                logger.debug("自动检测到无头环境，使用Agg后端")
            else:
                # 确保有合适的后端
                current_backend = matplotlib.get_backend()
                if current_backend.lower() in ['agg', 'svg', 'pdf', 'ps']:
                    # 已经是非交互后端，保持不变
                    pass
                else:
                    # 尝试使用交互后端，如果失败则回退到Agg
                    try:
                        import matplotlib.pyplot as plt
                        plt.ion()
                    except Exception:
                        logger.warning("交互模式配置失败，回退到无头模式")
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        plt.ioff()
                
                logger.debug(f"matplotlib配置完成，后端: {matplotlib.get_backend()}")
                
            self._matplotlib_configured = True
            
        except ImportError:
            logger.warning("matplotlib未安装，跳过配置")

    def safe_show(self):
        """
        安全的显示函数 - 在无头模式下不显示
        """
        if not self._no_display:
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except ImportError:
                logger.warning("matplotlib未安装，无法显示图表")
        else:
            logger.debug("无头模式下跳过图表显示")

    def safe_close(self):
        """
        安全的关闭函数
        """
        try:
            import matplotlib.pyplot as plt
            plt.close()
        except ImportError:
            pass


# 默认配置实例
_default_config = DisplayConfig()


@contextmanager
def display_config(no_display: bool = False):
    """
    显示配置上下文管理器
    
    Args:
        no_display: 是否启用无头模式
        
    Usage:
        with display_config(no_display=True):
            # 在这个上下文中matplotlib使用无头模式
            plt.plot([1, 2, 3])
            plt.savefig('plot.png')
    """
    config = DisplayConfig(no_display=no_display)
    with config:
        yield config


def _get_current_no_display_mode() -> bool:
    """获取当前线程的无头模式状态"""
    return getattr(_thread_local, 'no_display', False)


def _should_use_headless_mode():
    """
    自动检测是否应该使用无头模式
    """
    # 检查环境变量
    if os.environ.get('DISPLAY') is None and os.name != 'nt':
        return True
    
    # 检查是否在CI环境中
    ci_indicators = ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'GITLAB_CI']
    if any(os.environ.get(indicator) for indicator in ci_indicators):
        return True
    
    # 检查是否通过SSH连接
    if os.environ.get('SSH_CLIENT') or os.environ.get('SSH_TTY'):
        return True
    
    return False


# 向后兼容的函数接口
def set_no_display_mode(enabled: bool = True):
    """
    设置无头模式 (向后兼容函数)
    
    Args:
        enabled: 是否启用无头模式
        
    注意: 推荐使用 DisplayConfig 类或 display_config() 上下文管理器
    """
    global _default_config
    _default_config = DisplayConfig(no_display=enabled)
    _default_config._activate_no_display_mode() if enabled else _default_config._configure_display_mode()


def is_no_display_mode() -> bool:
    """
    检查是否为无头模式 (向后兼容函数)
    
    Returns:
        bool: 是否为无头模式
    """
    return _get_current_no_display_mode()


def configure_matplotlib_for_display():
    """
    根据当前模式配置matplotlib (向后兼容函数)
    """
    try:
        import matplotlib
        
        # 检查线程本地状态
        if _get_current_no_display_mode() or _should_use_headless_mode():
            matplotlib.use('Agg')
            
            import matplotlib.pyplot as plt
            # 禁用交互模式
            plt.ioff()
            
            logger.debug("matplotlib已配置为无头模式")
        else:
            # 确保有合适的后端
            current_backend = matplotlib.get_backend()
            if current_backend.lower() in ['agg', 'svg', 'pdf', 'ps']:
                # 已经是非交互后端，保持不变
                pass
            else:
                # 尝试使用交互后端，如果失败则回退到Agg
                try:
                    import matplotlib.pyplot as plt
                    plt.ion()
                except Exception:
                    logger.warning("交互模式配置失败，回退到无头模式")
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    plt.ioff()
            
            logger.debug(f"matplotlib配置完成，后端: {matplotlib.get_backend()}")
            
    except ImportError:
        logger.warning("matplotlib未安装，跳过配置")


def safe_show():
    """
    安全的显示函数 - 在无头模式下不显示 (向后兼容函数)
    """
    if not _get_current_no_display_mode():
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            logger.warning("matplotlib未安装，无法显示图表")
    else:
        logger.debug("无头模式下跳过图表显示")


def safe_close():
    """
    安全的关闭函数 (向后兼容函数)
    """
    try:
        import matplotlib.pyplot as plt
        plt.close()
    except ImportError:
        pass
