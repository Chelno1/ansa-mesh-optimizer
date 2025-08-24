#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试显示配置重构
"""

import os
from unittest.mock import patch

import pytest

from src.utils.display_config import (
    DisplayConfig,
    display_config,
    is_no_display_mode,
    set_no_display_mode,
)


class TestDisplayConfig:
    """测试DisplayConfig类"""

    def test_display_config_init(self):
        """测试DisplayConfig初始化"""
        # 测试默认初始化
        config = DisplayConfig()
        assert not config.no_display_mode

        # 测试指定无头模式
        config = DisplayConfig(no_display=True)
        assert config.no_display_mode

    @patch("matplotlib.use")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_display_config_context_manager(self, mock_close, mock_show, mock_use):
        """测试DisplayConfig上下文管理器"""
        # 测试无头模式上下文
        with DisplayConfig(no_display=True) as config:
            assert config.no_display_mode
            # 在无头模式下应该调用matplotlib.use('Agg')
            mock_use.assert_called_with("Agg")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_safe_methods(self, mock_close, mock_show):
        """测试safe_show和safe_close方法"""
        config = DisplayConfig(no_display=False)
        config.safe_show()
        mock_show.assert_called_once()

        config.safe_close()
        mock_close.assert_called_once()

        # 测试无头模式下不显示
        config = DisplayConfig(no_display=True)
        config.safe_show()  # 应该不调用plt.show()

    def test_display_config_function(self):
        """测试display_config上下文管理器函数"""
        with display_config(no_display=True) as config:
            assert isinstance(config, DisplayConfig)
            assert config.no_display_mode

        with display_config(no_display=False) as config:
            assert isinstance(config, DisplayConfig)
            assert not config.no_display_mode


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_set_no_display_mode_compatibility(self):
        """测试set_no_display_mode函数的向后兼容性"""
        # 这些函数应该仍然可以工作，但使用新的实现
        set_no_display_mode(True)
        assert is_no_display_mode()

        set_no_display_mode(False)
        assert not is_no_display_mode()

    def test_configure_matplotlib_for_display_compatibility(self):
        """测试configure_matplotlib_for_display函数的向后兼容性"""
        from ansa_mesh_optimizer.utils.display_config import configure_matplotlib_for_display

        # 应该能够正常调用而不报错
        try:
            configure_matplotlib_for_display()
            assert True  # 如果没有抛出异常就是成功
        except Exception as e:
            # 允许ImportError (matplotlib未安装)
            if "matplotlib" not in str(e).lower():
                raise

    def test_safe_functions_compatibility(self):
        """测试safe_show和safe_close函数的向后兼容性"""
        from ansa_mesh_optimizer.utils.display_config import safe_close, safe_show

        # 应该能够正常调用而不报错
        try:
            safe_show()
            safe_close()
            assert True  # 如果没有抛出异常就是成功
        except Exception as e:
            # 允许ImportError (matplotlib未安装)
            if "matplotlib" not in str(e).lower():
                raise


class TestThreadSafety:
    """测试线程安全性"""

    def test_thread_local_storage(self):
        """测试线程本地存储"""
        from ansa_mesh_optimizer.utils.display_config import _get_current_no_display_mode, _thread_local

        # 默认情况下应该是False
        assert not _get_current_no_display_mode()

        # 设置线程本地状态
        _thread_local.no_display = True
        assert _get_current_no_display_mode()

        # 清理
        _thread_local.no_display = False


class TestEnvironmentDetection:
    """测试环境检测"""

    def test_should_use_headless_mode_no_display(self):
        """测试在没有DISPLAY环境变量时使用无头模式"""
        from ansa_mesh_optimizer.utils.display_config import _should_use_headless_mode

        # 在Linux系统中，没有DISPLAY应该使用无头模式
        with patch.dict(os.environ, {}, clear=True):  # 清除所有环境变量
            with patch("src.utils.display_config.os.name", "posix"):
                assert _should_use_headless_mode()

    @patch.dict(os.environ, {"CI": "1"})
    def test_should_use_headless_mode_ci(self):
        """测试在CI环境中使用无头模式"""
        from ansa_mesh_optimizer.utils.display_config import _should_use_headless_mode

        assert _should_use_headless_mode()

    @patch.dict(os.environ, {"SSH_CLIENT": "192.168.1.100 55842 22"})
    def test_should_use_headless_mode_ssh(self):
        """测试在SSH连接中使用无头模式"""
        from ansa_mesh_optimizer.utils.display_config import _should_use_headless_mode

        assert _should_use_headless_mode()


if __name__ == "__main__":
    pytest.main([__file__])
