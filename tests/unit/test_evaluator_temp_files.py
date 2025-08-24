#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试evaluator.temp_files模块

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.evaluators.temp_files import TempFileManager, create_temp_file_manager


class TestTempFileManager:
    """测试TempFileManager类"""

    def setup_method(self):
        """设置测试环境"""
        # 创建mock配置管理器
        self.mock_config_manager = Mock()
        self.mock_ansa_config = Mock()
        self.mock_parameter_replacer = Mock()
        
        self.mock_config_manager.ansa_config = self.mock_ansa_config
        self.mock_ansa_config.mpar_file_pattern = "*.ansa_mpar"

    def test_init_success(self):
        """测试正常初始化"""
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        
        assert manager.config_manager == self.mock_config_manager
        assert manager.config == self.mock_ansa_config
        assert manager.parameter_replacer == self.mock_parameter_replacer
        assert manager.temp_dir is None
        assert manager.temp_files == []

    def test_init_without_config_manager(self):
        """测试缺少配置管理器时的初始化"""
        with pytest.raises(ValueError, match="TempFileManager requires a config_manager instance"):
            TempFileManager(None, self.mock_parameter_replacer)

    def test_init_without_parameter_replacer(self):
        """测试缺少参数替换器时的初始化"""
        with pytest.raises(ValueError, match="TempFileManager requires a parameter_replacer instance"):
            TempFileManager(self.mock_config_manager, None)

    @patch('src.evaluators.temp_files.create_timestamped_temp_dir')
    @patch('src.evaluators.temp_files.copy_mpar_files_to_temp_dir')
    @patch('src.evaluators.temp_files.create_temp_config_in_dir')
    @patch('src.evaluators.temp_files.process_parameter_files_in_temp_dir')
    def test_setup_temp_environment_success(self, mock_process, mock_create_config, 
                                          mock_copy_mpar, mock_create_temp):
        """测试成功设置临时环境"""
        # 设置mock返回值
        temp_dir = "/tmp/test_dir"
        config_file = "/tmp/test_dir/config.json"
        mock_create_temp.return_value = temp_dir
        mock_copy_mpar.return_value = ["file1.ansa_mpar", "file2.ansa_mpar"]
        mock_create_config.return_value = config_file
        
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        params = {"distortion_distance": 20.0}
        
        result_dir = manager.setup_temp_environment(params)
        
        assert result_dir == temp_dir
        assert manager.temp_dir == temp_dir
        assert config_file in manager.temp_files
        
        # 验证函数调用
        mock_create_temp.assert_called_once()
        mock_copy_mpar.assert_called_once()
        mock_create_config.assert_called_once()
        mock_process.assert_called_once()

    @patch('src.evaluators.temp_files.create_timestamped_temp_dir')
    def test_setup_temp_environment_failure(self, mock_create_temp):
        """测试设置临时环境失败"""
        # 设置mock抛出异常
        mock_create_temp.side_effect = OSError("Directory creation failed")
        
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        manager.cleanup = Mock()  # Mock cleanup方法
        params = {"distortion_distance": 20.0}
        
        with pytest.raises(OSError, match="Directory creation failed"):
            manager.setup_temp_environment(params)
        
        # 验证cleanup被调用
        manager.cleanup.assert_called_once()

    def test_get_temp_dir(self):
        """测试获取临时目录"""
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        
        # 初始状态
        assert manager.get_temp_dir() is None
        
        # 设置临时目录
        test_dir = "/tmp/test"
        manager.temp_dir = test_dir
        assert manager.get_temp_dir() == test_dir

    def test_add_temp_file(self):
        """测试添加临时文件"""
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        
        # 添加有效文件
        file_path = "/tmp/test.txt"
        manager.add_temp_file(file_path)
        assert file_path in manager.temp_files
        
        # 添加None文件（应该被忽略）
        manager.add_temp_file(None)
        assert len(manager.temp_files) == 1

    @patch('src.evaluators.temp_files.cleanup_temp_files')
    def test_cleanup(self, mock_cleanup):
        """测试清理临时文件"""
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        
        # 添加一些临时文件
        manager.temp_files = ["/tmp/file1.txt", "/tmp/file2.txt"]
        manager.temp_dir = "/tmp/test_dir"
        
        manager.cleanup()
        
        # 验证清理函数被调用
        mock_cleanup.assert_called_once_with(manager.temp_files)
        # 验证临时文件列表被清空
        assert manager.temp_files == []

    @patch('src.evaluators.temp_files.cleanup_temp_files')
    def test_cleanup_with_exception(self, mock_cleanup):
        """测试清理过程中发生异常"""
        mock_cleanup.side_effect = OSError("Cleanup failed")
        
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        manager.temp_files = ["/tmp/file1.txt"]
        
        # 异常应该被捕获，不会向上传播
        manager.cleanup()
        
        mock_cleanup.assert_called_once()

    def test_context_manager(self):
        """测试上下文管理器功能"""
        manager = TempFileManager(self.mock_config_manager, self.mock_parameter_replacer)
        manager.cleanup = Mock()
        
        with manager as mgr:
            assert mgr == manager
        
        # 退出时应该调用cleanup
        manager.cleanup.assert_called_once()


class TestCreateTempFileManager:
    """测试create_temp_file_manager函数"""

    def test_create_temp_file_manager(self):
        """测试创建临时文件管理器"""
        mock_config_manager = Mock()
        mock_parameter_replacer = Mock()
        mock_config_manager.ansa_config = Mock()
        mock_config_manager.ansa_config.mpar_file_pattern = "*.ansa_mpar"
        
        manager = create_temp_file_manager(mock_config_manager, mock_parameter_replacer)
        
        assert isinstance(manager, TempFileManager)
        assert manager.config_manager == mock_config_manager
        assert manager.parameter_replacer == mock_parameter_replacer