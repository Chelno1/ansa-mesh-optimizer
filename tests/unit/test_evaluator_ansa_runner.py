#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试evaluator.ansa_runner模块

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.evaluators.ansa_runner import AnsaRunner, create_ansa_runner


class TestAnsaRunner:
    """测试AnsaRunner类"""

    def setup_method(self):
        """设置测试环境"""
        # 创建mock配置管理器
        self.mock_config_manager = Mock()
        self.mock_ansa_config = Mock()
        
        self.mock_config_manager.ansa_config = self.mock_ansa_config
        self.mock_ansa_config.ansa_executable = "/path/to/ansa"
        self.mock_ansa_config.script_dir = Path("/scripts")
        self.mock_ansa_config.batch_script = "batch.py"
        self.mock_ansa_config.input_model = "model.ansa"
        self.mock_ansa_config.execution_timeout = 300

    def test_init_success(self):
        """测试正常初始化"""
        runner = AnsaRunner(self.mock_config_manager)
        
        assert runner.config_manager == self.mock_config_manager
        assert runner.config == self.mock_ansa_config
        assert isinstance(runner.cwd_dir, Path)

    def test_init_without_config_manager(self):
        """测试缺少配置管理器时的初始化"""
        with pytest.raises(ValueError, match="AnsaRunner requires a config_manager instance"):
            AnsaRunner(None)

    @patch('src.evaluators.ansa_runner.check_input_files')
    @patch('src.evaluators.ansa_runner.run_ansa_batch')
    @patch('src.evaluators.ansa_runner.handle_ansa_returncode')
    @patch('src.evaluators.ansa_runner.parse_ansa_output')
    def test_run_ansa_batch_success(self, mock_parse, mock_handle, mock_run, mock_check):
        """测试成功运行ANSA批处理"""
        # 设置mock返回值
        mock_check.return_value = (True, "")
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Bad elements: 42"
        mock_run.return_value = mock_result
        mock_handle.return_value = (True, "")
        mock_parse.return_value = 42.0
        
        runner = AnsaRunner(self.mock_config_manager)
        temp_dir = "/tmp/test"
        
        result = runner.run_ansa_batch(temp_dir)
        
        assert result == 42.0
        mock_check.assert_called_once()
        mock_run.assert_called_once()
        mock_handle.assert_called_once_with(mock_result)
        mock_parse.assert_called_once_with("Bad elements: 42")

    @patch('src.evaluators.ansa_runner.check_input_files')
    @patch('src.evaluators.ansa_runner.simulate_evaluation')
    def test_run_ansa_batch_input_validation_failure(self, mock_simulate, mock_check):
        """测试输入文件验证失败"""
        # 设置mock返回值
        mock_check.return_value = (False, "Input file not found")
        mock_simulate.return_value = 999.0
        
        runner = AnsaRunner(self.mock_config_manager)
        temp_dir = "/tmp/test"
        
        result = runner.run_ansa_batch(temp_dir)
        
        assert result == 999.0
        mock_check.assert_called_once()
        mock_simulate.assert_called_once()

    @patch('src.evaluators.ansa_runner.check_input_files')
    @patch('src.evaluators.ansa_runner.run_ansa_batch')
    @patch('src.evaluators.ansa_runner.handle_ansa_returncode')
    @patch('src.evaluators.ansa_runner.simulate_evaluation')
    def test_run_ansa_batch_execution_failure(self, mock_simulate, mock_handle, mock_run, mock_check):
        """测试ANSA执行失败"""
        # 设置mock返回值
        mock_check.return_value = (True, "")
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        mock_handle.return_value = (False, "Execution failed")
        mock_simulate.return_value = 888.0
        
        runner = AnsaRunner(self.mock_config_manager)
        temp_dir = "/tmp/test"
        
        result = runner.run_ansa_batch(temp_dir)
        
        assert result == 888.0
        mock_simulate.assert_called_once()

    @patch('src.evaluators.ansa_runner.check_input_files')
    @patch('src.evaluators.ansa_runner.simulate_evaluation')
    def test_run_ansa_batch_exception(self, mock_simulate, mock_check):
        """测试运行过程中发生异常"""
        # 设置mock抛出异常
        mock_check.side_effect = OSError("System error")
        mock_simulate.return_value = 777.0
        
        runner = AnsaRunner(self.mock_config_manager)
        temp_dir = "/tmp/test"
        
        result = runner.run_ansa_batch(temp_dir)
        
        assert result == 777.0
        mock_simulate.assert_called_once()

    def test_build_ansa_command(self):
        """测试构建ANSA命令"""
        runner = AnsaRunner(self.mock_config_manager)
        temp_dir = "/tmp/test"
        
        command = runner._build_ansa_command(temp_dir)
        
        expected_command = [
            "/path/to/ansa",
            "-b",
            "-execpy",
            f"load_script: '{Path('/scripts') / 'batch.py'}'",
            "-i",
            f"{runner.cwd_dir / 'model.ansa'}",
            "-changedir",
            temp_dir,
        ]
        
        assert command == expected_command

    @patch('src.evaluators.ansa_runner.check_input_files')
    def test_validate_input_files_success(self, mock_check):
        """测试输入文件验证成功"""
        mock_check.return_value = (True, "")
        
        runner = AnsaRunner(self.mock_config_manager)
        result = runner._validate_input_files()
        
        assert result is True
        mock_check.assert_called_once()

    @patch('src.evaluators.ansa_runner.check_input_files')
    def test_validate_input_files_failure(self, mock_check):
        """测试输入文件验证失败"""
        mock_check.return_value = (False, "File not found")
        
        runner = AnsaRunner(self.mock_config_manager)
        result = runner._validate_input_files()
        
        assert result is False
        mock_check.assert_called_once()

    def test_get_input_model_path(self):
        """测试获取输入模型路径"""
        runner = AnsaRunner(self.mock_config_manager)
        path = runner.get_input_model_path()
        
        expected_path = str(runner.cwd_dir / "model.ansa")
        assert path == expected_path

    def test_get_batch_script_path(self):
        """测试获取批处理脚本路径"""
        runner = AnsaRunner(self.mock_config_manager)
        path = runner.get_batch_script_path()
        
        expected_path = str(Path("/scripts") / "batch.py")
        assert path == expected_path

    def test_get_ansa_executable(self):
        """测试获取ANSA可执行文件路径"""
        runner = AnsaRunner(self.mock_config_manager)
        executable = runner.get_ansa_executable()
        
        assert executable == "/path/to/ansa"

    def test_get_execution_timeout(self):
        """测试获取执行超时时间"""
        runner = AnsaRunner(self.mock_config_manager)
        timeout = runner.get_execution_timeout()
        
        assert timeout == 300


class TestCreateAnsaRunner:
    """测试create_ansa_runner函数"""

    def test_create_ansa_runner(self):
        """测试创建ANSA运行器"""
        mock_config_manager = Mock()
        mock_config_manager.ansa_config = Mock()
        mock_config_manager.ansa_config.ansa_executable = "/path/to/ansa"
        mock_config_manager.ansa_config.script_dir = Path("/scripts")
        mock_config_manager.ansa_config.batch_script = "batch.py"
        mock_config_manager.ansa_config.input_model = "model.ansa"
        mock_config_manager.ansa_config.execution_timeout = 300
        
        runner = create_ansa_runner(mock_config_manager)
        
        assert isinstance(runner, AnsaRunner)
        assert runner.config_manager == mock_config_manager