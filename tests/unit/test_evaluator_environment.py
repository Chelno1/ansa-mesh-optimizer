#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估器环境验证模块单元测试

作者: Chel
创建日期: 2025-08-24
"""

import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from src.evaluators.environment import (
    validate_ansa_environment,
    check_input_files,
    run_ansa_batch,
    handle_ansa_returncode,
    AnsaEnvironmentValidator
)


class TestEvaluatorEnvironment(unittest.TestCase):
    """评估器环境验证模块测试类"""
    
    def setUp(self) -> None:
        """测试前准备"""
        self.mock_config = MagicMock()
        self.mock_config.ansa_executable = '/path/to/ansa'
        self.mock_config.input_model = 'test_model.k'
        self.mock_config.script_dir = Path('/path/to/scripts')
        self.mock_config.batch_script = 'batch.py'
        self.mock_config.validation_timeout = 10
    
    def test_validate_ansa_environment_success(self):
        """测试Ansa环境验证成功"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            # 模拟成功的subprocess运行
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            is_valid, error_msg = validate_ansa_environment('/path/to/ansa', timeout=10)
            
            self.assertTrue(is_valid)
            self.assertIsNone(error_msg)
            mock_run.assert_called_once_with(
                ['/path/to/ansa', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
    
    def test_validate_ansa_environment_executable_fails(self):
        """测试Ansa可执行文件运行失败"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            # 模拟失败的subprocess运行
            mock_result = Mock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result
            
            is_valid, error_msg = validate_ansa_environment('/path/to/ansa', timeout=10)
            
            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "Ansa可执行文件无法运行")
    
    def test_validate_ansa_environment_timeout(self):
        """测试Ansa环境验证超时"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('ansa', 10)
            
            is_valid, error_msg = validate_ansa_environment('/path/to/ansa', timeout=10)
            
            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "Ansa版本检查超时（10秒）")
    
    def test_validate_ansa_environment_file_not_found(self):
        """测试Ansa可执行文件未找到"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            is_valid, error_msg = validate_ansa_environment('/path/to/ansa', timeout=10)
            
            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "Ansa可执行文件未找到: /path/to/ansa")
    
    def test_validate_ansa_environment_runtime_error(self):
        """测试Ansa环境验证运行时错误"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            mock_run.side_effect = RuntimeError("Test error")
            
            is_valid, error_msg = validate_ansa_environment('/path/to/ansa', timeout=10)
            
            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "Ansa环境验证失败: Test error")
    
    def test_validate_ansa_environment_unexpected_error(self):
        """测试Ansa环境验证意外错误"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Unexpected error")
            
            is_valid, error_msg = validate_ansa_environment('/path/to/ansa', timeout=10)
            
            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "Ansa环境验证异常: Unexpected error")
    
    def test_check_input_files_success(self):
        """测试输入文件检查成功"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            is_valid, error_msg = check_input_files('/path/to/model.k', '/path/to/batch.py')
            
            self.assertTrue(is_valid)
            self.assertEqual(error_msg, "")
    
    def test_check_input_files_model_not_found(self):
        """测试输入模型文件不存在"""
        with patch('pathlib.Path.exists') as mock_exists:
            # 第一次调用（模型文件）返回False，第二次调用不会被执行
            mock_exists.side_effect = [False]
            
            is_valid, error_msg = check_input_files('/path/to/missing_model.k', '/path/to/batch.py')
            
            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "输入模型文件不存在: /path/to/missing_model.k")
    
    def test_check_input_files_script_not_found(self):
        """测试批处理脚本不存在"""
        with patch('pathlib.Path.exists') as mock_exists:
            # 第一次调用（模型文件）返回True，第二次调用（脚本文件）返回False
            mock_exists.side_effect = [True, False]
            
            is_valid, error_msg = check_input_files('/path/to/model.k', '/path/to/missing_batch.py')
            
            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "批处理脚本不存在: /path/to/missing_batch.py")
    
    def test_run_ansa_batch_success(self):
        """测试运行Ansa批处理成功"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            # 创建模拟的subprocess结果
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Success output"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            command = ['ansa', '-b', '-i', 'model.k']
            result = run_ansa_batch(command, '/tmp/workdir', timeout=300)
            
            self.assertEqual(result, mock_result)
            mock_run.assert_called_once_with(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                cwd='/tmp/workdir'
            )
    
    def test_run_ansa_batch_timeout(self):
        """测试运行Ansa批处理超时"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('ansa', 300)
            
            command = ['ansa', '-b', '-i', 'model.k']
            
            with self.assertRaises(subprocess.TimeoutExpired):
                run_ansa_batch(command, '/tmp/workdir', timeout=300)
    
    def test_run_ansa_batch_file_not_found(self):
        """测试运行Ansa批处理时可执行文件未找到"""
        with patch('src.evaluators.environment.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            command = ['nonexistent_ansa', '-b', '-i', 'model.k']
            
            with self.assertRaises(FileNotFoundError):
                run_ansa_batch(command, '/tmp/workdir', timeout=300)
    
    def test_handle_ansa_returncode_success(self):
        """测试处理Ansa返回代码 - 成功（代码0）"""
        mock_result = Mock()
        mock_result.returncode = 0
        
        is_success, error_msg = handle_ansa_returncode(mock_result)
        
        self.assertTrue(is_success)
        self.assertIsNone(error_msg)
    
    def test_handle_ansa_returncode_warning(self):
        """测试处理Ansa返回代码 - 警告（代码1）"""
        mock_result = Mock()
        mock_result.returncode = 1
        
        is_success, error_msg = handle_ansa_returncode(mock_result)
        
        self.assertTrue(is_success)  # 代码1被视为可以继续
        self.assertIsNone(error_msg)
    
    def test_handle_ansa_returncode_fatal_error(self):
        """测试处理Ansa返回代码 - 致命错误（代码2）"""
        mock_result = Mock()
        mock_result.returncode = 2
        mock_result.stderr = "Fatal error occurred"
        
        is_success, error_msg = handle_ansa_returncode(mock_result)
        
        self.assertFalse(is_success)
        self.assertEqual(error_msg, "Ansa返回代码2 - 致命错误: Fatal error occurred")
    
    def test_handle_ansa_returncode_other_error(self):
        """测试处理Ansa返回代码 - 其他错误"""
        mock_result = Mock()
        mock_result.returncode = 3
        mock_result.stderr = "Unknown error"
        
        is_success, error_msg = handle_ansa_returncode(mock_result)
        
        self.assertFalse(is_success)
        self.assertEqual(error_msg, "Ansa执行失败，返回代码: 3, 错误输出: Unknown error")


class TestAnsaEnvironmentValidator(unittest.TestCase):
    """Ansa环境验证器测试类"""
    
    def setUp(self) -> None:
        """测试前准备"""
        self.mock_config = MagicMock()
        self.mock_config.ansa_executable = '/path/to/ansa'
        self.mock_config.input_model = 'test_model.k'
        self.mock_config.script_dir = Path('/path/to/scripts')
        self.mock_config.batch_script = 'batch.py'
        self.mock_config.validation_timeout = 10
    
    def test_validator_init(self):
        """测试验证器初始化"""
        validator = AnsaEnvironmentValidator(self.mock_config)
        
        self.assertEqual(validator.config, self.mock_config)
        self.assertFalse(validator.is_valid)
        self.assertIsNone(validator.error_message)
    
    @patch('src.evaluators.environment.validate_ansa_environment')
    @patch('src.evaluators.environment.check_input_files')
    def test_validate_success(self, mock_check_files, mock_validate_ansa):
        """测试完整验证成功"""
        # 模拟Ansa环境验证成功
        mock_validate_ansa.return_value = (True, None)
        # 模拟输入文件检查成功
        mock_check_files.return_value = (True, "")
        
        validator = AnsaEnvironmentValidator(self.mock_config)
        result = validator.validate()
        
        self.assertTrue(result)
        self.assertTrue(validator.is_valid)
        self.assertIsNone(validator.error_message)
        
        # 验证调用
        mock_validate_ansa.assert_called_once_with('/path/to/ansa', timeout=10)
        mock_check_files.assert_called_once()
    
    @patch('src.evaluators.environment.validate_ansa_environment')
    def test_validate_ansa_failure(self, mock_validate_ansa):
        """测试Ansa环境验证失败"""
        mock_validate_ansa.return_value = (False, "Ansa not found")
        
        validator = AnsaEnvironmentValidator(self.mock_config)
        result = validator.validate()
        
        self.assertFalse(result)
        self.assertFalse(validator.is_valid)
        self.assertEqual(validator.error_message, "Ansa not found")
    
    @patch('src.evaluators.environment.validate_ansa_environment')
    @patch('src.evaluators.environment.check_input_files')
    def test_validate_files_failure(self, mock_check_files, mock_validate_ansa):
        """测试文件检查失败"""
        mock_validate_ansa.return_value = (True, None)
        mock_check_files.return_value = (False, "File not found")
        
        validator = AnsaEnvironmentValidator(self.mock_config)
        result = validator.validate()
        
        self.assertFalse(result)
        self.assertFalse(validator.is_valid)
        self.assertEqual(validator.error_message, "File not found")
    
    def test_get_validation_result_success(self):
        """测试获取验证结果 - 成功"""
        validator = AnsaEnvironmentValidator(self.mock_config)
        validator.is_valid = True
        validator.error_message = None
        
        is_valid, error_msg = validator.get_validation_result()
        
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)
    
    def test_get_validation_result_failure(self):
        """测试获取验证结果 - 失败"""
        validator = AnsaEnvironmentValidator(self.mock_config)
        validator.is_valid = False
        validator.error_message = "Test error"
        
        is_valid, error_msg = validator.get_validation_result()
        
        self.assertFalse(is_valid)
        self.assertEqual(error_msg, "Test error")
    
    def test_validate_with_default_timeout(self):
        """测试使用默认超时时间的验证"""
        # 移除validation_timeout属性来测试默认值
        del self.mock_config.validation_timeout
        
        with patch('src.evaluators.environment.validate_ansa_environment') as mock_validate:
            mock_validate.return_value = (True, None)
            with patch('src.evaluators.environment.check_input_files') as mock_check:
                mock_check.return_value = (True, "")
                
                validator = AnsaEnvironmentValidator(self.mock_config)
                validator.validate()
                
                # 验证使用了默认超时时间（10秒）
                mock_validate.assert_called_once_with('/path/to/ansa', timeout=10)


if __name__ == '__main__':
    unittest.main()