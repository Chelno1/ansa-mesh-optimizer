#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估器工具函数单元测试

作者: Chel
创建日期: 2025-08-24
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.evaluators.utils import normalize_params
from src.evaluators.io_utils import (
    create_timestamped_temp_dir,
    copy_mpar_files_to_temp_dir,
    create_temp_config_in_dir,
    create_temp_config,
    parse_ansa_output,
    cleanup_temp_files,
    cleanup_temp_directory,
    process_parameter_files_in_temp_dir,
    simulate_evaluation
)


class TestEvaluatorUtils(unittest.TestCase):
    """评估器工具函数测试类"""
    
    def setUp(self) -> None:
        """测试前准备"""
        self.test_params = {
            'distortion_distance': 20.0,
            'rule_fillet_width_1': 3.0,
            'perimeter_distance': 0.8
        }
        
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
    
    def test_normalize_params(self):
        """测试参数标准化"""
        # 测试正常的float参数
        params = {'param1': 1.5, 'param2': 2.0}
        normalized = normalize_params(params)
        self.assertEqual(normalized, {'param1': 1.5, 'param2': 2.0})
        
        # 测试字符串数字
        params = {'param1': '1.5', 'param2': '2'}
        normalized = normalize_params(params)
        self.assertEqual(normalized, {'param1': 1.5, 'param2': 2.0})
        
        # 测试整数
        params = {'param1': 1, 'param2': 2}
        normalized = normalize_params(params)
        self.assertEqual(normalized, {'param1': 1.0, 'param2': 2.0})
    
    @patch('src.evaluators.utils.numpy', create=True)
    def test_normalize_params_with_numpy(self, mock_numpy):
        """测试numpy类型参数标准化"""
        # 模拟numpy scalar
        mock_value = MagicMock()
        mock_value.item.return_value = 1.5
        params = {'param1': mock_value}
        
        normalized = normalize_params(params)
        self.assertEqual(normalized['param1'], 1.5)
        mock_value.item.assert_called_once()
    
    def test_create_timestamped_temp_dir(self):
        """测试创建时间戳临时目录"""
        with patch('src.evaluators.io_utils.os.getcwd', return_value='/test'):
            with patch('src.evaluators.io_utils.os.makedirs') as mock_makedirs:
                temp_dir = create_timestamped_temp_dir()
                
                # 验证目录名格式
                self.assertTrue(temp_dir.startswith('/test/ansa_mesh_eval_'))
                mock_makedirs.assert_called_once()
    
    def test_create_timestamped_temp_dir_failure(self):
        """测试临时目录创建失败的情况"""
        with patch('src.evaluators.io_utils.os.getcwd', return_value='/test'):
            with patch('src.evaluators.io_utils.os.makedirs', side_effect=OSError("Permission denied")):
                temp_dir = create_timestamped_temp_dir()
                # 应该回退到当前目录
                self.assertEqual(temp_dir, '/test')
    
    def test_copy_mpar_files_to_temp_dir(self):
        """测试复制mpar文件到临时目录"""
        # 创建源目录和文件
        source_dir = Path(self.temp_dir) / 'source'
        source_dir.mkdir()
        
        # 创建测试mpar文件
        test_file = source_dir / 'test.ansa_mpar'
        test_file.write_text('test content')
        
        # 创建目标目录
        target_dir = Path(self.temp_dir) / 'target'
        target_dir.mkdir()
        
        # 执行复制
        result = copy_mpar_files_to_temp_dir(str(target_dir), source_dir, '*.ansa_mpar')
        
        # 验证结果
        expected_path = target_dir / 'test.ansa_mpar'
        self.assertEqual(result, str(expected_path))
        self.assertTrue(expected_path.exists())
        self.assertEqual(expected_path.read_text(), 'test content')
    
    def test_copy_mpar_files_no_files(self):
        """测试当没有mpar文件时的情况"""
        source_dir = Path(self.temp_dir) / 'empty_source'
        source_dir.mkdir()
        
        target_dir = Path(self.temp_dir) / 'target'
        target_dir.mkdir()
        
        result = copy_mpar_files_to_temp_dir(str(target_dir), source_dir, '*.ansa_mpar')
        self.assertEqual(result, "")
    
    def test_create_temp_config_in_dir(self):
        """测试在目录中创建临时配置文件"""
        def mock_format_func(key, value):
            return f"{value}.formatted"
        
        params = {'param1': 1.0, 'param2': 2.0}
        result = create_temp_config_in_dir(self.temp_dir, params, mock_format_func)
        
        # 验证文件创建
        expected_path = os.path.join(self.temp_dir, "mesh_config.json")
        self.assertEqual(result, expected_path)
        self.assertTrue(os.path.exists(result))
        
        # 验证文件内容
        import json
        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        expected_data = {
            'param1': '1.0.formatted',
            'param2': '2.0.formatted'
        }
        self.assertEqual(data, expected_data)
    
    def test_create_temp_config(self):
        """测试创建临时配置文件"""
        def mock_format_func(key, value):
            return f"{value}"
        
        params = {'param1': 1.0, 'param2': 2.0}
        
        with patch('src.evaluators.io_utils.tempfile.NamedTemporaryFile') as mock_temp:
            mock_file = MagicMock()
            mock_file.name = '/tmp/test_config.txt'
            mock_temp.return_value.__enter__.return_value = mock_file
            
            result = create_temp_config(params, mock_format_func)
            
            self.assertEqual(result, '/tmp/test_config.txt')
            # 验证写入的内容
            from unittest.mock import call
            expected_calls = [
                call('param1 = 1.0\n'),
                call('param2 = 2.0\n')
            ]
            mock_file.write.assert_has_calls(expected_calls, any_order=True)
    
    def test_parse_ansa_output_with_patterns(self):
        """测试解析Ansa输出 - 匹配模式"""
        # 测试各种输出模式
        test_cases = [
            ("bad elements: 123", 123.0),
            ("failed elements: 456", 456.0),
            ("poor quality elements: 789", 789.0),
            ("质量不合格元素: 100", 100.0),
            ("不合格单元: 200", 200.0)
        ]
        
        for output, expected in test_cases:
            with self.subTest(output=output):
                result = parse_ansa_output(output)
                self.assertEqual(result, expected)
    
    def test_parse_ansa_output_from_lines(self):
        """测试从输出行解析数字"""
        output = "Some output\nProcess completed\nElements processed: 1000 500 200\nFinal result"
        result = parse_ansa_output(output)
        self.assertEqual(result, 1000.0)  # 应该取最大的数字
    
    def test_parse_ansa_output_no_match(self):
        """测试无法解析输出的情况"""
        output = "No relevant information here"
        result = parse_ansa_output(output)
        self.assertEqual(result, 99999.0)
    
    def test_cleanup_temp_files(self):
        """测试清理临时文件"""
        # 创建测试文件
        test_files = []
        for i in range(3):
            temp_file = os.path.join(self.temp_dir, f'test_{i}.tmp')
            with open(temp_file, 'w') as f:
                f.write('test')
            test_files.append(temp_file)
        
        # 添加一个None值
        test_files.append(None)
        
        # 执行清理
        cleanup_temp_files(test_files)
        
        # 验证文件已删除
        for temp_file in test_files:
            if temp_file is not None:
                self.assertFalse(os.path.exists(temp_file))
    
    def test_cleanup_temp_directory(self):
        """测试清理临时目录"""
        # 创建测试目录和文件
        test_dir = os.path.join(self.temp_dir, 'test_cleanup')
        os.makedirs(test_dir)
        
        test_file = os.path.join(test_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        # 验证目录存在
        self.assertTrue(os.path.exists(test_dir))
        
        # 执行清理
        cleanup_temp_directory(test_dir)
        
        # 验证目录已删除
        self.assertFalse(os.path.exists(test_dir))
    
    def test_process_parameter_files_in_temp_dir(self):
        """测试在临时目录中处理参数文件"""
        # 创建临时mpar文件
        mpar_file = Path(self.temp_dir) / 'test.ansa_mpar'
        mpar_file.write_text('original content')
        
        # 创建模拟的参数替换器
        mock_replacer = MagicMock()
        mock_replacer.process_parameter_replacements.return_value = str(mpar_file)
        
        params = {'param1': 1.0}
        
        # 执行处理
        process_parameter_files_in_temp_dir(self.temp_dir, params, mock_replacer)
        
        # 验证调用
        mock_replacer.process_parameter_replacements.assert_called_once_with(
            str(mpar_file), params
        )
    
    def test_process_parameter_files_with_new_file(self):
        """测试处理参数文件时创建新文件的情况"""
        # 创建临时mpar文件
        mpar_file = Path(self.temp_dir) / 'test.ansa_mpar'
        mpar_file.write_text('original content')
        
        # 创建模拟的参数替换器，返回相同文件路径（模拟就地更新）
        mock_replacer = MagicMock()
        mock_replacer.process_parameter_replacements.return_value = str(mpar_file)
        
        params = {'param1': 1.0}
        
        # 执行处理
        process_parameter_files_in_temp_dir(self.temp_dir, params, mock_replacer)
        
        # 验证调用了参数替换器
        mock_replacer.process_parameter_replacements.assert_called_once_with(
            str(mpar_file), params
        )
        
        # 验证文件内容保持不变（因为返回的是相同路径）
        self.assertEqual(mpar_file.read_text(), 'original content')
    
    def test_process_parameter_files_no_mpar(self):
        """测试当没有mpar文件时的处理"""
        mock_replacer = MagicMock()
        params = {'param1': 1.0}
        
        # 执行处理（应该不会报错）
        process_parameter_files_in_temp_dir(self.temp_dir, params, mock_replacer)
        
        # 验证没有调用参数替换器
        mock_replacer.process_parameter_replacements.assert_not_called()
    
    @patch('random.uniform')
    def test_simulate_evaluation(self, mock_uniform):
        """测试模拟评估"""
        mock_uniform.return_value = 123.45
        
        result = simulate_evaluation()
        self.assertEqual(result, 123.45)
        mock_uniform.assert_called_once_with(50, 500)
        
        # 重置mock并测试自定义范围
        mock_uniform.reset_mock()
        mock_uniform.return_value = 10.0
        
        result = simulate_evaluation(base_range=(5, 15))
        self.assertEqual(result, 10.0)
        mock_uniform.assert_called_once_with(5, 15)


if __name__ == '__main__':
    unittest.main()