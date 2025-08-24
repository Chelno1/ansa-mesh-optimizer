#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI集成测试

作者: Chel
创建日期: 2025-07-07
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

from src.cli.cli_main import main_cli, create_parser
from src.core.ansa_mesh_optimizer import MeshOptimizer
from src.batch_mesh import AnsaBatchMeshRunner

class TestCLIIntegration(unittest.TestCase):
    """CLI集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 禁用日志输出
        logging.disable(logging.CRITICAL)
        
    def tearDown(self):
        """测试后清理"""
        logging.disable(logging.NOTSET)
    
    def test_parser_creation(self):
        """测试参数解析器创建"""
        parser = create_parser()
        self.assertIsNotNone(parser)
        
        # 测试版本参数会导致SystemExit，这是正常行为
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(['--version'])
        self.assertEqual(cm.exception.code, 0)  # 版本显示应该是正常退出
    
    @patch('src.cli.commands.command_dispatcher.dispatch_command')
    def test_optimize_command(self, mock_dispatch):
        """测试优化命令"""
        mock_dispatch.return_value = 0
        
        test_args = [
            'optimize',
            '--optimizer', 'genetic',
            '--n-calls', '5',
            '--evaluator', 'mock'
        ]
        
        with patch('sys.argv', ['cli_main.py'] + test_args):
            exit_code = main_cli()
            self.assertEqual(exit_code, 0)
            mock_dispatch.assert_called_once()
    
    @patch('src.cli.commands.command_dispatcher.dispatch_command')
    def test_info_command(self, mock_dispatch):
        """测试信息命令"""
        mock_dispatch.return_value = 0
        
        test_args = ['info', '--check-deps']
        
        with patch('sys.argv', ['cli_main.py'] + test_args):
            exit_code = main_cli()
            self.assertEqual(exit_code, 0)
            mock_dispatch.assert_called_once()
    
    @patch('src.batch_mesh.AnsaBatchMeshRunner.run_batch_mesh')
    def test_batch_mesh_execution(self, mock_run_batch):
        """测试批处理网格执行"""
        mock_run_batch.return_value = True
        
        runner = AnsaBatchMeshRunner()
        params = {
            'quality_threshold': 0.6,
            'distortion_distance': 20
        }
        
        success = runner.run_batch_mesh(params)
        self.assertTrue(success)
        mock_run_batch.assert_called_once()
    
    @patch('src.core.ansa_mesh_optimizer.MeshOptimizer.optimize')
    def test_mesh_optimization(self, mock_optimize):
        """测试网格优化"""
        from src.optimizers.optimizer_config import OptimizationResult
        
        expected_result = OptimizationResult(
            best_value=0.5,
            best_params={'param1': 1.0},
            optimizer_name='genetic',
            optimization_history=[],
            success=True
        )
        mock_optimize.return_value = expected_result
        
        # 创建带配置管理器的优化器
        from src.config.config import UnifiedConfigManager
        from src.core.ansa_mesh_optimizer import ConfigManagerWrapper
        
        unified_manager = UnifiedConfigManager()
        config_manager = ConfigManagerWrapper(unified_manager)
        
        optimizer = MeshOptimizer(config_manager=config_manager, evaluator_type='mock')
        result = optimizer.optimize(optimizer='genetic', n_calls=5)
        
        self.assertEqual(result, expected_result)
        mock_optimize.assert_called_once()
    
    def test_invalid_command(self):
        """测试无效命令"""
        test_args = ['invalid_command']
        
        with patch('sys.argv', ['cli_main.py'] + test_args):
            with self.assertRaises(SystemExit) as cm:
                main_cli()
            self.assertEqual(cm.exception.code, 2)  # argparse错误退出码
    
    @patch('src.cli.commands.command_dispatcher.dispatch_command')
    def test_error_handling(self, mock_dispatch):
        """测试错误处理"""
        mock_dispatch.side_effect = Exception("Test error")
        
        test_args = [
            'optimize',
            '--optimizer', 'genetic',
            '--n-calls', '5'
        ]
        
        with patch('sys.argv', ['cli_main.py'] + test_args):
            exit_code = main_cli()
            self.assertEqual(exit_code, 1)

if __name__ == '__main__':
    unittest.main()