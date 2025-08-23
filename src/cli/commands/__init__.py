"""
CLI命令模块 - 各种命令的实现
"""

from .optimize_cmd import register_optimize_command
from .compare_cmd import register_compare_command
from .config_cmd import register_config_command
from .info_cmd import register_info_command
from .test_cmd import register_test_command
from .command_dispatcher import dispatch_command

__all__ = [
    'register_optimize_command',
    'register_compare_command', 
    'register_config_command',
    'register_info_command',
    'register_test_command',
    'dispatch_command'
]
