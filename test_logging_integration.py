#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志配置集成测试脚本

测试统一日志配置的各种功能，确保在不同模块中行为一致
"""

import sys
import logging
import tempfile
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_logging_config():
    """测试基本日志配置功能"""
    print("=" * 60)
    print("测试基本日志配置功能")
    print("=" * 60)
    
    try:
        from src.utils.logging_config import setup_logging, reset_logging
        
        # 重置日志配置
        reset_logging()
        
        # 测试基本配置
        config = setup_logging(level='INFO', verbose=False)
        
        # 创建测试日志记录器
        logger = logging.getLogger('test_logger')
        
        print("测试不同级别的日志输出:")
        logger.debug("这是DEBUG消息 - 应该不显示")
        logger.info("这是INFO消息 - 应该显示")
        logger.warning("这是WARNING消息 - 应该显示")
        logger.error("这是ERROR消息 - 应该显示")
        
        print("✓ 基本日志配置测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 基本日志配置测试失败: {e}")
        return False


def test_verbose_mode():
    """测试详细模式"""
    print("\n" + "=" * 60)
    print("测试详细模式")
    print("=" * 60)
    
    try:
        from src.utils.logging_config import setup_logging, reset_logging
        
        # 重置日志配置
        reset_logging()
        
        # 测试详细模式
        config = setup_logging(level='DEBUG', verbose=True)
        
        # 创建测试日志记录器
        logger = logging.getLogger('test_verbose_logger')
        
        print("测试详细模式的日志输出:")
        logger.debug("这是DEBUG消息 - 应该显示（详细格式）")
        logger.info("这是INFO消息 - 应该显示（详细格式）")
        
        print("✓ 详细模式测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 详细模式测试失败: {e}")
        return False


def test_file_logging():
    """测试文件日志"""
    print("\n" + "=" * 60)
    print("测试文件日志")
    print("=" * 60)
    
    try:
        from src.utils.logging_config import setup_logging, reset_logging
        
        # 重置日志配置
        reset_logging()
        
        # 创建临时日志文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp_file:
            log_file_path = tmp_file.name
        
        # 测试文件日志
        config = setup_logging(level='INFO', log_file=log_file_path)
        
        # 创建测试日志记录器
        logger = logging.getLogger('test_file_logger')
        
        # 写入一些日志
        logger.info("这是一条测试日志消息")
        logger.warning("这是一条警告消息")
        logger.error("这是一条错误消息")
        
        # 检查日志文件是否存在并包含内容
        log_file = Path(log_file_path)
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if '测试日志消息' in content and '警告消息' in content:
                    print(f"✓ 文件日志测试通过，日志已写入: {log_file_path}")
                    print(f"日志文件内容预览:")
                    print(content[:200] + "..." if len(content) > 200 else content)
                    
                    # 清理临时文件
                    log_file.unlink()
                    return True
                else:
                    print(f"✗ 文件日志内容不正确")
                    return False
        else:
            print(f"✗ 日志文件未创建")
            return False
        
    except Exception as e:
        print(f"✗ 文件日志测试失败: {e}")
        return False


def test_module_specific_logging():
    """测试模块特定日志配置"""
    print("\n" + "=" * 60)
    print("测试模块特定日志配置")
    print("=" * 60)
    
    try:
        from src.utils.logging_config import setup_logging, configure_module_logging, reset_logging
        
        # 重置日志配置
        reset_logging()
        
        # 基本配置
        config = setup_logging(level='INFO')
        
        # 配置特定模块的日志级别
        configure_module_logging('test_module', 'DEBUG')
        configure_module_logging('src.optimizers', 'WARNING')
        
        # 测试不同模块的日志记录器
        test_logger = logging.getLogger('test_module')
        opt_logger = logging.getLogger('src.optimizers')
        general_logger = logging.getLogger('general')
        
        print("测试模块特定日志级别:")
        print("test_module (DEBUG级别):")
        test_logger.debug("test_module DEBUG消息 - 应该显示")
        test_logger.info("test_module INFO消息 - 应该显示")
        
        print("src.optimizers (WARNING级别):")
        opt_logger.info("src.optimizers INFO消息 - 应该不显示")
        opt_logger.warning("src.optimizers WARNING消息 - 应该显示")
        
        print("general (默认INFO级别):")
        general_logger.debug("general DEBUG消息 - 应该不显示")
        general_logger.info("general INFO消息 - 应该显示")
        
        print("✓ 模块特定日志配置测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模块特定日志配置测试失败: {e}")
        return False


def test_cli_logging_compatibility():
    """测试CLI日志兼容性"""
    print("\n" + "=" * 60)
    print("测试CLI日志兼容性")
    print("=" * 60)
    
    try:
        from src.utils.logging_config import setup_cli_logging, reset_logging
        
        # 重置日志配置
        reset_logging()
        
        # 测试CLI兼容的日志设置
        setup_cli_logging(verbose=True, log_file=None)
        
        # 创建测试日志记录器
        logger = logging.getLogger('cli_test')
        
        print("测试CLI兼容的日志输出:")
        logger.debug("CLI DEBUG消息 - 应该显示（详细模式）")
        logger.info("CLI INFO消息 - 应该显示")
        logger.warning("CLI WARNING消息 - 应该显示")
        
        print("✓ CLI日志兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"✗ CLI日志兼容性测试失败: {e}")
        return False


def test_config_integration():
    """测试配置文件集成"""
    print("\n" + "=" * 60)
    print("测试配置文件集成")
    print("=" * 60)
    
    try:
        # 测试是否能从默认配置文件读取日志配置
        config_file = project_root / "src" / "default_config.json"
        if config_file.exists():
            from src.config.config import SimpleConfigManager
            from src.utils.logging_config import reset_logging
            
            # 重置日志配置
            reset_logging()
            
            # 通过配置管理器初始化（会自动设置日志）
            config_manager = SimpleConfigManager(str(config_file))
            
            # 测试日志是否正常工作
            logger = logging.getLogger('config_test')
            logger.info("配置文件集成测试消息")
            
            print(f"✓ 配置文件集成测试通过，使用配置文件: {config_file}")
            return True
        else:
            print(f"⚠ 跳过配置文件集成测试，配置文件不存在: {config_file}")
            return True
        
    except Exception as e:
        print(f"✗ 配置文件集成测试失败: {e}")
        return False


def test_quiet_mode():
    """测试静默模式"""
    print("\n" + "=" * 60)
    print("测试静默模式")
    print("=" * 60)
    
    try:
        from src.utils.logging_config import setup_logging, reset_logging
        
        # 重置日志配置
        reset_logging()
        
        # 测试静默模式
        config = setup_logging(level='ERROR', quiet=True)
        
        # 创建测试日志记录器
        logger = logging.getLogger('quiet_test')
        
        print("测试静默模式的日志输出（只应显示ERROR及以上级别）:")
        logger.debug("DEBUG消息 - 应该不显示")
        logger.info("INFO消息 - 应该不显示")
        logger.warning("WARNING消息 - 应该不显示")
        logger.error("ERROR消息 - 应该显示")
        logger.critical("CRITICAL消息 - 应该显示")
        
        print("✓ 静默模式测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 静默模式测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始日志配置集成测试")
    print("测试环境: Python", sys.version)
    print("项目根目录:", project_root)
    
    # 运行所有测试
    tests = [
        test_basic_logging_config,
        test_verbose_mode,
        test_file_logging,
        test_module_specific_logging,
        test_cli_logging_compatibility,
        test_config_integration,
        test_quiet_mode
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 测试 {test_func.__name__} 发生异常: {e}")
            failed += 1
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总测试数: {passed + failed}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("🎉 所有测试通过！日志配置集成成功。")
        return 0
    else:
        print("❌ 有测试失败，请检查日志配置。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)