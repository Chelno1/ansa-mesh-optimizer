#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa网格优化器主程序

作者: Chel
创建日期: 2025-06-19
版本: 1.1.0
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入模块
from ansa_mesh_optimizer_improved import optimize_mesh_parameters, MeshOptimizer
from compare_optimizers_improved import compare_optimizers
from config import config_manager

# 配置日志
def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Ansa网格参数优化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用贝叶斯优化，20次迭代
  python main.py optimize --optimizer bayesian --n-calls 20
  
  # 比较多个优化器
  python main.py compare --optimizers bayesian random genetic --n-calls 15
  
  # 使用配置文件
  python main.py optimize --config config.json
  
  # 使用真实Ansa评估器
  python main.py optimize --evaluator ansa --optimizer genetic
        """
    )
    
    # 添加全局参数
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='启用详细输出')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--evaluator', choices=['ansa', 'mock'], default='mock',
                       help='评估器类型 (默认: mock)')
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 优化命令
    optimize_parser = subparsers.add_parser('optimize', help='运行单个优化器')
    optimize_parser.add_argument('--optimizer', choices=['bayesian', 'random', 'forest', 'genetic', 'parallel'],
                                default='bayesian', help='优化器类型 (默认: bayesian)')
    optimize_parser.add_argument('--n-calls', type=int, default=20,
                                help='优化迭代次数 (默认: 20)')
    optimize_parser.add_argument('--n-initial-points', type=int, default=5,
                                help='初始随机点数量 (默认: 5)')
    optimize_parser.add_argument('--random-state', type=int, default=42,
                                help='随机种子 (默认: 42)')
    optimize_parser.add_argument('--no-cache', action='store_true',
                                help='禁用缓存')
    optimize_parser.add_argument('--no-early-stopping', action='store_true',
                                help='禁用早停')
    optimize_parser.add_argument('--no-sensitivity', action='store_true',
                                help='禁用敏感性分析')
    optimize_parser.add_argument('--output', type=str,
                                help='结果输出文件路径')
    
    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较多个优化器')
    compare_parser.add_argument('--optimizers', nargs='+',
                               choices=['bayesian', 'random', 'forest', 'genetic', 'parallel'],
                               default=['bayesian', 'random', 'genetic'],
                               help='要比较的优化器列表')
    compare_parser.add_argument('--n-calls', type=int, default=20,
                               help='每个优化器的迭代次数 (默认: 20)')
    compare_parser.add_argument('--n-runs', type=int, default=3,
                               help='每个优化器的运行次数 (默认: 3)')
    compare_parser.add_argument('--no-sensitivity', action='store_true',
                               help='禁用敏感性分析')
    compare_parser.add_argument('--no-report', action='store_true',
                               help='禁用详细报告生成')
    
    # 配置命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    # 生成默认配置
    config_subparsers.add_parser('generate', help='生成默认配置文件')
    
    # 验证配置
    validate_parser = config_subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('config_file', help='要验证的配置文件')
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.add_argument('--check-deps', action='store_true',
                            help='检查依赖库')
    
    return parser

def cmd_optimize(args) -> int:
    """执行优化命令"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("开始网格参数优化")
        
        # 加载配置
        if args.config:
            config_manager.load_config(args.config)
        
        # 更新配置
        config = config_manager.optimization_config
        config.n_calls = args.n_calls
        config.n_initial_points = args.n_initial_points
        config.random_state = args.random_state
        config.use_cache = not args.no_cache
        config.early_stopping = not args.no_early_stopping
        config.sensitivity_analysis = not args.no_sensitivity
        
        # 执行优化
        result = optimize_mesh_parameters(
            n_calls=args.n_calls,
            optimizer=args.optimizer,
            evaluator_type=args.evaluator,
            use_cache=not args.no_cache
        )
        
        # 输出结果
        logger.info("优化完成！")
        logger.info(f"最佳目标值: {result['best_value']:.6f}")
        logger.info(f"执行时间: {result['execution_time']:.2f}秒")
        
        print("\n最佳参数:")
        for name, value in result['best_params'].items():
            print(f"  {name}: {value}")
        
        # 保存结果（如果指定输出文件）
        if args.output:
            import json
            output_data = {
                'best_params': result['best_params'],
                'best_value': result['best_value'],
                'optimizer': result['optimizer_name'],
                'execution_time': result['execution_time']
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已保存到: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"优化失败: {e}")
        return 1

def cmd_compare(args) -> int:
    """执行比较命令"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"开始比较优化器: {args.optimizers}")
        
        # 运行比较
        results = compare_optimizers(
            optimizers=args.optimizers,
            n_calls=args.n_calls,
            n_runs=args.n_runs,
            evaluator_type=args.evaluator,
            run_sensitivity_analysis=not args.no_sensitivity,
            generate_report=not args.no_report
        )
        
        # 输出结果
        logger.info("比较完成！")
        
        if 'best_optimizer' in results:
            best_opt = results['best_optimizer']
            best_info = results['best_optimizer_info']
            
            print(f"\n推荐的最佳优化器: {best_opt}")
            print(f"平均最佳值: {best_info['mean_best_value']:.6f}")
            print(f"标准差: {best_info['std_best_value']:.6f}")
            print(f"平均执行时间: {best_info['mean_execution_time']:.2f}秒")
        
        if 'results_dir' in results:
            print(f"\n详细结果保存在: {results['results_dir']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"比较失败: {e}")
        return 1

def cmd_config(args) -> int:
    """执行配置命令"""
    logger = logging.getLogger(__name__)
    
    try:
        if args.config_action == 'generate':
            # 生成默认配置文件
            config_file = 'default_config.json'
            config_manager.save_config(config_file)
            print(f"默认配置文件已生成: {config_file}")
            
        elif args.config_action == 'validate':
            # 验证配置文件
            try:
                config_manager.load_config(args.config_file)
                print(f"配置文件 {args.config_file} 验证通过")
            except Exception as e:
                print(f"配置文件验证失败: {e}")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"配置操作失败: {e}")
        return 1

def cmd_info(args) -> int:
    """显示系统信息"""
    print("Ansa网格优化器 v1.1.0")
    print("=" * 40)
    
    # Python信息
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 依赖库检查
    if args.check_deps:
        print("\n依赖库检查:")
        print("-" * 20)
        
        dependencies = [
            ('numpy', '数值计算'),
            ('scikit-optimize', '贝叶斯优化'),
            ('matplotlib', '可视化'),
            ('pandas', '数据分析'),
            ('seaborn', '统计图表'),
            ('scipy', '科学计算'),
            ('deap', '遗传算法'),
        ]
        
        for lib_name, description in dependencies:
            try:
                __import__(lib_name)
                status = "✓ 已安装"
            except ImportError:
                status = "✗ 未安装"
            
            print(f"  {lib_name:<20} {description:<15} {status}")
    
    # 当前配置
    print(f"\n当前配置:")
    print("-" * 20)
    config = config_manager.optimization_config
    print(f"  优化器: {config.optimizer}")
    print(f"  迭代次数: {config.n_calls}")
    print(f"  随机种子: {config.random_state}")
    print(f"  使用缓存: {config.use_cache}")
    print(f"  早停机制: {config.early_stopping}")
    
    return 0

def main() -> int:
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose, args.log_file)
    
    # 检查命令
    if not args.command:
        parser.print_help()
        return 1
    
    # 执行对应命令
    if args.command == 'optimize':
        return cmd_optimize(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'config':
        return cmd_config(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        print(f"未知命令: {args.command}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n用户中断程序")
        sys.exit(130)
    except Exception as e:
        print(f"程序异常: {e}")
        logging.getLogger(__name__).exception("程序异常")
        sys.exit(1)