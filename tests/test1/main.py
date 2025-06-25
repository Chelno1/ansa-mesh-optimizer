#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa网格优化器主程序 - 改进版本

作者: Chel
创建日期: 2025-06-19
版本: 1.2.0
更新日期: 2025-06-20
修复: 依赖检查，错误处理，用户体验
"""

import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 全局变量
APP_VERSION = "1.2.0"
APP_NAME = "Ansa Mesh Optimizer"

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # 创建格式化器
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
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
    console_handler.setFormatter(simple_formatter if not verbose else detailed_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"警告: 无法创建日志文件 {log_file}: {e}")

def check_and_import_modules():
    """检查并导入必要模块"""
    missing_modules = []
    available_modules = []
    
    # 检查必需的本地模块
    required_local_modules = [
        'config',
        'mesh_evaluator', 
        'optimization_cache',
        'early_stopping',
        'genetic_optimizer_improved',
        'utils'
    ]
    
    print("检查本地模块...")
    for module_name in required_local_modules:
        try:
            # 构造相对于main.py的导入路径
            # 例如 'config' -> 'config.config'
            # 例如 'mesh_evaluator' -> 'evaluators.mesh_evaluator'
            if module_name == 'config':
                import_name = 'config.config'
            elif module_name == 'mesh_evaluator':
                import_name = 'evaluators.mesh_evaluator'
            elif module_name == 'optimization_cache':
                import_name = 'utils.optimization_cache'
            elif module_name == 'utils':
                import_name = 'utils.utils'
            else:
                import_name = f'core.{module_name}'
            __import__(import_name)
            available_modules.append(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            missing_modules.append((module_name, str(e)))
            print(f"  ✗ {module_name}: {e}")
    
    # 检查可选的第三方模块
    optional_modules = [
        ('numpy', 'numpy'),
        ('scikit-optimize', 'skopt'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('pandas', 'pandas'),
        ('seaborn', 'seaborn'),
        ('scipy', 'scipy')
    ]
    
    print("\n检查第三方模块...")
    for display_name, import_name in optional_modules:
        try:
            __import__(import_name)
            available_modules.append(display_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ○ {display_name} (可选)")
    
    if missing_modules:
        print(f"\n❌ 缺少必需模块:")
        for module_name, error in missing_modules:
            print(f"  - {module_name}: {error}")
        return False, missing_modules, available_modules
    
    print(f"\n✓ 所有必需模块已加载")
    return True, [], available_modules

def import_core_modules():
    """导入核心模块"""
    try:
        from core.ansa_mesh_optimizer_improved import optimize_mesh_parameters, MeshOptimizer, check_dependencies
        from core.compare_optimizers_improved import compare_optimizers
        from config.config import config_manager
        return True, (optimize_mesh_parameters, MeshOptimizer, compare_optimizers, config_manager, check_dependencies)
    except ImportError as e:
        print(f"❌ 核心模块导入失败: {e}")
        return False, None

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} v{APP_VERSION} - 高级网格参数优化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用贝叶斯优化，20次迭代
  python main.py optimize --optimizer bayesian --n-calls 20 --evaluator mock

  # 比较多个优化器
  python main.py compare --optimizers bayesian random genetic --n-calls 15 --evaluator mock

  # 生成配置文件
  python main.py config generate

  # 检查系统信息
  python main.py info --check-deps

  # 使用真实Ansa评估器
  python main.py optimize --optimizer genetic --evaluator ansa --config my_config.json
        """
    )
    
    # 全局参数
    parser.add_argument('--version', action='version', version=f'{APP_NAME} {APP_VERSION}')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='启用详细输出')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式（仅显示错误）')
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 优化命令
    optimize_parser = subparsers.add_parser('optimize', help='运行单个优化器')
    optimize_parser.add_argument('--optimizer', 
                                choices=['bayesian', 'random', 'forest', 'genetic', 'parallel'],
                                default='bayesian', 
                                help='优化器类型 (默认: bayesian)')
    optimize_parser.add_argument('--evaluator', 
                                choices=['ansa', 'mock', 'mock_ackley', 'mock_rastrigin'], 
                                default='mock',
                                help='评估器类型 (默认: mock)')
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
    optimize_parser.add_argument('--save-plots', action='store_true',
                                help='保存优化图表')
    
    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较多个优化器')
    compare_parser.add_argument('--optimizers', nargs='+',
                               choices=['bayesian', 'random', 'forest', 'genetic', 'parallel'],
                               default=['bayesian', 'random', 'genetic'],
                               help='要比较的优化器列表')
    compare_parser.add_argument('--evaluator', 
                               choices=['ansa', 'mock', 'mock_ackley', 'mock_rastrigin'], 
                               default='mock',
                               help='评估器类型 (默认: mock)')
    compare_parser.add_argument('--n-calls', type=int, default=20,
                               help='每个优化器的迭代次数 (默认: 20)')
    compare_parser.add_argument('--n-runs', type=int, default=3,
                               help='每个优化器的运行次数 (默认: 3)')
    compare_parser.add_argument('--no-sensitivity', action='store_true',
                               help='禁用敏感性分析')
    compare_parser.add_argument('--no-report', action='store_true',
                               help='禁用详细报告生成')
    compare_parser.add_argument('--parallel-runs', action='store_true',
                               help='并行运行比较（实验性）')
    
    # 配置命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    # 生成默认配置
    generate_parser = config_subparsers.add_parser('generate', help='生成默认配置文件')
    generate_parser.add_argument('--output', type=str, default='default_config.json',
                                help='配置文件输出路径')
    generate_parser.add_argument('--example', action='store_true',
                                help='生成示例配置而非默认配置')
    
    # 验证配置
    validate_parser = config_subparsers.add_parser('validate', help='验证配置文件')
    validate_parser.add_argument('config_file', help='要验证的配置文件')
    
    # 显示配置
    show_parser = config_subparsers.add_parser('show', help='显示当前配置')
    show_parser.add_argument('--section', choices=['optimization', 'ansa', 'parameter_space'],
                            help='显示特定配置节')
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.add_argument('--check-deps', action='store_true',
                            help='检查依赖库')
    info_parser.add_argument('--check-ansa', action='store_true',
                            help='检查Ansa环境')
    info_parser.add_argument('--performance', action='store_true',
                            help='运行性能测试')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    test_parser.add_argument('--quick', action='store_true',
                            help='快速测试')
    test_parser.add_argument('--evaluator', choices=['mock', 'ansa'], default='mock',
                            help='测试使用的评估器')
    test_parser.add_argument('--verbose-test', action='store_true',
                            help='详细测试输出')
    
    return parser

def cmd_optimize(args, modules) -> int:
    """执行优化命令"""
    logger = logging.getLogger(__name__)
    optimize_mesh_parameters, MeshOptimizer, compare_optimizers, config_manager, check_dependencies = modules
    
    try:
        print(f"🚀 开始网格参数优化")
        print(f"   优化器: {args.optimizer}")
        print(f"   评估器: {args.evaluator}")
        print(f"   迭代次数: {args.n_calls}")
        
        # 检查优化器可用性
        deps = check_dependencies()
        if args.optimizer in ['bayesian', 'random', 'forest'] and not deps['skopt_available']:
            print(f"❌ 优化器 {args.optimizer} 需要 scikit-optimize 库")
            print("请运行: pip install scikit-optimize")
            return 1
        
        # 加载配置
        if args.config:
            try:
                config_manager.load_config(args.config)
                print(f"✓ 配置已从 {args.config} 加载")
            except Exception as e:
                print(f"⚠️  配置文件加载失败，使用默认配置: {e}")
        
        # 更新配置
        config = config_manager.optimization_config
        config.n_calls = args.n_calls
        config.n_initial_points = args.n_initial_points
        config.random_state = args.random_state
        config.use_cache = not args.no_cache
        config.early_stopping = not args.no_early_stopping
        config.sensitivity_analysis = not args.no_sensitivity
        
        # 执行优化
        start_time = time.time()
        result = optimize_mesh_parameters(
            n_calls=args.n_calls,
            optimizer=args.optimizer,
            evaluator_type=args.evaluator,
            use_cache=not args.no_cache
        )
        execution_time = time.time() - start_time
        
        # 输出结果
        print(f"\n🎉 优化完成！")
        print(f"   执行时间: {execution_time:.2f}秒")
        print(f"   最佳目标值: {result['best_value']:.6f}")
        
        print(f"\n📊 最佳参数:")
        for name, value in result['best_params'].items():
            if isinstance(value, float):
                print(f"   {name}: {value:.6f}")
            else:
                print(f"   {name}: {value}")
        
        # 显示额外信息
        if 'total_evaluations' in result:
            print(f"\n📈 统计信息:")
            print(f"   总评估次数: {result['total_evaluations']}")
            if result['total_evaluations'] > 0:
                print(f"   平均评估时间: {execution_time/result['total_evaluations']:.3f}秒")
        
        # 保存结果（如果指定输出文件）
        if args.output:
            try:
                save_optimization_result(result, args.output, args.save_plots)
                print(f"✓ 结果已保存到: {args.output}")
            except Exception as e:
                print(f"⚠️  保存结果失败: {e}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断优化")
        return 130
    except Exception as e:
        logger.error(f"优化失败: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

def cmd_compare(args, modules) -> int:
    """执行比较命令"""
    logger = logging.getLogger(__name__)
    optimize_mesh_parameters, MeshOptimizer, compare_optimizers, config_manager, check_dependencies = modules
    
    try:
        print(f"🔍 开始优化器比较")
        print(f"   优化器: {', '.join(args.optimizers)}")
        print(f"   评估器: {args.evaluator}")
        print(f"   迭代次数: {args.n_calls} × {args.n_runs} 运行")
        
        # 检查优化器可用性
        deps = check_dependencies()
        unavailable_optimizers = []
        
        for optimizer in args.optimizers:
            if optimizer in ['bayesian', 'random', 'forest'] and not deps['skopt_available']:
                unavailable_optimizers.append(optimizer)
        
        if unavailable_optimizers:
            print(f"⚠️  以下优化器不可用（需要 scikit-optimize）: {', '.join(unavailable_optimizers)}")
            available_optimizers = [opt for opt in args.optimizers if opt not in unavailable_optimizers]
            if not available_optimizers:
                print("❌ 没有可用的优化器")
                return 1
            args.optimizers = available_optimizers
            print(f"✓ 继续使用可用优化器: {', '.join(available_optimizers)}")
        
        # 运行比较
        start_time = time.time()
        results = compare_optimizers(
            optimizers=args.optimizers,
            n_calls=args.n_calls,
            n_runs=args.n_runs,
            evaluator_type=args.evaluator,
            run_sensitivity_analysis=not args.no_sensitivity,
            generate_report=not args.no_report
        )
        execution_time = time.time() - start_time
        
        # 输出结果
        print(f"\n🎉 比较完成！")
        print(f"   总执行时间: {execution_time:.2f}秒")
        
        if 'best_optimizer' in results:
            best_opt = results['best_optimizer']
            best_info = results['best_optimizer_info']
            
            print(f"\n🏆 推荐的最佳优化器: {best_opt}")
            print(f"   平均最佳值: {best_info['mean_best_value']:.6f}")
            print(f"   标准差: {best_info['std_best_value']:.6f}")
            print(f"   平均执行时间: {best_info['mean_execution_time']:.2f}秒")
            print(f"   成功运行次数: {best_info['successful_runs']}")
        
        # 显示所有结果摘要
        if 'summary' in results and results['summary']:
            print(f"\n📊 详细比较结果:")
            summary_data = results['summary']
            for optimizer_data in summary_data:
                opt_name = optimizer_data['optimizer']
                mean_val = optimizer_data['mean_best_value']
                std_val = optimizer_data['std_best_value']
                exec_time = optimizer_data['mean_execution_time']
                print(f"   {opt_name:12}: {mean_val:.6f} ± {std_val:.6f} ({exec_time:.2f}s)")
        
        if 'results_dir' in results:
            print(f"\n📁 详细结果保存在: {results['results_dir']}")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断比较")
        return 130
    except Exception as e:
        logger.error(f"比较失败: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

def cmd_config(args, modules) -> int:
    """执行配置命令"""
    logger = logging.getLogger(__name__)
    optimize_mesh_parameters, MeshOptimizer, compare_optimizers, config_manager, check_dependencies = modules
    
    try:
        if args.config_action == 'generate':
            output_file = args.output
            
            if args.example:
                config_manager.create_example_config(output_file)
                print(f"✓ 示例配置文件已生成: {output_file}")
            else:
                config_manager.save_config(output_file)
                print(f"✓ 默认配置文件已生成: {output_file}")
            
            print(f"\n📝 配置文件说明:")
            print(f"   - optimization: 优化器设置")
            print(f"   - ansa: Ansa软件配置")
            print(f"   - parameter_space: 参数空间定义")
            
        elif args.config_action == 'validate':
            try:
                config_manager.load_config(args.config_file)
                print(f"✓ 配置文件 {args.config_file} 验证通过")
                
                # 显示配置摘要
                summary = config_manager.get_config_summary()
                print(f"\n📊 配置摘要:")
                for section, info in summary.items():
                    print(f"   {section}:")
                    for key, value in info.items():
                        print(f"     {key}: {value}")
                        
            except Exception as e:
                print(f"❌ 配置文件验证失败: {e}")
                return 1
        
        elif args.config_action == 'show':
            summary = config_manager.get_config_summary()
            
            if args.section:
                if args.section in summary:
                    print(f"📋 {args.section} 配置:")
                    for key, value in summary[args.section].items():
                        print(f"   {key}: {value}")
                else:
                    print(f"❌ 未知配置节: {args.section}")
                    return 1
            else:
                print(f"📋 完整配置:")
                for section, info in summary.items():
                    print(f"\n{section}:")
                    for key, value in info.items():
                        print(f"   {key}: {value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"配置操作失败: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

def cmd_info(args, modules=None) -> int:
    """显示系统信息"""
    print(f"📋 {APP_NAME} v{APP_VERSION}")
    print("=" * 50)
    
    # Python信息
    print(f"🐍 Python版本: {sys.version}")
    print(f"📍 Python路径: {sys.executable}")
    print(f"📂 工作目录: {Path.cwd()}")
    
    # 依赖库检查
    if args.check_deps:
        print(f"\n🔍 依赖库检查:")
        print("-" * 30)
        
        dependencies = [
            ('numpy', '数值计算', True),
            ('scikit-optimize', '贝叶斯优化', False),
            ('matplotlib', '可视化', False),
            ('pandas', '数据分析', False),
            ('seaborn', '统计图表', False),
            ('scipy', '科学计算', False),
        ]
        
        for lib_name, description, required in dependencies:
            try:
                __import__(lib_name)
                status = "✓ 已安装"
                color = ""
            except ImportError:
                status = "✗ 未安装"
                color = "" if not required else "❌ "
            
            req_text = "必需" if required else "可选"
            print(f"   {color}{lib_name:<20} {description:<15} {status:<10} ({req_text})")
    
    # Ansa环境检查
    if args.check_ansa:
        print(f"\n🔧 Ansa环境检查:")
        print("-" * 30)
        
        try:
            import ansa
            print("   ✓ Ansa模块可用")
            
            # 尝试检查Ansa版本
            try:
                # 这里可以添加更多Ansa特定的检查
                print("   ✓ Ansa导入成功")
            except Exception as e:
                print(f"   ⚠️  Ansa检查警告: {e}")
                
        except ImportError:
            print("   ○ Ansa模块不可用（将使用模拟模式）")
    
    # 性能测试
    if args.performance:
        print(f"\n⚡ 性能测试:")
        print("-" * 30)
        
        try:
            run_performance_test()
        except Exception as e:
            print(f"   ❌ 性能测试失败: {e}")
    
    # 模块导入测试
    print(f"\n🧪 模块导入测试:")
    print("-" * 30)
    success, missing, available = check_and_import_modules()
    
    if success:
        print("   ✓ 所有必要模块导入成功")
        print(f"   📦 可用模块: {len(available)} 个")
    else:
        print("   ❌ 模块导入失败")
        print(f"   📦 缺少模块: {len(missing)} 个")
    
    return 0

def cmd_test(args, modules) -> int:
    """运行测试命令"""
    logger = logging.getLogger(__name__)
    
    try:
        print(f"🧪 运行系统测试")
        
        if args.quick:
            print("   模式: 快速测试")
            test_iterations = 5
        else:
            print("   模式: 标准测试")
            test_iterations = 10
        
        print(f"   评估器: {args.evaluator}")
        
        # 导入测试所需模块
        if not modules:
            success, modules = import_core_modules()
            if not success:
                return 1
        
        optimize_mesh_parameters, MeshOptimizer, compare_optimizers, config_manager, check_dependencies = modules
        
        # 运行基础功能测试
        success = run_basic_tests(modules, args.evaluator, test_iterations, args.verbose_test)
        
        if success:
            print(f"\n✅ 所有测试通过!")
            return 0
        else:
            print(f"\n❌ 部分测试失败!")
            return 1
            
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

def save_optimization_result(result: Dict[str, Any], output_file: str, save_plots: bool = False) -> None:
    """保存优化结果"""
    import json
    from pathlib import Path
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备保存数据
    output_data = {
        'metadata': {
            'app_name': APP_NAME,
            'app_version': APP_VERSION,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimizer': result.get('optimizer', 'unknown')
        },
        'best_params': result['best_params'],
        'best_value': result['best_value'],
        'execution_time': result.get('execution_time', 0),
        'total_evaluations': result.get('total_evaluations', 0),
        'optimizer_name': result.get('optimizer_name', 'Unknown')
    }
    
    # 添加额外信息（如果可用）
    if 'convergence_info' in result:
        output_data['convergence_info'] = result['convergence_info']
    
    # 保存JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 保存图表（如果请求）
    if save_plots and 'report_dir' in result:
        print(f"📊 优化图表已保存到: {result['report_dir']}")

def run_performance_test() -> None:
    """运行性能测试"""
    import time
    
    # 测试numpy运算性能
    try:
        import numpy as np
        
        print("   🧮 NumPy矩阵运算测试...")
        start_time = time.time()
        
        # 创建大矩阵并执行运算
        size = 1000
        a = np.random.random((size, size))
        b = np.random.random((size, size))
        c = np.dot(a, b)
        
        numpy_time = time.time() - start_time
        print(f"      {size}x{size} 矩阵乘法: {numpy_time:.3f}秒")
        
        if numpy_time < 1.0:
            print("      ✓ 性能良好")
        elif numpy_time < 5.0:
            print("      ○ 性能一般")
        else:
            print("      ⚠️  性能较慢")
            
    except ImportError:
        print("   ○ NumPy不可用，跳过性能测试")
    
    # 测试文件I/O性能
    print("   💾 文件I/O测试...")
    try:
        import tempfile
        
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
            # 写入测试数据
            for i in range(10000):
                f.write(f"test line {i}\n")
            f.flush()
            
        io_time = time.time() - start_time
        print(f"      写入10K行: {io_time:.3f}秒")
        
        if io_time < 0.1:
            print("      ✓ I/O性能良好")
        else:
            print("      ○ I/O性能一般")
            
    except Exception as e:
        print(f"      ❌ I/O测试失败: {e}")

def run_basic_tests(modules, evaluator_type: str, n_iterations: int, verbose: bool) -> bool:
    """运行基础功能测试"""
    optimize_mesh_parameters, MeshOptimizer, compare_optimizers, config_manager, check_dependencies = modules
    
    all_tests_passed = True
    
    try:
        print(f"\n1️⃣  测试参数验证...")
        
        # 测试配置验证
        is_valid, error_msg = config_manager.optimization_config.validate()
        if is_valid:
            print("   ✓ 配置验证通过")
        else:
            print(f"   ❌ 配置验证失败: {error_msg}")
            all_tests_passed = False
        
        print(f"\n2️⃣  测试评估器...")
        
        # 测试评估器
        from evaluators.mesh_evaluator import create_mesh_evaluator
        evaluator = create_mesh_evaluator(evaluator_type)
        
        test_params = {
            'element_size': 1.0,
            'mesh_density': 2.0,
            'mesh_quality_threshold': 0.5,
            'smoothing_iterations': 40,
            'mesh_growth_rate': 1.0,
            'mesh_topology': 2
        }
        
        if evaluator.validate_params(test_params):
            print("   ✓ 参数验证通过")
        else:
            print("   ❌ 参数验证失败")
            all_tests_passed = False
        
        # 测试评估功能
        result = evaluator.evaluate_mesh(test_params)
        if isinstance(result, (int, float)) and result >= 0:
            print(f"   ✓ 评估器工作正常 (结果: {result:.6f})")
        else:
            print(f"   ❌ 评估器返回无效结果: {result}")
            all_tests_passed = False
        
        print(f"\n3️⃣  测试优化功能...")
        
        # 测试基础优化
        try:
            result = optimize_mesh_parameters(
                n_calls=n_iterations,
                optimizer='genetic',  # 使用总是可用的遗传算法
                evaluator_type=evaluator_type,
                use_cache=False
            )
            
            if 'best_value' in result and isinstance(result['best_value'], (int, float)):
                print(f"   ✓ 优化功能正常 (最佳值: {result['best_value']:.6f})")
            else:
                print("   ❌ 优化返回无效结果")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ 优化测试失败: {e}")
            if verbose:
                traceback.print_exc()
            all_tests_passed = False
        
        if n_iterations >= 10:  # 只在标准测试中运行
            print(f"\n4️⃣  测试比较功能...")
            
            try:
                comparison_results = compare_optimizers(
                    optimizers=['random', 'genetic'],
                    n_calls=5,  # 快速测试
                    n_runs=1,
                    evaluator_type=evaluator_type,
                    run_sensitivity_analysis=False,
                    generate_report=False
                )
                
                if 'best_optimizer' in comparison_results:
                    print(f"   ✓ 比较功能正常 (推荐: {comparison_results['best_optimizer']})")
                else:
                    print("   ○ 比较功能运行但无推荐结果")
                    
            except Exception as e:
                print(f"   ❌ 比较测试失败: {e}")
                if verbose:
                    traceback.print_exc()
                all_tests_passed = False
        
        return all_tests_passed
        
    except Exception as e:
        print(f"❌ 测试运行异常: {e}")
        if verbose:
            traceback.print_exc()
        return False

def main() -> int:
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    # 设置日志
    setup_logging(args.verbose, args.log_file)
    
    # 检查命令
    if not args.command:
        parser.print_help()
        return 1
    
    # 对于info命令，不需要导入复杂模块
    if args.command == 'info':
        return cmd_info(args)
    
    # 检查和导入模块
    print("🔍 检查系统环境...")
    success, missing, available = check_and_import_modules()
    
    if not success:
        print(f"\n❌ 系统环境检查失败")
        print(f"建议操作:")
        print(f"  1. 确保所有必需的Python文件存在")
        print(f"  2. 检查文件权限")
        print(f"  3. 运行: pip install -r requirements.txt")
        return 1
    
    print("✓ 系统环境检查通过")
    
    # 导入核心模块
    print("📦 加载核心模块...")
    success, modules = import_core_modules()
    if not success:
        return 1
    
    print("✓ 核心模块加载成功")
    
    # 执行对应命令
    try:
        if args.command == 'optimize':
            return cmd_optimize(args, modules)
        elif args.command == 'compare':
            return cmd_compare(args, modules)
        elif args.command == 'config':
            return cmd_config(args, modules)
        elif args.command == 'test':
            return cmd_test(args, modules)
        else:
            print(f"❌ 未知命令: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断程序")
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception("程序异常")
        print(f"❌ 程序异常: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 再见!")
        sys.exit(130)
    except Exception as e:
        print(f"💥 未捕获的异常: {e}")
        logging.getLogger(__name__).exception("未捕获的异常")
        sys.exit(1)