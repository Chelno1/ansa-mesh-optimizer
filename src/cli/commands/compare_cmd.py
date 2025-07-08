"""
比较命令处理器
"""

import logging
import traceback
import time

def register_compare_command(subparsers):
    """注册比较命令"""
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