"""
优化命令处理器
"""

import logging
import time
import traceback


def register_optimize_command(subparsers):
    """注册优化命令"""
    optimize_parser = subparsers.add_parser("optimize", help="运行单个优化器")
    optimize_parser.add_argument(
        "--optimizer",
        choices=["bayesian", "random", "forest", "genetic", "parallel"],
        default="bayesian",
        help="优化器类型 (默认: bayesian)",
    )
    optimize_parser.add_argument(
        "--evaluator",
        choices=["ansa", "mock", "mock_ackley", "mock_rastrigin"],
        default="mock",
        help="评估器类型 (默认: mock)",
    )
    optimize_parser.add_argument(
        "--n-calls", type=int, default=20, help="优化迭代次数 (默认: 20)"
    )
    optimize_parser.add_argument(
        "--n-initial-points", type=int, default=5, help="初始随机点数量 (默认: 5)"
    )
    optimize_parser.add_argument(
        "--random-state", type=int, default=42, help="随机种子 (默认: 42)"
    )
    optimize_parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    optimize_parser.add_argument(
        "--no-early-stopping", action="store_true", help="禁用早停"
    )
    optimize_parser.add_argument(
        "--no-sensitivity", action="store_true", help="禁用敏感性分析"
    )
    optimize_parser.add_argument("--output", type=str, help="结果输出文件路径")
    optimize_parser.add_argument(
        "--save-plots", action="store_true", help="保存优化图表"
    )
    optimize_parser.add_argument(
        "--no-display", action="store_true", help="禁用图表显示（无头模式）"
    )


def cmd_optimize(args, modules) -> int:
    """执行优化命令"""
    logger = logging.getLogger(__name__)
    (
        optimize_mesh_parameters,
        MeshOptimizer,
        compare_optimizers,
        UnifiedConfigManager,
        check_dependencies,
    ) = modules

    try:
        # 设置显示配置（推荐使用新的上下文管理器）
        no_display = hasattr(args, "no_display") and args.no_display
        if no_display:
            print("🖼️  已启用无头模式 - 图表将保存但不显示")

        # 使用上下文管理器来管理显示配置
        from src.utils.display_config import display_config

        with display_config(no_display=no_display):
            print("🚀 开始网格参数优化")
            print(f"   优化器: {args.optimizer}")
            print(f"   评估器: {args.evaluator}")
            print(f"   迭代次数: {args.n_calls}")

            # 检查优化器可用性
            deps = check_dependencies()
            if (
                args.optimizer in ["bayesian", "random", "forest"]
                and not deps["skopt_available"]
            ):
                print(f"❌ 优化器 {args.optimizer} 需要 scikit-optimize 库")
                print("请运行: pip install scikit-optimize")
                return 1

            # 检查配置文件是否提供
            if not args.config:
                print("❌ 错误: 未指定配置文件")
                print("请使用 --config 参数指定配置文件路径")
                print(
                    "示例: python main.py optimize --config my_config.json --optimizer bayesian"
                )
                print("可以使用以下命令生成默认配置文件:")
                print("  python main.py config generate --output my_config.json")
                return 1

            # 创建配置管理器并加载配置
            try:
                config_manager = UnifiedConfigManager(
                    config_file=args.config, require_config=True
                )
                print(f"✓ 配置已从 {args.config} 加载")
            except Exception as e:
                print(f"❌ 配置文件加载失败: {e}")
                return 1

            # 更新配置
            config = config_manager.optimization_config
            config.n_calls = args.n_calls
            config.n_initial_points = args.n_initial_points
            config.random_state = args.random_state
            config.use_cache = not args.no_cache
            config.early_stopping = not args.no_early_stopping
            config.sensitivity_analysis = not args.no_sensitivity

            # 创建配置管理器包装器
            from src.core.ansa_mesh_optimizer import ConfigManagerWrapper

            config_wrapper = ConfigManagerWrapper(config_manager)

            # 执行优化
            start_time = time.time()
            result = optimize_mesh_parameters(
                n_calls=args.n_calls,
                optimizer=args.optimizer,
                evaluator_type=args.evaluator,
                use_cache=not args.no_cache,
                config_manager=config_wrapper,
            )
            execution_time = time.time() - start_time

            # 输出结果
            print("\n🎉 优化完成！")
            print(f"   执行时间: {execution_time:.2f}秒")
            print(f"   最佳目标值: {result.best_value:.6f}")

            print("\n📊 最佳参数:")
            for name, value in result.best_params.items():
                if isinstance(value, float):
                    print(f"   {name}: {value:.6f}")
                else:
                    print(f"   {name}: {value}")

            # 显示额外信息
            if hasattr(result, "n_evaluations") and result.n_evaluations:
                print("\n📈 统计信息:")
                print(f"   总评估次数: {result.n_evaluations}")
                if result.n_evaluations > 0:
                    print(
                        f"   平均评估时间: {execution_time/result.n_evaluations:.3f}秒"
                    )

            # 保存结果（如果指定输出文件）
            if args.output:
                try:
                    save_optimization_result(result, args.output, args.save_plots)
                    print(f"✓ 结果已保存到: {args.output}")
                except Exception as e:
                    print(f"⚠️  保存结果失败: {e}")

            return 0

    except KeyboardInterrupt:
        print("\n⚠️  用户中断优化")
        return 130
    except Exception as e:
        logger.error(f"优化失败: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def save_optimization_result(
    result, output_file: str, save_plots: bool = False
) -> None:
    """保存优化结果"""
    import json
    import time
    from pathlib import Path

    APP_NAME = "Ansa Mesh Optimizer"
    APP_VERSION = "2.1.0"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 处理OptimizationResult对象或字典
    if hasattr(result, "to_dict"):
        # OptimizationResult对象
        output_data = {
            "metadata": {
                "app_name": APP_NAME,
                "app_version": APP_VERSION,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "optimizer": result.optimizer_name,
            },
            "best_params": result.best_params,
            "best_value": result.best_value,
            "execution_time": getattr(result, "execution_time", 0),
            "total_evaluations": getattr(result, "n_evaluations", 0),
            "optimizer_name": result.optimizer_name,
        }

        # 添加额外信息（如果可用）
        if hasattr(result, "convergence_info") and result.convergence_info:
            output_data["convergence_info"] = result.convergence_info
    else:
        # 字典格式（向后兼容）
        output_data = {
            "metadata": {
                "app_name": APP_NAME,
                "app_version": APP_VERSION,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "optimizer": result.get("optimizer", "unknown"),
            },
            "best_params": (
                result.best_params
                if hasattr(result, "best_params")
                else (
                    result.get("best_params", {})
                    if isinstance(result, dict)
                    else getattr(result, "x", [])
                )
            ),
            "best_value": getattr(result, "best_score", None)
            or (
                result.get("best_value", float("inf"))
                if isinstance(result, dict)
                else getattr(result, "fun", float("inf"))
            ),
            "execution_time": (
                getattr(result, "execution_time", 0)
                if hasattr(result, "execution_time")
                else result.get("execution_time", 0) if isinstance(result, dict) else 0
            ),
            "total_evaluations": result.get("total_evaluations", 0),
            "optimizer_name": result.get("optimizer_name", "Unknown"),
        }

        # 添加额外信息（如果可用）
        # Handle both dictionary and object result formats
        if isinstance(result, dict) and "convergence_info" in result:
            output_data["convergence_info"] = result["convergence_info"]
        elif (
            "convergence_info" in result
        ):  # Use 'in' operator which works with our __contains__ method
            output_data["convergence_info"] = result["convergence_info"]
        elif hasattr(result, "convergence_info"):
            output_data["convergence_info"] = getattr(result, "convergence_info", None)

    # 保存JSON文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # 保存图表（如果请求）
    if save_plots:
        print("📊 优化图表保存功能已启用")
