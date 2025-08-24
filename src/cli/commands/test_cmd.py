"""
测试命令处理器
"""

import logging
import traceback


def register_test_command(subparsers):
    """注册测试命令"""
    test_parser = subparsers.add_parser("test", help="运行测试")
    test_parser.add_argument("--quick", action="store_true", help="快速测试")
    test_parser.add_argument(
        "--evaluator", choices=["mock", "ansa"], default="mock", help="测试使用的评估器"
    )
    test_parser.add_argument("--verbose-test", action="store_true", help="详细测试输出")


def cmd_test(args, modules) -> int:
    """运行测试命令"""
    logger = logging.getLogger(__name__)

    try:
        print("🧪 运行系统测试")

        if args.quick:
            print("   模式: 快速测试")
            test_iterations = 5
        else:
            print("   模式: 标准测试")
            test_iterations = 10

        print(f"   评估器: {args.evaluator}")

        # 导入测试所需模块
        if not modules:
            from .command_dispatcher import import_core_modules

            success, modules = import_core_modules()
            if not success:
                return 1

        if modules is None:
            print("❌ 模块导入失败")
            return 1

        (
            optimize_mesh_parameters,
            MeshOptimizer,
            compare_optimizers,
            UnifiedConfigManager,
            check_dependencies,
        ) = modules

        # 为测试创建配置管理器（使用默认配置文件）
        try:
            import os

            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "default_config.json"
            )
            # 直接使用类创建实例
            config_manager_class = UnifiedConfigManager
            config_manager_instance = config_manager_class()
            config_manager_instance.load_config(default_config_path)
            print("✓ 测试配置已从默认配置文件加载")
        except Exception as e:
            print(f"❌ 无法加载默认配置文件: {e}")
            return 1

        # 运行基础功能测试
        success = run_basic_tests(
            (
                optimize_mesh_parameters,
                MeshOptimizer,
                compare_optimizers,
                config_manager_instance,
                check_dependencies,
            ),
            args.evaluator,
            test_iterations,
            args.verbose_test,
        )

        if success:
            print("\n✅ 所有测试通过!")
            return 0
        else:
            print("\n❌ 部分测试失败!")
            return 1

    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


def run_basic_tests(
    modules, evaluator_type: str, n_iterations: int, verbose: bool
) -> bool:
    """运行基础功能测试"""
    (
        optimize_mesh_parameters,
        MeshOptimizer,
        compare_optimizers,
        config_manager,
        check_dependencies,
    ) = modules

    all_tests_passed = True

    try:
        print("\n1️⃣  测试参数验证...")

        # 测试配置验证
        is_valid, error_msg = config_manager.optimization_config.validate()
        if is_valid:
            print("   ✓ 配置验证通过")
        else:
            print(f"   ❌ 配置验证失败: {error_msg}")
            all_tests_passed = False

        print("\n2️⃣  测试评估器...")

        # 测试评估器
        from src.evaluators.mesh_evaluator import create_mesh_evaluator

        evaluator = create_mesh_evaluator(evaluator_type, config_manager=config_manager)

        test_params = {
            "distortion_distance": 20,
            "rule_fillet_width_1": 3.0,
            "rule_fillet_width_2": 10.0,
            "rule_fillet_width_3": 20.0,
            "rule_fillet_width_4": 30.0,
            "recognize_chamfers_min_angle": 20.0,
            "recognize_chamfers_max_angle": 70.0,
            "recognize_chamfers_max_width": 20.0,
            "rule_chamfer_width_1": 10.0,
            "distortion_angle": 0.0,
            "perimeter_distance": 0.667,
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

        print("\n3️⃣  测试优化功能...")

        # 测试基础优化
        try:
            # 为测试创建临时配置文件路径
            import os

            default_config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "default_config.json"
            )

            result = optimize_mesh_parameters(
                n_calls=n_iterations,
                optimizer="genetic",  # 使用总是可用的遗传算法
                evaluator_type=evaluator_type,
                use_cache=False,
                config_file=default_config_path,
            )

            if hasattr(result, "best_value") and isinstance(
                result.best_value, (int, float)
            ):
                print(f"   ✓ 优化功能正常 (最佳值: {result.best_value:.6f})")
            else:
                print("   ❌ 优化返回无效结果")
                all_tests_passed = False

        except Exception as e:
            print(f"   ❌ 优化测试失败: {e}")
            if verbose:
                traceback.print_exc()
            all_tests_passed = False

        if n_iterations >= 10:  # 只在标准测试中运行
            print("\n4️⃣  测试比较功能...")

            try:
                # 为比较测试使用配置文件
                import os

                default_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "default_config.json"
                )

                comparison_results = compare_optimizers(
                    optimizers=["random", "genetic"],
                    n_calls=5,  # 快速测试
                    n_runs=1,
                    evaluator_type=evaluator_type,
                    run_sensitivity_analysis=False,
                    generate_report=False,
                    config_file=default_config_path,
                )

                if "best_optimizer" in comparison_results:
                    print(
                        f"   ✓ 比较功能正常 (推荐: {comparison_results['best_optimizer']})"
                    )
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
