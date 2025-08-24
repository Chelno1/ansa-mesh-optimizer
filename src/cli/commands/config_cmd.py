"""
配置命令处理器
"""

import logging
import traceback


def register_config_command(subparsers):
    """注册配置命令"""
    config_parser = subparsers.add_parser("config", help="配置管理")
    config_subparsers = config_parser.add_subparsers(dest="config_action")

    # 生成默认配置
    generate_parser = config_subparsers.add_parser("generate", help="生成默认配置文件")
    generate_parser.add_argument(
        "--output", type=str, default="default_config.json", help="配置文件输出路径"
    )
    generate_parser.add_argument(
        "--example", action="store_true", help="生成示例配置而非默认配置"
    )

    # 验证配置
    validate_parser = config_subparsers.add_parser("validate", help="验证配置文件")
    validate_parser.add_argument("config_file", help="要验证的配置文件")

    # 显示配置
    show_parser = config_subparsers.add_parser("show", help="显示当前配置")
    show_parser.add_argument(
        "--section",
        choices=["optimization", "ansa", "parameter_space"],
        help="显示特定配置节",
    )


def cmd_config(args, modules) -> int:
    """执行配置命令"""
    logger = logging.getLogger(__name__)
    (
        optimize_mesh_parameters,
        MeshOptimizer,
        compare_optimizers,
        ConfigManagerClass,
        check_dependencies,
    ) = modules

    try:
        # 创建配置管理器实例
        config_manager = ConfigManagerClass()

        if args.config_action == "generate":
            output_file = args.output

            if args.example:
                config_manager.create_example_config(output_file)
                print(f"✓ 示例配置文件已生成: {output_file}")
            else:
                config_manager.save_config(output_file)
                print(f"✓ 默认配置文件已生成: {output_file}")

            print("\n📝 配置文件说明:")
            print("   - optimization: 优化器设置")
            print("   - ansa: Ansa软件配置")
            print("   - parameter_space: 参数空间定义")

        elif args.config_action == "validate":
            try:
                # 为验证操作创建新的配置管理器实例
                validate_manager = ConfigManagerClass(
                    config_file=args.config_file, require_config=True
                )
                print(f"✓ 配置文件 {args.config_file} 验证通过")

                # 显示配置摘要
                summary = validate_manager.get_config_summary()
                print("\n📊 配置摘要:")
                for section, info in summary.items():
                    print(f"   {section}:")
                    for key, value in info.items():
                        print(f"     {key}: {value}")

            except Exception as e:
                print(f"❌ 配置文件验证失败: {e}")
                return 1

        elif args.config_action == "show":
            # 为显示操作使用默认配置管理器
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
                print("📋 完整配置:")
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
