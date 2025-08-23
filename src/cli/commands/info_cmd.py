"""
信息命令处理器
"""

import sys
from pathlib import Path

def register_info_command(subparsers):
    """注册信息命令"""
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.add_argument('--check-deps', action='store_true',
                            help='检查依赖库')
    info_parser.add_argument('--check-ansa', action='store_true',
                            help='检查Ansa环境')
    info_parser.add_argument('--performance', action='store_true',
                            help='运行性能测试')

def cmd_info(args, modules=None) -> int:
    """显示系统信息"""
    APP_NAME = "Ansa Mesh Optimizer"
    APP_VERSION = "1.3.0"
    
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
        
        # 依赖配置：(显示名称, 导入名称, 描述, 是否必需)
        dependencies = [
            ('numpy', 'numpy', '数值计算', True),
            ('scikit-optimize', 'skopt', '贝叶斯优化', False),
            ('matplotlib', 'matplotlib', '可视化', False),
            ('pandas', 'pandas', '数据分析', False),
            ('seaborn', 'seaborn', '统计图表', False),
            ('scipy', 'scipy', '科学计算', False),
        ]
        
        for display_name, import_name, description, required in dependencies:
            try:
                __import__(import_name)
                status = "✓ 已安装"
                color = ""
            except ImportError:
                status = "✗ 未安装"
                color = "" if not required else "❌ "
            
            req_text = "必需" if required else "可选"
            print(f"   {color}{display_name:<20} {description:<15} {status:<10} ({req_text})")
    
    # Ansa环境检查
    if args.check_ansa:
        print(f"\n🔧 Ansa环境检查:")
        print("-" * 30)
        
        try:
            import ansa  # type: ignore
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
    
    # 导入命令分发器中的检查函数
    from .command_dispatcher import check_and_import_modules
    success, missing, available = check_and_import_modules()
    
    if success:
        print("   ✓ 所有必要模块导入成功")
        print(f"   📦 可用模块: {len(available)} 个")
    else:
        print("   ❌ 模块导入失败")
        print(f"   📦 缺少模块: {len(missing)} 个")
    
    return 0

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
