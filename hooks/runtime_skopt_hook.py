"""
Runtime hook for scikit-optimize

This hook ensures skopt and its dependencies are properly initialized at runtime.
"""

import sys
import os

def setup_skopt():
    """Setup scikit-optimize for frozen execution"""
    print("=" * 50)
    print("scikit-optimize 运行时初始化")
    print("=" * 50)
    
    # 确保所有必要的包都在 sys.modules 中
    try:
        # 首先导入基础依赖
        import numpy
        print("✓ NumPy 已加载")
        
        import scipy
        import scipy.optimize
        import scipy.stats
        print("✓ SciPy 已加载")
        
        import sklearn
        import sklearn.base
        import sklearn.ensemble
        import sklearn.gaussian_process
        import sklearn.tree
        import sklearn.utils
        print("✓ scikit-learn 已加载")
        
        import joblib
        print("✓ joblib 已加载")
        
        # 现在尝试导入 skopt
        import skopt
        print(f"✓ scikit-optimize 版本: {skopt.__version__}")
        
        # 导入主要功能
        from skopt import gp_minimize, forest_minimize, dummy_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
        print("✓ scikit-optimize 核心功能已加载")
        
        # 测试基本功能
        test_space = [Real(0.0, 1.0, name='test')]
        print("✓ scikit-optimize 空间定义可用")
        
        print("=" * 50)
        print("scikit-optimize 初始化成功")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ scikit-optimize 初始化失败: {e}")
        print(f"   Python路径: {sys.path}")
        print(f"   已加载模块: {list(sys.modules.keys())[:20]}...")
        print("=" * 50)
        return False
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        print("=" * 50)
        return False

# 在导入时立即运行设置
if getattr(sys, 'frozen', False):
    setup_skopt()