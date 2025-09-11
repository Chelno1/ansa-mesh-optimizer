
import sys
import os

# 修复numpy导入问题
if hasattr(sys, '_MEIPASS'):
    # 确保numpy从正确的位置导入
    import numpy
    # 设置numpy的环境变量
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    print(f"NumPy version: {numpy.__version__}")
    print(f"NumPy location: {numpy.__file__}")
