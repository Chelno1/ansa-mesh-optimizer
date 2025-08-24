#!/usr/bin/env python3
"""
测试贝叶斯优化收敛图生成的脚本
"""

import sys
import logging
from pathlib import Path
from ansa_mesh_optimizer.optimizers.optimizer_config import OptimizationResult
from ansa_mesh_optimizer.visualization.optimization_visualizer import OptimizationVisualizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_skopt_result():
    """创建模拟的scikit-optimize结果"""
    class MockSkoptResult:
        def __init__(self):
            self.func_vals = [10.0, 8.5, 7.2, 6.8, 6.5, 6.2, 6.0, 5.8, 5.7, 5.6]
            self.x_iters = [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4], 
                           [1.5, 2.5], [1.6, 2.6], [1.7, 2.7], [1.8, 2.8], [1.9, 2.9]]
            self.x = [1.9, 2.9]
            self.fun = 5.6
            self.__class__.__module__ = 'skopt.optimizer.optimizer'
    
    return MockSkoptResult()

def test_convergence_plot():
    """测试收敛图生成"""
    logger.info("开始测试收敛图生成...")
    
    # 创建模拟的优化结果
    mock_skopt_result = create_mock_skopt_result()
    
    optimization_history = [
        {'params': {'param1': 1.0, 'param2': 2.0}, 'result': 10.0, 'timestamp': '2024-01-01T00:00:00'},
        {'params': {'param1': 1.1, 'param2': 2.1}, 'result': 8.5, 'timestamp': '2024-01-01T00:01:00'},
        {'params': {'param1': 1.2, 'param2': 2.2}, 'result': 7.2, 'timestamp': '2024-01-01T00:02:00'},
        {'params': {'param1': 1.9, 'param2': 2.9}, 'result': 5.6, 'timestamp': '2024-01-01T00:09:00'},
    ]
    
    # 创建OptimizationResult对象
    result = OptimizationResult(
        best_params={'param1': 1.9, 'param2': 2.9},
        best_value=5.6,
        optimizer_name='Bayesian Optimization',
        optimization_history=optimization_history,
        skopt_result=mock_skopt_result,
        convergence_info={'n_calls': 10, 'best_iteration': 9, 'improvement_ratio': 0.44}
    )
    
    # 创建可视化器
    output_dir = Path("test_convergence_output")
    visualizer = OptimizationVisualizer(output_dir)
    
    # 生成收敛图
    logger.info("生成收敛图...")
    visualizer.generate_optimization_plots(result)
    
    # 检查输出文件
    convergence_file = output_dir / "convergence.png"
    if convergence_file.exists():
        logger.info(f"收敛图生成成功: {convergence_file}")
        return True
    else:
        logger.error("收敛图生成失败")
        return False

if __name__ == "__main__":
    success = test_convergence_plot()
    sys.exit(0 if success else 1)