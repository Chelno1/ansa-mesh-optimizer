#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试遗传算法优化历史记录
"""

import sys
import os
import json
import logging
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_genetic_optimization_history():
    """测试遗传算法优化历史记录"""
    print("🧬 测试遗传算法优化历史记录...")
    
    try:
        from core.ansa_mesh_optimizer import optimize_mesh_parameters, UnifiedConfigManager, ConfigManagerWrapper
        
        # 使用默认配置
        config_file = "src/default_config.json"
        if not os.path.exists(config_file):
            print(f"❌ 配置文件不存在: {config_file}")
            return
        
        print(f"✓ 使用配置文件: {config_file}")
        
        # 创建配置管理器
        unified_manager = UnifiedConfigManager(config_file=config_file, require_config=True)
        config_manager = ConfigManagerWrapper(unified_manager)
        
        # 执行遗传算法优化
        print("🚀 开始遗传算法优化...")
        result = optimize_mesh_parameters(
            n_calls=20,  # 少量迭代用于测试
            optimizer='genetic',
            evaluator_type='mock',
            use_cache=False,  # 禁用缓存以确保每次都重新评估
            config_manager=config_manager
        )
        
        print(f"✓ 优化完成!")
        print(f"  最佳参数: {result.best_params}")
        print(f"  最佳值: {result.best_value:.6f}")
        print(f"  评估次数: {result.n_evaluations}")
        
        # 检查历史记录
        if hasattr(result, 'optimization_history') and result.optimization_history:
            print(f"✓ 找到优化历史记录，共 {len(result.optimization_history)} 条")
            
            # 显示前几条记录
            for i, entry in enumerate(result.optimization_history[:3]):
                print(f"  记录 {i+1}: params={entry.get('params', {})}, result={entry.get('result', 'N/A')}")
            
        else:
            print("❌ 未找到优化历史记录")
        
        # 检查生成的报告目录
        print("\n📁 检查生成的报告目录...")
        report_dirs = list(Path("optimization_reports").glob("*Genetic*")) if Path("optimization_reports").exists() else []
        
        if report_dirs:
            latest_dir = max(report_dirs, key=lambda x: x.stat().st_mtime)
            print(f"✓ 找到报告目录: {latest_dir}")
            
            # 检查optimization_history.json文件
            history_file = latest_dir / "optimization_history.json"
            if history_file.exists():
                print(f"✓ 找到历史文件: {history_file}")
                
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                print(f"  历史记录条数: {len(history_data)}")
                if len(history_data) > 0:
                    print(f"  第一条记录: {history_data[0]}")
                    print("✅ optimization_history.json文件记录正常!")
                else:
                    print("❌ optimization_history.json文件为空!")
            else:
                print(f"❌ 未找到历史文件: {history_file}")
        else:
            print("❌ 未找到报告目录")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_genetic_optimization_history()