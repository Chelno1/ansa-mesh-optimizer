#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的运行时钩子 - 仅处理必要的初始化
"""

import os
import sys
import platform
from pathlib import Path

def setup_environment():
    """设置基本环境变量"""
    try:
        # 设置matplotlib为无头模式
        os.environ['MPLBACKEND'] = 'Agg'
        
        # 设置临时目录
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        app_temp_dir = temp_dir / 'ansa-mesh-optimizer'
        
        if not app_temp_dir.exists():
            app_temp_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ['ANSA_OPTIMIZER_TEMP'] = str(app_temp_dir)
        
        return True
    except Exception:
        return False

def main():
    """静默初始化"""
    setup_environment()

if __name__ == '__main__':
    main()