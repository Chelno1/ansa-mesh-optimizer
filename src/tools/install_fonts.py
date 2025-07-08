#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
字体安装脚本

用于在没有中文字体的系统上自动安装字体
"""

import requests
import zipfile
import tempfile
import shutil
import matplotlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def install_source_han_sans():
    """安装思源黑体"""
    try:
        # 思源黑体下载链接
        font_url = "https://bgithub.xyz/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_file = temp_path / "font.zip"
            
            print("正在下载思源黑体字体...")
            response = requests.get(font_url, stream=True)
            response.raise_for_status()
            
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("正在解压字体文件...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # 查找TTF文件
            ttf_files = list(temp_path.rglob("*.ttf"))
            if ttf_files:
                # 复制字体文件到matplotlib字体目录
                font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
                font_dir.mkdir(exist_ok=True)
                
                for ttf_file in ttf_files[:3]:  # 只安装前3个字体文件
                    target_file = font_dir / ttf_file.name
                    shutil.copy2(ttf_file, target_file)
                    print(f"字体文件已安装: {target_file.name}")
                
                # 清除matplotlib字体缓存
                matplotlib.font_manager._rebuild()
                
                print("字体安装完成！")
                return True
            else:
                print("未找到字体文件")
                return False
            
    except Exception as e:
        print(f"字体安装失败: {e}")
        return False

if __name__ == "__main__":
    install_source_han_sans()