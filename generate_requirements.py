#!/usr/bin/env python3
"""
生成requirements.txt文件的脚本
扫描项目中的所有Python文件，识别第三方库并获取其安装版本
"""

import importlib.metadata
import sys
from pathlib import Path

def get_installed_version(package_name):
    """获取已安装包的版本"""
    try:
        # 尝试不同的包名变体
        name_variants = [
            package_name,
            package_name.lower(),
            package_name.replace('_', '-'),
            package_name.replace('-', '_'),
        ]
        
        # 特殊情况映射
        special_cases = {
            'skopt': 'scikit-optimize',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'PyInstaller': 'pyinstaller',
        }
        
        if package_name in special_cases:
            name_variants.insert(0, special_cases[package_name])
        
        for name in name_variants:
            try:
                version = importlib.metadata.version(name)
                return version
            except importlib.metadata.PackageNotFoundError:
                continue
        
        return None
    except Exception as e:
        return None

def main():
    """主函数"""
    
    # 基于代码扫描和pyproject.toml识别的第三方库
    third_party_libs = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scikit-optimize': 'skopt',  # import名是skopt，包名是scikit-optimize
        'scipy': 'scipy',
        'pandas': 'pandas',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm',
        'joblib': 'joblib',
        'psutil': 'psutil',
        'colorama': 'colorama',
        'pytest': 'pytest',
        'pyinstaller': 'PyInstaller',  # import名是PyInstaller
        'requests': 'requests',
    }
    
    # 收集已安装的包及其版本
    requirements = []
    not_installed = []
    
    for package_name, import_name in third_party_libs.items():
        version = get_installed_version(package_name)
        if version:
            requirements.append(f"{package_name}=={version}")
            print(f"✓ 找到 {package_name}: {version}")
        else:
            # 尝试使用import名
            version = get_installed_version(import_name)
            if version:
                requirements.append(f"{package_name}=={version}")
                print(f"✓ 找到 {package_name}: {version}")
            else:
                not_installed.append(package_name)
                print(f"✗ 未找到 {package_name} (可能未安装)")
    
    # 按字母顺序排序
    requirements.sort()
    
    # 生成requirements.txt文件
    output_file = Path("requirements.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for req in requirements:
            f.write(req + '\n')
    
    # 输出结果
    print("\n" + "="*50)
    print(f"requirements.txt 文件已成功生成，共包含 {len(requirements)} 个依赖项。")
    
    if not_installed:
        print(f"\n警告：以下 {len(not_installed)} 个库未找到安装版本：")
        for lib in not_installed:
            print(f"  - {lib}")
        print("\n这些库可能：")
        print("  1. 未在当前环境中安装")
        print("  2. 是可选依赖")
        print("  3. 包名与导入名不一致")
    
    print(f"\n文件已保存到: {output_file.absolute()}")
    
    # 显示文件内容
    print("\n" + "="*50)
    print("requirements.txt 内容：")
    print("-"*50)
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    
    return len(requirements)

if __name__ == "__main__":
    try:
        count = main()
        sys.exit(0)
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)