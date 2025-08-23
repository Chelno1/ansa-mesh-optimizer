.PHONY: install install-dev format lint test check-newlines setup-pre-commit clean

# 安装项目
install:
	pip install -e .

# 安装开发依赖
install-dev:
	pip install -e .[dev]

# 设置pre-commit钩子
setup-pre-commit:
	pre-commit install

# 格式化代码
format:
	black src/ tests/
	isort src/ tests/

# 代码检查
lint:
	flake8 src/ tests/
	mypy src/

# 检查行尾换行符
check-newlines:
	@echo "检查Python文件的行尾换行符..."
	@find src -name "*.py" -exec sh -c 'if [ -n "$$(tail -c1 "$$1")" ]; then echo "缺少换行符: $$1"; exit 1; fi' _ {} \;
	@echo "所有Python文件都有正确的行尾换行符！"

# 运行测试
test:
	pytest tests/ -v --cov=src

# 完整的代码质量检查
check: format lint check-newlines test

# 清理缓存文件
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/

# 开发环境完整设置
dev-setup: install-dev setup-pre-commit
	@echo "开发环境设置完成！"
	@echo "运行 'make check' 来验证代码质量"