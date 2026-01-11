# tests/ 目录说明

本目录包含单元测试。

## 运行测试

```bash
# 使用pytest运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_normalizer.py

# 运行特定测试类
pytest tests/test_normalizer.py::TestTextNormalizer

# 显示详细输出
pytest tests/ -v
```

## 测试文件

- `test_normalizer.py`: TextNormalizer测试
- `test_config_loader.py`: 配置加载器测试
- `test_version_fields.py`: 版本字段测试

## 添加新测试

1. 在`tests/`目录下创建新的测试文件
2. 测试文件命名：`test_*.py`
3. 测试类命名：`Test*`
4. 测试方法命名：`test_*`

