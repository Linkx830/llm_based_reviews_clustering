# utils/ 模块说明

## 职责

工具函数模块，提供通用功能。

## 文件说明

- `config_loader.py`: 配置加载工具
  - `load_prompt_template()`: 加载Prompt模板
  - `load_run_config()`: 加载运行配置
  - `load_taxonomy_files()`: 加载词表文件

- `logger.py`: 日志工具
  - `setup_logger()`: 设置日志记录器
  - `get_logger()`: 获取日志记录器

- `retry.py`: 重试机制
  - `retry_with_backoff()`: 带指数退避的重试装饰器
  - `retry_on_failure()`: 简单重试装饰器

- `normalizer.py`: 文本归一化工具
  - `TextNormalizer`: 同义词归一、噪声识别、有效性判断

## 使用示例

```python
from src.utils import (
    load_prompt_template,
    setup_logger,
    retry_with_backoff,
    TextNormalizer
)

# 加载Prompt模板
template = load_prompt_template("extraction", "v1.0")

# 设置日志
logger = setup_logger("my_module")

# 使用重试装饰器
@retry_with_backoff(max_retries=3)
def my_function():
    # ...
    pass

# 使用归一化器
normalizer = TextNormalizer()
aspect_norm = normalizer.normalize_aspect("shipping")
is_noise = normalizer.is_noise_aspect("product")
```

