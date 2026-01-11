# 新增或修改我时需要修改这个文件夹中的README.md文件
"""配置加载器测试"""
import pytest
from pathlib import Path
from src.utils.config_loader import load_prompt_template, load_taxonomy_files


class TestConfigLoader:
    """配置加载器测试类"""
    
    def test_load_prompt_template(self):
        """测试Prompt模板加载"""
        # 测试extraction prompt
        template = load_prompt_template("extraction", "v1.0")
        assert isinstance(template, str)
        assert len(template) > 0
    
    def test_load_taxonomy_files(self):
        """测试词表文件加载"""
        taxonomy = load_taxonomy_files()
        
        assert "aspect_synonyms" in taxonomy
        assert "noise_terms" in taxonomy
        assert "aspect_allowlist" in taxonomy
        
        # 检查同义词表
        assert isinstance(taxonomy["aspect_synonyms"], dict)
        
        # 检查噪声词表
        assert isinstance(taxonomy["noise_terms"], dict)

