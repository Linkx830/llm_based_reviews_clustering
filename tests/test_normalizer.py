# 新增或修改我时需要修改这个文件夹中的README.md文件
"""TextNormalizer测试"""
import pytest
from src.utils.normalizer import TextNormalizer


class TestTextNormalizer:
    """TextNormalizer测试类"""
    
    def test_normalize_aspect_with_synonyms(self):
        """测试同义词归一"""
        normalizer = TextNormalizer()
        
        # 测试同义词映射
        assert normalizer.normalize_aspect("shipping") == "delivery"
        assert normalizer.normalize_aspect("battery life") == "battery"
        assert normalizer.normalize_aspect("screen quality") == "display"
    
    def test_is_noise_aspect(self):
        """测试噪声aspect识别"""
        normalizer = TextNormalizer()
        
        # 噪声词
        assert normalizer.is_noise_aspect("product") is True
        assert normalizer.is_noise_aspect("item") is True
        assert normalizer.is_noise_aspect("it") is True
        
        # 正常词
        assert normalizer.is_noise_aspect("battery") is False
        assert normalizer.is_noise_aspect("screen") is False
    
    def test_is_noise_issue(self):
        """测试噪声issue识别"""
        normalizer = TextNormalizer()
        
        # 噪声issue
        assert normalizer.is_noise_issue("good") is True
        assert normalizer.is_noise_issue("bad") is True
        assert normalizer.is_noise_issue("works") is True
        
        # 正常issue
        assert normalizer.is_noise_issue("drains fast") is False
        assert normalizer.is_noise_issue("stopped working") is False
    
    def test_judge_validity(self):
        """测试有效性判断"""
        normalizer = TextNormalizer()
        
        # VALID
        assert normalizer.judge_validity("battery", "drains fast") == "VALID"
        assert normalizer.judge_validity("screen", "stopped working") == "VALID"
        
        # NOISE
        assert normalizer.judge_validity("product", "good") == "NOISE"
        assert normalizer.judge_validity("battery", "good") == "NOISE"
        
        # INVALID
        assert normalizer.judge_validity("", "drains fast") == "INVALID"
        assert normalizer.judge_validity("battery", "") == "INVALID"

