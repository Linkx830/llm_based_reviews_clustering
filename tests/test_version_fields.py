# 新增或修改我时需要修改这个文件夹中的README.md文件
"""版本字段测试"""
import pytest
from src.storage.version_fields import VersionFields


class TestVersionFields:
    """版本字段测试类"""
    
    def test_generate_run_id(self):
        """测试run_id生成"""
        run_id = VersionFields.generate_run_id("category", "headphones")
        assert isinstance(run_id, str)
        assert "category" in run_id
        assert "headphones" in run_id
    
    def test_generate_data_slice_id(self):
        """测试data_slice_id生成"""
        # 测试不同参数组合
        slice_id1 = VersionFields.generate_data_slice_id(main_category="Appliances")
        slice_id2 = VersionFields.generate_data_slice_id(main_category="Appliances")
        
        # 相同参数应生成相同ID
        assert slice_id1 == slice_id2
        
        # 不同参数应生成不同ID
        slice_id3 = VersionFields.generate_data_slice_id(main_category="Electronics")
        assert slice_id1 != slice_id3
    
    def test_get_base_version_fields(self):
        """测试基础版本字段"""
        fields = VersionFields.get_base_version_fields(
            run_id="test_001",
            pipeline_version="v1.0",
            data_slice_id="slice_001"
        )
        
        assert fields["run_id"] == "test_001"
        assert fields["pipeline_version"] == "v1.0"
        assert fields["data_slice_id"] == "slice_001"
        assert "created_at" in fields

