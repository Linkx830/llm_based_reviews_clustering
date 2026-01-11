# 新增或修改我时需要修改这个文件夹中的README.md文件
"""配置加载工具"""
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_prompt_template(prompt_type: str, version: str) -> str:
    """
    加载Prompt模板
    
    Args:
        prompt_type: 类型（extraction/insight/judge_recheck/report）
        version: 版本（如v1.0）
    
    Returns:
        Prompt模板内容
    """
    prompt_path = Path(f"configs/prompts/{prompt_type}/{version}.md")
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt模板不存在: {prompt_path}")
    
    content = prompt_path.read_text(encoding="utf-8")
    
    # 提取模板部分（跳过元数据）
    lines = content.split("\n")
    template_start = False
    template_lines = []
    
    for line in lines:
        if "## Prompt模板" in line or "```" in line:
            template_start = True
            continue
        if template_start and line.strip() and not line.startswith("#"):
            template_lines.append(line)
    
    if not template_lines:
        # 如果没有找到模板部分，返回整个文件内容
        return content
    
    return "\n".join(template_lines).strip()


def load_run_config(config_path: Path) -> Dict[str, Any]:
    """加载运行配置"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_taxonomy_files() -> Dict[str, Dict[str, str]]:
    """
    加载词表文件
    
    Returns:
        {
            "aspect_synonyms": {原始词: 归一词},
            "noise_terms": {词: "noise"},
            "aspect_allowlist": {词: "allowed"}
        }
    """
    taxonomy_dir = Path("configs/taxonomy")
    result = {
        "aspect_synonyms": {},
        "noise_terms": {},
        "aspect_allowlist": {}
    }
    
    # 加载同义词表
    synonyms_path = taxonomy_dir / "aspect_synonyms.csv"
    if synonyms_path.exists():
        import csv
        with open(synonyms_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                original = row.get("original", "").strip().lower()
                normalized = row.get("normalized", "").strip().lower()
                if original and normalized:
                    result["aspect_synonyms"][original] = normalized
    
    # 加载噪声词表
    noise_path = taxonomy_dir / "noise_terms.csv"
    if noise_path.exists():
        import csv
        with open(noise_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = row.get("term", "").strip().lower()
                if term:
                    result["noise_terms"][term] = "noise"
    
    # 加载允许列表
    allowlist_path = taxonomy_dir / "aspect_allowlist.csv"
    if allowlist_path.exists():
        import csv
        with open(allowlist_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = row.get("term", "").strip().lower()
                if term:
                    result["aspect_allowlist"][term] = "allowed"
    else:
        taxonomy_dir.mkdir(parents=True, exist_ok=True)
    
    return result

