import json
import hashlib
import os
from typing import Dict, Optional
from pathlib import Path

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM响应缓存文件
        self.llm_cache_file = self.cache_dir / "llm_responses.json"
        self.llm_cache = self._load_cache(self.llm_cache_file)
        
        # 提取结果缓存文件
        self.extraction_cache_file = self.cache_dir / "extraction_results.json"
        self.extraction_cache = self._load_cache(self.extraction_cache_file)
    
    def _load_cache(self, cache_file: Path) -> Dict:
        """加载缓存文件"""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载缓存文件失败 {cache_file}: {e}")
                return {}
        return {}
    
    def _save_cache(self, cache_file: Path, cache_data: Dict):
        """保存缓存文件"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存文件失败 {cache_file}: {e}")
    
    def compute_hash(self, *args) -> str:

        content = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_llm_response(self, prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[str]:

        cache_key = self.compute_hash(prompt, model, temperature, max_tokens)
        return self.llm_cache.get(cache_key)
    
    def save_llm_response(self, prompt: str, model: str, temperature: float, 
                          max_tokens: int, response: str):
        cache_key = self.compute_hash(prompt, model, temperature, max_tokens)
        self.llm_cache[cache_key] = response
        self._save_cache(self.llm_cache_file, self.llm_cache)
    
    def _convert_edges_for_json(self, edges: Dict) -> Dict:
        converted = {}
        for key, value in edges.items():
            if isinstance(key, tuple):
                # 将tuple转换为字符串："(src, tgt)"
                str_key = json.dumps(list(key), ensure_ascii=False)
            else:
                str_key = str(key)
            converted[str_key] = value
        return converted
    
    def _convert_edges_from_json(self, edges: Dict) -> Dict:
        converted = {}
        for key, value in edges.items():
            try:
                # 尝试将字符串解析为tuple
                key_list = json.loads(key)
                if isinstance(key_list, list) and len(key_list) == 2:
                    tuple_key = tuple(key_list)
                else:
                    tuple_key = key
            except:
                tuple_key = key
            converted[tuple_key] = value
        return converted
    
    def get_extraction_result(self, data_item_id: str) -> Optional[Dict]:
        cached = self.extraction_cache.get(data_item_id)
        if cached is None:
            return None
        # 转换edges的键从字符串恢复为tuple
        result = {
            "entities": cached.get("entities", {}),
            "edges": self._convert_edges_from_json(cached.get("edges", {}))
        }
        return result
    
    def save_extraction_result(self, data_item_id: str, entities: Dict, edges: Dict):
        # 转换edges的tuple键为字符串键以便JSON序列化
        converted_edges = self._convert_edges_for_json(edges)
        
        self.extraction_cache[data_item_id] = {
            "entities": entities,
            "edges": converted_edges
        }
        self._save_cache(self.extraction_cache_file, self.extraction_cache)
    
    def get_data_item_id(self, data_item: Dict) -> str:
        user_id = str(data_item.get("user_id", ""))
        timestamp = str(data_item.get("timestamp", ""))
        problem_name = str(data_item.get("problem_name", ""))
        question = str(data_item.get("question", ""))[:100]  # 只取前100个字符
        
        return self.compute_hash(user_id, timestamp, problem_name, question)
    
    def clear_cache(self):
        """清空所有缓存"""
        self.llm_cache = {}
        self.extraction_cache = {}
        if self.llm_cache_file.exists():
            self.llm_cache_file.unlink()
        if self.extraction_cache_file.exists():
            self.extraction_cache_file.unlink()
        print("缓存已清空")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "llm_cache_size": len(self.llm_cache),
            "extraction_cache_size": len(self.extraction_cache)
        }

