import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from collections import Counter

# 添加 student_sim 路径
STUDENT_SIM_PATH = "./student_sim-main"
if STUDENT_SIM_PATH not in sys.path:
    sys.path.insert(0, STUDENT_SIM_PATH)

try:
    from src.nano_graphrag import GraphRAG, QueryParam
    from src.nano_graphrag._llm import gpt_4o_complete, gpt_35_complete, openai_embedding
    from src.nano_graphrag._utils import wrap_embedding_func_with_attrs
    from src.utils import generate, read_json, write_json, write_jsonl
    from src.mapping import MODEL_FUNC_MAP, STR_TO_MODEL_NAME_MAP
except ImportError as e:
    print(f"错误: 无法导入 student_sim 模块: {e}")
    print(f"请确保 student_sim 路径正确: {STUDENT_SIM_PATH}")
    sys.exit(1)

# 导入 OpenAI 用于自定义 API 调用
from openai import OpenAI, AsyncOpenAI
import asyncio
import numpy as np
import httpx

STUDENT_IDS_20 = [
    "134015446", "135363688", "156242714", "190904875", "303035807", "451443795",
    "138766644", "172255240", "174144880", "190778770", "216337872", "235541946",
    "271819945", "291946738", "292852703", "314863369", "339292326", "380644542",
    "405568650", "427624027"
]

STUDENT_IDS_30 = [
    "57706049", "64904597", "69501469", "134015446", "135363688", "138766644", 
    "156242714", "172255240", "174144880", "190778770", "190904875", "216337872", 
    "235541946", "271819945", "291946738", "292852703", "303035807", "314863369", 
    "339292326", "380644542", "405568650", "427624027", "451443795", "463554746", 
    "466662879", "604128734", "626294389", "662511357", "679796740", "756999517"
]

STUDENT_IDS_40 = [
    "57706049", "64904597", "69501469", "86094393", "86151237", "134015446", 
    "135363688", "138766644", "156242714", "172255240", "174144880", "190778770", 
    "190904875", "216337872", "235541946", "271819945", "291946738", "292852703", 
    "303035807", "314863369", "339292326", "380644542", "405568650", "427624027", 
    "451443795", "463554746", "466662879", "604128734", "626294389", "662511357", 
    "679796740", "756999517", "767155168", "771346787", "783928287", "800795936", 
    "822141656", "831427858", "843247902", "915205165"
]

# 得分区间定义
SCORE_BINS = [
    (0, 20, "0-20"),
    (20, 40, "20-40"),
    (40, 60, "40-60"),
    (60, 80, "60-80"),
    (80, 100, "80-100"),
]


def get_student_ids(num_students: int) -> List[str]:
    """根据学生数量返回对应的学生ID列表"""
    if num_students == 20:
        return STUDENT_IDS_20
    elif num_students == 30:
        return STUDENT_IDS_30
    elif num_students == 40:
        return STUDENT_IDS_40
    else:
        raise ValueError(f"不支持的学生数量: {num_students}，只支持 20、30、40")


def create_custom_embedding_func(api_key: str, base_url: str, model: str = "conan", embedding_dim: int = 1792):

    import httpx
    
    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
    async def custom_embedding(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        
        try:
            payload = {"model": model, "input": texts}
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
            
            data = result.get("data", [])
            if not data:
                return np.zeros((0, embedding_dim), dtype=np.float32)
            
            vectors = [item["embedding"] for item in data]
            return np.asarray(vectors, dtype=np.float32)
        except Exception as e:
            print(f"Embedding 调用失败: {e}")
            raise
    
    return custom_embedding


def discretize_score(score: float) -> str:
    for min_val, max_val, label in SCORE_BINS:
        if min_val <= score < max_val:
            return label
    # 如果 score == 100，返回最后一个区间
    return SCORE_BINS[-1][2]


def generate_with_custom_api(request: Dict, llm_api_key: str = None, llm_base_url: str = None):
    if llm_api_key and llm_base_url:
        # 检查是否已有缓存
        if os.path.isfile(request["output_file"]):
            with open(request["output_file"]) as f:
                for line in f:
                    try:
                        cached = json.loads(line.strip())
                        if cached.get('id') == request["id"]:
                            return cached.get("output", {})
                    except:
                        continue
        
        client = OpenAI(api_key=llm_api_key, base_url=llm_base_url, timeout=180.0)
        
        # 重试机制：最多重试 3 次
        max_retries = 3
        retry_delay = 2  # 重试间隔（秒）
        
        for attempt in range(max_retries):
            try:
                # 构建请求参数
                create_kwargs = {
                    "model": request["model"],
                    "messages": request["messages"],
                }
                if "dashscope.aliyuncs.com" in llm_base_url.lower():
                    create_kwargs["extra_body"] = {
                        "enable_thinking": False
                    }
                
                stream = client.chat.completions.create(**create_kwargs).to_dict()
                
                output_data = {
                    "id": request["id"],
                    "output": stream,
                    "input": request
                }
                write_jsonl(file_path=request["output_file"], single_data=output_data)
                
                return stream
            except Exception as e:
                error_msg = str(e)
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                
                if attempt < max_retries - 1:
                    if is_timeout:
                        print(f"API 调用超时（尝试 {attempt + 1}/{max_retries}），{retry_delay}秒后重试...")
                    else:
                        print(f"API 调用失败（尝试 {attempt + 1}/{max_retries}）：{error_msg[:100]}，{retry_delay}秒后重试...")
                    import time
                    time.sleep(retry_delay)
                else:
                    if is_timeout:
                        print(f"API 调用超时（已重试 {max_retries} 次），可能是 prompt 太长或网络问题")
                    else:
                        print(f"API 调用失败（已重试 {max_retries} 次）：{error_msg[:200]}")
                    raise
    else:
        return generate(request)


def parse_problem_from_src(src_text: str) -> Dict[str, str]:
    problem_title = ""
    problem_statement = ""
    
    if "Problem title:" in src_text:
        parts = src_text.split("Problem title:")
        if len(parts) > 1:
            title_part = parts[1].split("Problem statement:")[0].strip()
            problem_title = title_part
    
    if "Problem statement:" in src_text:
        parts = src_text.split("Problem statement:")
        if len(parts) > 1:
            problem_statement = parts[1].strip()
    
    return {
        "problem_title": problem_title,
        "problem_statement": problem_statement,
        "full_text": src_text
    }


class StudentProfileRetrieval:    
    def __init__(self, student_id: str, model: str = "3.5", 
                 kg_base_dir: str = "./edu_kg_builder/baseline_entity_output",
                 working_dir: str = "retrieval_results",
                 train_data: List[Dict] = None,
                 test_problem_names: set = None,
                 validation_problem_names: set = None,
                 llm_api_key: str = None,
                 llm_base_url: str = None,
                 embedding_api_key: str = None,
                 embedding_base_url: str = None,
                 embedding_model: str = "conan",
                 embedding_dim: int = 1792):

        self.student_id = student_id
        self.model = model
        self.working_dir = Path(working_dir) / f"student_{student_id}"
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.kg_dir = Path(kg_base_dir) / f"graph_{student_id}_practice" / "model_General"
        
        if not self.kg_dir.exists():
            raise FileNotFoundError(f"知识图谱不存在: {self.kg_dir}")
        
        model_func = MODEL_FUNC_MAP.get(model, gpt_35_complete)
        
        embedding_func = None
        if embedding_api_key and embedding_base_url:
            embedding_func = create_custom_embedding_func(
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                model=embedding_model,
                embedding_dim=embedding_dim
            )
        else:
            if not os.getenv('OPENAI_API_KEY'):
                print(f"  警告: 未设置 OPENAI_API_KEY 且未提供自定义 embedding API，检索可能失败")
        
        self.graph_func = GraphRAG(
            working_dir=str(self.kg_dir),
            best_model_func=model_func,
            cheap_model_func=model_func,
            embedding_func=embedding_func if embedding_func else openai_embedding
        )
        
        self.train_data = train_data or []
        self.test_problem_names = test_problem_names or set()
        self.validation_problem_names = validation_problem_names or set()
        
        if self.train_data:
            train_timestamps = [item.get('timestamp', '') for item in self.train_data if item.get('timestamp')]
            if train_timestamps:
                train_timestamps.sort()
                self.train_end_time = train_timestamps[-1]  
            else:
                self.train_end_time = None
        else:
            self.train_end_time = None
        
        self.summarize_kc = {}
        self.retrieval_cache = {}
        
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url

        if self.llm_api_key and self.llm_base_url:
            print(f"  使用自定义 LLM API: {self.llm_base_url}")
        
        kg_files = list(self.kg_dir.glob("*.json"))
        vdb_files = list(self.kg_dir.glob("vdb*.json"))
        vdb_entities_file = self.kg_dir / "vdb_entities.json"
        print(f"  知识图谱文件: {len(kg_files)} 个JSON文件, {len(vdb_files)} 个向量数据库文件")
        if len(vdb_files) == 0:
            print(f"\n   严重警告: 未找到向量数据库文件（vdb_entities.json）！")
            print(f"  这会导致检索功能完全失效（100%检索失败）。")
            print(f"  检查路径: {self.kg_dir}")
            # 列出所有JSON文件，帮助调试
            json_files = [f.name for f in self.kg_dir.glob("*.json")]
            print(f"  现有JSON文件: {json_files}")
            print(f"\n  解决方案：")
            print(f"  1. 重新构建知识图谱，确保调用 graph._index_done() 或 graph.entities_vdb.index_done_callback()")
            print(f"  2. 或者运行修复脚本，重新保存向量数据库")
            print(f"  3. 或者使用 --disable_filter 参数，但这会引入数据泄露风险\n")
    
    def _is_from_train_data(self, text_content: str) -> bool:

        if not self.test_problem_names and not self.validation_problem_names:
            return True
        
        text_lower = text_content.lower()
        
        for problem_name in self.test_problem_names:
            if problem_name:
                problem_lower = problem_name.lower()
                if problem_lower in text_lower:
                    return False
        
        for problem_name in self.validation_problem_names:
            if problem_name:
                problem_lower = problem_name.lower()
                if problem_lower in text_lower:
                    return False
        
        if self.train_end_time:
            import re
            timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
            timestamps = re.findall(timestamp_pattern, text_content)
            for ts in timestamps:
                if ts > self.train_end_time:
                    return False
        
        return True
    
    def retrieve_similar_cases(self, query_text: str, top_k: int = 5, filter_test_data: bool = True) -> List[Dict]:
        cache_key = f"{hash(query_text)}_{top_k}_{filter_test_data}"
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        try:
            retrieve_k = top_k * 10 if filter_test_data else top_k
            
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
            
            result = self.graph_func.query(
                query_text,
                param=QueryParam(mode="local", only_need_context=True, top_k=retrieve_k)
            )
            
            if not hasattr(self, '_debug_result_printed'):
                self._debug_result_printed = True
            
            similar_cases = []
            raw_count = 0  # 原始检索到的数量
            filtered_count = 0  # 被过滤掉的数量
            
            if result:
                if "text_units_section_list" in result and len(result["text_units_section_list"]) > 1:
                    for text_unit in result["text_units_section_list"][1:]:
                        if len(text_unit) >= 2:
                            content = text_unit[1] if len(text_unit) > 1 else ""
                            raw_count += 1
                            
                            if filter_test_data:
                                if not self._is_from_train_data(content):
                                    filtered_count += 1
                                    continue
                            
                            similar_cases.append({
                                "type": "text_unit",
                                "content": content,
                                "similarity": 0.0  
                            })
                            
                            if len(similar_cases) >= top_k:
                                break
                
                if len(similar_cases) < top_k and "results" in result:
                    for item in result["results"]:
                        description = item.get("description", "")
                        raw_count += 1
                        
                        if filter_test_data:
                            if not self._is_from_train_data(description):
                                filtered_count += 1
                                continue
                        
                        similar_cases.append({
                            "type": "entity",
                            "entity_name": item.get("entity_name", ""),
                            "description": description,
                            "similarity": item.get("similarity", 0.0)
                        })
                        
                        if len(similar_cases) >= top_k:
                            break
            
            if len(similar_cases) == 0:
                if raw_count > 0:
                    if not hasattr(self, '_warned_no_results'):
                        self._warned_no_results = True

                        if result and "text_units_section_list" in result and len(result["text_units_section_list"]) > 1:
                            sample_content = result["text_units_section_list"][1][1] if len(result["text_units_section_list"][1]) > 1 else ""
                            print(f"  示例内容（前200字符）: {sample_content[:200]}")
                else:
                    if not hasattr(self, '_warned_no_raw_results'):
                        self._warned_no_raw_results = True

            
            self.retrieval_cache[cache_key] = similar_cases
            return similar_cases
            
        except Exception as e:
            print(f"检索失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def predict_score_range(self, test_item: Dict) -> Dict:

        problem_info = parse_problem_from_src(test_item.get('src', ''))
        current_knowledge_graph = test_item.get('e', '')  
        query_text = f"""Question:
{problem_info['problem_title']}

{problem_info['problem_statement']}
"""

        top_k = getattr(self, 'retrieval_top_k', 5)
        filter_test_data = not getattr(self, 'disable_filter', False)
        similar_cases = self.retrieve_similar_cases(query_text, top_k=top_k, filter_test_data=filter_test_data)
        
        # 构建 prompt 用于预测
        similar_cases_text = ""
        if similar_cases:
            similar_cases_text = "\n检索到的相似历史案例（这些案例展示了学生在类似问题上的历史表现，可以作为参考）：\n"
            for i, case in enumerate(similar_cases, 1):
                similar_cases_text += f"\n相似案例 {i}:\n"
                if case.get('type') == 'text_unit':
                    content = case.get('content', '')
                    if 'Score:' in content or 'score' in content.lower() or 'label' in content.lower():
                        similar_cases_text += f"历史记录（包含得分信息）: {content[:1000]}...\n"
                    else:
                        similar_cases_text += f"历史记录: {content[:600]}...\n"
                    similar_cases_text += "（注意：如果历史记录中包含得分信息，可以参考学生在相似问题上的实际表现）\n"
                else:
                    similar_cases_text += f"知识点: {case.get('entity_name', '')}\n"
                    similar_cases_text += f"描述: {case.get('description', '')[:300]}...\n"
                    if case.get('similarity', 0) > 0:
                        similar_cases_text += f"相似度: {case.get('similarity', 0.0):.3f}\n"
            similar_cases_text += "\n重要提示：检索到的案例主要提供相似问题的内容和知识图谱信息，但通常不包含学生在问题上的实际得分。请主要参考知识图谱中的知识点掌握程度（score）进行判断。"
        else:
            similar_cases_text = "\n注意：未找到相似的历史案例（可能因为过滤条件或知识图谱数据不足）。请主要基于学生的当前知识图谱进行预测。"
        
        if similar_cases:
            retrieval_context = f"""
检索到的相似历史案例（来自训练数据）：
{similar_cases_text}

请结合学生的知识图谱和相似历史案例，预测学生在该问题上的得分区间。"""
        else:
            retrieval_context = f"""
注意：未检索到相似的历史案例，请主要基于学生的知识图谱进行预测。
{similar_cases_text}"""
        
        # Few-shot
        few_shot_examples = """以下是几个示例，展示如何从知识图谱推断得分区间。**特别注意：不要过于保守，要根据知识图谱的实际信息做出判断，包括极端区间（0-20和80-100）！**

示例1（0-20区间 - 极端低分）：
问题：字符串处理问题，需要理解字符串操作和模式匹配
知识图谱：
Strengths: "GREEDY" (score 0.95), "CONTROL_STRUCTURE" (score 0.95)
Needs Improvement: "STRING_OPERATION" (score 0.35), "PATTERN_MATCHING" (score 0.35), "CORRECT_LOGIC" (score 0.35)
分析：**关键判断**：虽然学生掌握了一些基础知识点，但解决该问题所需的所有关键知识点（STRING_OPERATION、PATTERN_MATCHING）都在"Needs Improvement"中，且score非常低（0.35）。这说明学生完全无法解决这类问题，应该预测0-20，而不是20-40或40-60。
预测：0-20

示例2（20-40区间）：
问题：动态规划问题
知识图谱：
Strengths: "GREEDY" (score 0.95), "ENUMERATION_AND_SORTING" (score 0.95)
Needs Improvement: "DYNAMIC_PROGRAMMING" (score 0.35), "PROBLEM_REQUIREMENTS" (score 0.35)
分析：学生掌握了一些算法基础，但动态规划（该问题的核心）在"Needs Improvement"中且score很低，说明学生可能只能解决部分简单部分，但会有很多错误。
预测：20-40

示例3（40-60区间）：
问题：图论问题
知识图谱：
Strengths: "SEARCH" (score 0.95), "GRAPH_STRUCTURE" (score 0.61)
Needs Improvement: "GRAPH_STRUCTURE" (score 0.49), "UNDERSTANDING" (score 0.35)
分析：学生掌握了一些图论基础知识（GRAPH_STRUCTURE score 0.61），但掌握程度一般（不是0.95），且"UNDERSTANDING"在Needs Improvement中，说明学生能解决部分问题但不够完整。
预测：40-60

示例4（60-80区间）：
问题：动态规划问题
知识图谱：
Strengths: "DYNAMIC_PROGRAMMING" (score 0.95), "GREEDY" (score 0.95), "ENUMERATION_AND_SORTING" (score 0.95)
Needs Improvement: "PROBLEM_REQUIREMENTS" (score 0.35)
分析：学生掌握了解决该问题所需的核心知识点（DYNAMIC_PROGRAMMING score 0.95），说明学生掌握较好，能解决大部分问题。
预测：60-80

示例5（80-100区间 - 极端高分）：
问题：数组操作问题（easy难度）
知识图谱：
Strengths: "ARRAY_OPERATION" (score 0.95), "GREEDY" (score 0.95), "CONTROL_STRUCTURE" (score 0.95), "ENUMERATION_AND_SORTING" (score 0.95), "DYNAMIC_PROGRAMMING" (score 0.95)
Needs Improvement: "BAD: DYNAMIC_PROGRAMMING" (score 0.35), "BAD: STRING_ALGORITHMS" (score 0.35), "SEGMENT_TREE" (score 0.35)
分析：**关键判断**：
1. 解决该问题所需的关键知识点（ARRAY_OPERATION、GREEDY、CONTROL_STRUCTURE、DYNAMIC_PROGRAMMING）都在"Strengths"中且score非常高（0.95）
2. 虽然"Needs Improvement"中有"BAD: DYNAMIC_PROGRAMMING"，但Strengths中已经有"DYNAMIC_PROGRAMMING" (score 0.95)，说明学生掌握很好
3. **重要原则**：如果关键知识点在Strengths中且score≥0.95，即使Needs Improvement中有类似的知识点（即使是"BAD"开头的），也应该以Strengths为准，预测高分
4. "Needs Improvement"中的"BAD: STRING_ALGORITHMS"和"SEGMENT_TREE"与当前问题无关，应该忽略
5. 因此应该预测80-100，而不是0-20或40-60
预测：80-100

**关键判断规则（请严格按照以下规则判断，要平衡判断，不要过度偏向任何一方！）**：

**首先，识别问题的关键知识点（非常重要！）**：
- **关键知识点**：解决该问题必须掌握的核心知识点（如动态规划问题的"DYNAMIC_PROGRAMMING"，数组问题的"ARRAY_OPERATION"，图论问题的"GRAPH_STRUCTURE"等）
- **次要知识点**：有助于解决问题但不是必须的知识点（如"PROBLEM_REQUIREMENTS"、"UNDERSTANDING"等通用知识点）
- **无关知识点**：与当前问题完全无关的知识点（如"NOWCODER"、"STUDENT'S CODE"、"998244353"、"BINARY TREE POSITIONS"、"NEW STUDENT COMPETITION"、"24-HOUR TIME FORMAT"等通用实体）→ **必须完全忽略！这些知识点不应该影响判断！**

**然后，按照以下规则判断（只关注与问题相关的关键知识点）**：

1. **80-100区间**：
   - **条件**：问题所需的**所有关键知识点**（只考虑与问题相关的）都在"Strengths"中且score≥0.95
   - **判断规则**：
     * 如果关键知识点在"Strengths"中且score≥0.95 → **以Strengths为准，预测高分**
     * 即使"Needs Improvement"中有类似的知识点（如"BAD: DYNAMIC_PROGRAMMING"），只要Strengths中有对应的知识点且score≥0.95，就应该以Strengths为准
     * **重要原则**：Strengths的优先级高于Needs Improvement！如果同一个知识点在Strengths中score≥0.95，即使Needs Improvement中有"BAD"开头的类似知识点，也应该以Strengths为准
   - **判断**：如果满足以上条件 → **必须预测 80-100**（不要预测60-80或0-20！）
   - **重要**：即使"Needs Improvement"中有很多知识点，只要关键知识点在Strengths中且score≥0.95，就应该预测80-100！

2. **60-80区间**：
   - **条件**：问题所需的核心知识点在"Strengths"中且score≥0.95，但有一些次要知识点在"Needs Improvement"
   - **判断**：预测 60-80

3. **40-60区间**：
   - **条件**：问题所需的知识点部分在"Strengths"（score中等如0.5-0.7），部分在"Needs Improvement"
   - **判断**：预测 40-60

4. **20-40区间**：
   - **条件**：问题所需的**核心知识点**在"Needs Improvement"中且score≤0.35，即使Strengths中有一些基础知识点
   - **判断**：预测 20-40

5. **0-20区间**：
   - **条件**：知识图谱中**没有与问题相关的关键知识点**，或**所有与问题相关的关键知识点都在"Needs Improvement"且score很低（≤0.35）**
   - **判断**：如果满足以上条件 → **必须预测 0-20**（不要预测20-40或40-60！）
   - **重要**：即使Strengths中有很多知识点，只要它们与当前问题无关，就应该忽略，预测0-20！

**最重要提醒**：
- **只关注与问题相关的关键知识点**：忽略所有无关知识点（如"NOWCODER"、"STUDENT'S CODE"等）
- **平衡判断**：不要过度偏向低分或高分，要根据**与问题相关的关键知识点**的实际分布做出判断
- **80-100区间**：如果所有与问题相关的关键知识点都在Strengths且score≥0.95，即使Needs Improvement中有很多无关知识点，也要预测80-100！
- **0-20区间**：如果所有与问题相关的关键知识点都在Needs Improvement且score≤0.35，即使Strengths中有很多无关知识点，也要预测0-20！
"""

        prediction_prompt = f"""你是一个AI导师。基于学生的知识图谱和问题，预测学生在该问题上的得分区间。

**重要：不要过于保守！要根据知识图谱的实际信息做出判断，包括极端区间（0-20和80-100）！**

得分区间定义（请严格按照以下定义）：
- **0-20**: 得分在 0 到 20 之间（不包括20），表示学生几乎不会做，知识掌握很差，无法解决该问题。**如果核心知识点都在"Needs Improvement"且score≤0.35，必须预测0-20！**
- 20-40: 得分在 20 到 40 之间（不包括40），表示学生掌握较差，有很多错误，只能解决部分简单部分
- 40-60: 得分在 40 到 60 之间（不包括60），表示学生掌握一般，部分正确，能解决部分问题但不够完整
- 60-80: 得分在 60 到 80 之间（不包括80），表示学生掌握较好，大部分正确，能解决大部分问题
- **80-100**: 得分在 80 到 100 之间（包括100），表示学生掌握很好，几乎全对，能完整解决该问题。**如果所有关键知识点都在"Strengths"且score≥0.95，且"Needs Improvement"中没有相关知识点，必须预测80-100！**

{few_shot_examples}

当前问题：
{problem_info['problem_title']}

{problem_info['problem_statement']}

学生当前知识图谱：
{current_knowledge_graph}
{retrieval_context}

请按照以下步骤仔细分析：

步骤1：识别问题类型和所需知识点（**关键步骤**）
- 分析问题属于什么类型（数组操作、动态规划、图论、字符串、数学等）
- **确定解决该问题需要哪些关键知识点**（只关注与问题直接相关的知识点）
- **重要**：忽略与问题无关的通用知识点（如"NOWCODER"、"STUDENT'S CODE"、"998244353"、"BINARY TREE POSITIONS"、"NEW STUDENT COMPETITION"等通用实体）
- **只关注**：与问题类型和解决思路直接相关的知识点（如动态规划问题的"DYNAMIC_PROGRAMMING"，数组问题的"ARRAY_OPERATION"等）

步骤2：分析知识图谱（**关键步骤，请仔细分析**）
- **首先，识别问题的关键知识点**：
  * **关键知识点**：解决该问题必须掌握的核心知识点（如动态规划问题的"DYNAMIC_PROGRAMMING"，数组问题的"ARRAY_OPERATION"，图论问题的"GRAPH_STRUCTURE"等）
  * **次要知识点**：有助于解决问题但不是必须的知识点（如"PROBLEM_REQUIREMENTS"、"UNDERSTANDING"等通用知识点）
  * **无关知识点**：与当前问题完全无关的知识点（如"NOWCODER"、"STUDENT'S CODE"、"998244353"、"BINARY TREE POSITIONS"、"NEW STUDENT COMPETITION"等通用实体）→ **必须忽略！**
- **检查知识图谱中"Strengths"（优势）部分**：
  * **只关注**与问题相关的关键知识点
  * 如果包含解决该问题所需的**关键知识点**，且score较高（≥0.95），说明学生掌握较好
  * **如果所有关键知识点都在"Strengths"且score≥0.95 → 必须考虑预测80-100！**
  * **重要原则**：Strengths的优先级高于Needs Improvement！如果关键知识点在Strengths中且score≥0.95，即使Needs Improvement中有类似的知识点（如"BAD: DYNAMIC_PROGRAMMING"），也应该以Strengths为准，预测高分
- **检查知识图谱中"Needs Improvement"（需要改进）部分**：
  * **只关注**与问题相关的关键知识点
  * **重要**：如果关键知识点在Strengths中且score≥0.95，即使Needs Improvement中有类似的知识点，也应该以Strengths为准，忽略Needs Improvement中的对应知识点
  * 只有当关键知识点**不在Strengths中**或**Strengths中score较低**时，才考虑Needs Improvement中的知识点
  * 如果包含解决该问题所需的**关键知识点**，且score较低（≤0.35），且**Strengths中没有对应的知识点或score较低**，说明学生掌握较差
  * **如果所有关键知识点都在"Needs Improvement"且score≤0.35，且Strengths中没有对应的知识点 → 必须考虑预测0-20！**
  * **重要**：如果"Needs Improvement"中的知识点与当前问题无关，必须忽略，不影响判断！
- 如果知识图谱中没有与问题相关的关键知识点，说明学生可能不会做（倾向于0-20或20-40）
- **平衡判断**：不要过度偏向低分或高分，要根据**与问题相关的关键知识点**的实际分布做出判断！

步骤3：分析相似历史案例（如果提供）
- **检索数据库的作用**：提供相似问题的内容和知识图谱信息
- **注意**：检索到的案例通常不包含学生在问题上的实际得分（label），主要包含：
  * 相似问题的内容描述
  * 知识图谱信息（知识点和掌握程度的 score）
- **如何使用检索结果**：
  * 参考相似问题的类型和所需知识点
  * 如果检索到的问题与当前问题非常相似，可以参考知识图谱中的知识点掌握程度
  * **但主要判断依据仍然是当前问题的知识图谱，而不是检索结果**

步骤4：综合判断（参考上面的关键判断规则）
- **主要判断依据**：严格按照知识图谱中的score和知识点位置进行判断
- **关键原则**：区分关键知识点和次要知识点，关键知识点决定判断！
- **检索结果的作用**：提供相似问题的参考，帮助理解问题类型和所需知识点
- **平衡判断**：不要过度偏向低分或高分，要根据关键知识点的实际分布做出判断！
- **极端区间判断**：
  * **0-20区间**：如果所有关键知识点都在"Needs Improvement"且score≤0.35 → **必须预测0-20**（不要预测20-40或40-60！）
  * **80-100区间**：如果所有关键知识点都在"Strengths"且score≥0.95 → **必须预测80-100**（不要预测60-80或0-20！即使Needs Improvement中有次要知识点也可以忽略！）
- **Few-shot示例 vs 检索数据库**：
  * Few-shot示例：提供抽象的规则和模式（如何理解知识图谱、如何判断）
  * 检索数据库：提供相似问题的内容和知识图谱信息（但不包含实际得分）
  * **两者结合使用**：Few-shot 提供判断规则，检索提供问题相似度参考，但最终判断主要依据知识图谱

**重要：只输出区间标签，格式必须严格为以下之一：0-20、20-40、40-60、60-80、80-100**
**不要输出其他任何文字，只输出区间标签**
**再次提醒（非常重要！）**：
- **只关注与问题相关的关键知识点**：完全忽略无关知识点（如"NOWCODER"、"STUDENT'S CODE"、"998244353"、"BINARY TREE POSITIONS"等）
- **平衡判断**：不要过度偏向低分或高分，要根据**与问题相关的关键知识点**的实际分布做出判断
- **80-100区间**：如果所有与问题相关的关键知识点都在Strengths且score≥0.95，必须预测80-100（即使Needs Improvement中有类似的知识点如"BAD: XXX"，也要以Strengths为准！Strengths的优先级高于Needs Improvement！）
- **0-20区间**：如果所有与问题相关的关键知识点都在Needs Improvement且score≤0.35，必须预测0-20（即使Strengths中有很多无关知识点也要忽略！）

预测结果："""
        
        try:
            if self.llm_api_key and self.llm_base_url:
                model_name = self.model
            else:
                model_name = STR_TO_MODEL_NAME_MAP.get(self.model, "gpt-3.5-turbo")
            
            request = {
                "id": f"{self.student_id}_{test_item.get('problem_name', 'unknown')}",
                "model": model_name,
                "messages": [{"role": "user", "content": prediction_prompt}],
                "output_file": str(self.working_dir / "predictions.jsonl"),
            }
            
            cache_file = Path(request["output_file"])
            from_cache = False
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                cached = json.loads(line.strip())
                                if cached.get('id') == request["id"]:
                                    from_cache = True
                                    break
                            except:
                                continue
                except:
                    pass
            
            import time
            from openai import APITimeoutError
            start_time = time.time()
            try:
                output = generate_with_custom_api(
                    request, 
                    llm_api_key=self.llm_api_key,
                    llm_base_url=self.llm_base_url
                )
                elapsed = time.time() - start_time
            except (APITimeoutError, Exception) as e:
                elapsed = time.time() - start_time
                error_msg = str(e)
                is_timeout = isinstance(e, APITimeoutError) or "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                
                if is_timeout:
                    print(f" LLM调用超时（耗时 {elapsed:.1f}秒），使用默认预测值")
                    return {
                        "student_id": self.student_id,
                        "problem_name": test_item.get('problem_name', ''),
                        "predicted_range": "40-60",  # 默认值
                        "true_range": discretize_score(test_item.get('label', 0.0)),
                        "true_score": test_item.get('label', 0.0),
                        "error": "timeout",
                        "timeout": True,
                        "elapsed_time": elapsed
                    }
                else:
                    raise
            
            if not hasattr(self, '_debug_llm_printed'):
                self._debug_llm_printed = True
            
            predicted_range = output['choices'][0]['message']['content'].strip()
            
            original_prediction = predicted_range
            predicted_range = predicted_range.replace("预测结果：", "").strip()
            used_default = False
            
            if predicted_range not in [bin[2] for bin in SCORE_BINS]:
                for bin_label in [bin[2] for bin in SCORE_BINS]:
                    if bin_label in predicted_range:
                        predicted_range = bin_label
                        break
                else:
                    predicted_range = "40-60"
                    used_default = True
            
            true_label = test_item.get('label', 0.0)
            true_range = discretize_score(true_label)
            
            return {
                "student_id": self.student_id,
                "problem_name": test_item.get('problem_name', ''),
                "predicted_range": predicted_range,
                "true_range": true_range,
                "true_score": true_label,
                "prediction_text": predicted_range,
                "original_prediction": original_prediction if 'original_prediction' in locals() else predicted_range,
                "used_default": used_default if 'used_default' in locals() else False,
                "similar_cases_count": len(similar_cases),
                "timeout": False,
                "elapsed_time": elapsed if 'elapsed' in locals() else 0
            }
            
        except Exception as e:
            error_msg = str(e)
            is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
            
            if is_timeout:
                print(f"预测超时: {error_msg[:100]}")
            else:
                print(f"预测失败: {error_msg[:200]}")
                import traceback
                traceback.print_exc()
            
            return {
                "student_id": self.student_id,
                "problem_name": test_item.get('problem_name', ''),
                "predicted_range": "40-60",  # 默认值
                "true_range": discretize_score(test_item.get('label', 0.0)),
                "true_score": test_item.get('label', 0.0),
                "error": error_msg[:200],
                "timeout": is_timeout
            }


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败: {e}, 行内容: {line[:100]}")
    return data


def split_train_test(data: List[Dict], student_ids: List[str], 
                     train_ratio: float = 0.8) -> Dict[str, Dict]:
    result = {}
    
    for student_id in student_ids:
        student_data = [item for item in data if item.get('user_id') == student_id]
        
        if not student_data:
            print(f"警告: 学生 {student_id} 没有数据")
            result[student_id] = {"train": [], "test": []}
            continue
        
        student_data.sort(key=lambda x: x.get('timestamp', ''))
        
        split_idx = int(len(student_data) * train_ratio)
        train_data = student_data[:split_idx]
        test_data = student_data[split_idx:]
        
        result[student_id] = {
            "train": train_data,
            "test": test_data
        }
        
        print(f"学生 {student_id}: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='基于向量检索的得分区间预测（支持20/30/40学生）')
    parser.add_argument('--num_students', type=int, default=20, choices=[20, 30, 40],
                        help='学生数量（默认：20）')
    parser.add_argument('--input_file', type=str, required=True,
                        help='训练数据 JSONL 文件路径（推荐使用 training set/train_processed_XX_new_no_embedding.jsonl）')
    parser.add_argument('--output_dir', type=str, default='retrieval_results',
                        help='输出目录（默认：retrieval_results）')
    parser.add_argument('--model', type=str, default='3.5',
                        choices=['3.5', '4o', 'claude', 'llama'],
                        help='使用的模型')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认0.8）')
    parser.add_argument('--kg_base_dir', type=str, 
                        default='./edu_kg_builder/baseline_entity_output',
                        help='知识图谱基础目录（默认使用已构建的知识图谱）')
    parser.add_argument('--test_file', type=str, default=None,
                        help='测试集文件路径（如果提供，将使用此文件而不是从输入文件分割）')
    parser.add_argument('--validation_file', type=str, default=None,
                        help='验证集文件路径（用于提取问题名称，过滤检索结果）')
    parser.add_argument('--llm_model', type=str, default=None,
                        help='自定义 LLM 模型名称（如 General）')
    parser.add_argument('--llm_api_key', type=str, default=None,
                        help='自定义 LLM API key')
    parser.add_argument('--llm_base_url', type=str, default=None,
                        help='自定义 LLM API base URL')
    parser.add_argument('--embedding_api_key', type=str, default=None,
                        help='自定义 Embedding API key（用于向量检索）')
    parser.add_argument('--embedding_base_url', type=str, default=None,
                        help='自定义 Embedding API base URL（用于向量检索）')
    parser.add_argument('--embedding_model', type=str, default='conan',
                        help='Embedding 模型名称（默认 conan，需与构建知识图谱时一致）')
    parser.add_argument('--embedding_dim', type=int, default=1792,
                        help='Embedding 向量维度（默认 1792，需与构建知识图谱时一致）')
    parser.add_argument('--disable_filter', action='store_true',
                        help='禁用测试集/验证集过滤（用于调试，会引入数据泄露）')
    parser.add_argument('--retrieval_top_k', type=int, default=5,
                        help='检索返回的相似案例数量（默认5）')
    
    args = parser.parse_args()
    
    STUDENT_IDS = get_student_ids(args.num_students)    
    all_data = load_jsonl(args.input_file)
    
    student_train_data = {}
    for item in all_data:
        student_id = item.get('user_id')
        if student_id in STUDENT_IDS:
            if student_id not in student_train_data:
                student_train_data[student_id] = []
            student_train_data[student_id].append(item)
    
    test_problem_names = set()
    validation_problem_names = set()
    
    if args.validation_file:
        validation_data = load_jsonl(args.validation_file)
        for item in validation_data:
            problem_name = item.get('problem_name', '')
            if problem_name:
                validation_problem_names.add(problem_name)
    
    if args.test_file:
        test_data = load_jsonl(args.test_file)
        for item in test_data:
            problem_name = item.get('problem_name', '')
            if problem_name:
                test_problem_names.add(problem_name)
    else:

        student_all_data = {}
        for item in all_data:
            student_id = item.get('user_id')
            if student_id in STUDENT_IDS:
                if student_id not in student_all_data:
                    student_all_data[student_id] = []
                student_all_data[student_id].append(item)
        
        test_data = []
        for student_id in STUDENT_IDS:
            if student_id in student_all_data:
                student_data = sorted(student_all_data[student_id], key=lambda x: x.get('timestamp', ''))
                split_idx = int(len(student_data) * args.train_ratio)
                test_data.extend(student_data[split_idx:])
    
    if not args.test_file and args.validation_file:
        test_data = load_jsonl(args.validation_file)
        for item in test_data:
            problem_name = item.get('problem_name', '')
            if problem_name:
                validation_problem_names.add(problem_name)
    
    student_test_data = {}
    for item in test_data:
        student_id = item.get('user_id')
        if student_id in STUDENT_IDS:
            if student_id not in student_test_data:
                student_test_data[student_id] = []
            student_test_data[student_id].append(item)
    
    student_data_dict = {}
    for student_id in STUDENT_IDS:
        test_data_stu = student_test_data.get(student_id, [])
        student_data_dict[student_id] = {
            "test": test_data_stu
        }
        if test_data_stu:
            print(f"学生 {student_id}: 待预测数据 {len(test_data_stu)} 条")
    
    all_predictions = []
    
    for student_id in STUDENT_IDS:
        if student_id not in student_data_dict:
            continue
        
        test_data = student_data_dict[student_id].get("test", [])
        
        if not test_data:
            print(f"跳过学生 {student_id}: 没有测试数据")
            continue
        
        
        # 创建学生画像（使用已有的知识图谱）
        try:
            student_profile = StudentProfileRetrieval(
                student_id=student_id,
                model=args.llm_model if args.llm_model else args.model,  
                kg_base_dir=args.kg_base_dir,
                working_dir=args.output_dir,
                train_data=student_train_data.get(student_id, []),
                test_problem_names=test_problem_names if not args.disable_filter else [],
                validation_problem_names=validation_problem_names if not args.disable_filter else [],
                llm_api_key=args.llm_api_key,
                llm_base_url=args.llm_base_url,
                embedding_api_key=args.embedding_api_key,
                embedding_base_url=args.embedding_base_url,
                embedding_model=args.embedding_model,
                embedding_dim=args.embedding_dim
            )
            student_profile.retrieval_top_k = args.retrieval_top_k
            student_profile.disable_filter = args.disable_filter
        except FileNotFoundError as e:
            print(f"✗ 跳过学生 {student_id}: {e}")
            continue
        
        for test_item in test_data:
            prediction = student_profile.predict_score_range(test_item)
            all_predictions.append(prediction)
            
            if len(all_predictions) % 10 == 0:
                print(f"已预测 {len(all_predictions)} 条...")
    
    output_file = Path(args.output_dir) / "all_predictions.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    timeout_count = sum(1 for p in all_predictions if p.get('timeout', False))
    error_count = sum(1 for p in all_predictions if p.get('error') and not p.get('timeout', False))
    
    valid_predictions = [p for p in all_predictions if not p.get('timeout', False) and not p.get('error')]
    total = len(all_predictions)
    valid_total = len(valid_predictions)
    
    correct = sum(1 for p in valid_predictions if p.get('predicted_range') == p.get('true_range'))
    accuracy = correct / valid_total if valid_total > 0 else 0
    
    for bin_label in [bin[2] for bin in SCORE_BINS]:
        bin_predictions = [p for p in valid_predictions if p.get('true_range') == bin_label]
        if bin_predictions:
            bin_correct = sum(1 for p in bin_predictions if p.get('predicted_range') == p.get('true_range'))
            bin_acc = bin_correct / len(bin_predictions) if bin_predictions else 0
            print(f"  {bin_label}: {bin_correct}/{len(bin_predictions)} = {bin_acc:.4f}")
    
    CLASS_TO_VALUE = {
        "0-20": 10,
        "20-40": 30,
        "40-60": 50,
        "60-80": 70,
        "80-100": 90
    }
    
    SCORE_RANGE_TO_CLASS = {
        "0-20": 0,
        "20-40": 1,
        "40-60": 2,
        "60-80": 3,
        "80-100": 4
    }
    
    CLASS_TO_SCORE_RANGE = {v: k for k, v in SCORE_RANGE_TO_CLASS.items()}
    class_names = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    
    all_predicted_classes = []
    all_ground_truth_classes = []
    all_predicted_ranges = []
    all_ground_truth_ranges = []
    
    for pred in valid_predictions:  
        predicted_range = pred.get('predicted_range', '')
        true_range = pred.get('true_range', '')
        
        if predicted_range in SCORE_RANGE_TO_CLASS and true_range in SCORE_RANGE_TO_CLASS:
            all_predicted_classes.append(SCORE_RANGE_TO_CLASS[predicted_range])
            all_ground_truth_classes.append(SCORE_RANGE_TO_CLASS[true_range])
            all_predicted_ranges.append(predicted_range)
            all_ground_truth_ranges.append(true_range)
    
    if len(all_predicted_classes) == 0:
        print("\n警告: 没有有效的预测结果用于详细统计！")
        return
    
    y_true = np.array(all_ground_truth_classes)
    y_pred = np.array(all_predicted_classes)
    
    # ========== 1. 混淆矩阵 ==========
    print(f"\n{'='*60}")
    print("混淆矩阵 (5x5):")
    print(f"{'='*60}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    # 打印混淆矩阵（带标签）
    print("\n预测类别 ->")
    print(f"{'真实类别':<12} {'0-20':<8} {'20-40':<8} {'40-60':<8} {'60-80':<8} {'80-100':<8}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:<12}"
        for j in range(5):
            row_str += f"{cm[i, j]:<8}"
        print(row_str)
    
    # 打印每行的总数
    print("-" * 60)
    row_sums = cm.sum(axis=1)
    print(f"{'总计':<12}", end="")
    for sum_val in row_sums:
        print(f"{sum_val:<8}", end="")
    print()
    
    # ========== 2. MAE (平均绝对误差) ==========
    print(f"\n{'='*60}")
    print("MAE (平均绝对误差):")
    print(f"{'='*60}")
    
    # 将类别转换为数值（使用区间中点）
    y_true_values = np.array([CLASS_TO_VALUE[all_ground_truth_ranges[i]] for i in range(len(all_ground_truth_ranges))])
    y_pred_values = np.array([CLASS_TO_VALUE[all_predicted_ranges[i]] for i in range(len(all_predicted_ranges))])
    
    mae = np.mean(np.abs(y_true_values - y_pred_values))
    print(f"MAE: {mae:.2f} 分")
    print(f"说明: 平均预测误差为 {mae:.2f} 分（基于区间中点：10, 30, 50, 70, 90）")
    
    # ========== 3. Precision, Recall, F1分数 ==========
    print(f"\n{'='*60}")
    print("Precision, Recall, F1分数:")
    print(f"{'='*60}")
    
    # 计算每个类别的Precision, Recall, F1分数
    precision_scores = precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    recall_scores = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    f1_scores = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    
    # 计算宏平均和加权平均
    precision_macro = precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted', zero_division=0)
    
    print("\n每个类别的指标:")
    print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12} {precision_scores[i]:<12.4f} {recall_scores[i]:<12.4f} {f1_scores[i]:<12.4f}")
    
    print(f"\n宏平均 (Macro Average):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1: {f1_macro:.4f}")
    
    print(f"\n加权平均 (Weighted Average):")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1: {f1_weighted:.4f}")
    
    # ========== 详细的分类报告 ==========
    print(f"\n{'='*60}")
    print("详细分类报告:")
    print(f"{'='*60}")
    report = classification_report(
        y_true, y_pred, 
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        zero_division=0
    )
    print(report)
    
    # ========== 分析预测分布 ==========
    print(f"\n{'='*60}")
    print("预测分布分析:")
    print(f"{'='*60}")
    
    # 统计每个类别的预测数量
    predicted_dist = Counter(all_predicted_ranges)
    true_dist = Counter(all_ground_truth_ranges)
    
    print("\n真实标签分布:")
    for class_name in class_names:
        count = true_dist.get(class_name, 0)
        percentage = count / len(all_ground_truth_ranges) * 100 if all_ground_truth_ranges else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("\n预测标签分布:")
    for class_name in class_names:
        count = predicted_dist.get(class_name, 0)
        percentage = count / len(all_predicted_ranges) * 100 if all_predicted_ranges else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # ========== 分析检索效果 ==========
    print(f"\n{'='*60}")
    print("检索效果分析:")
    print(f"{'='*60}")
    
    # 统计相似案例数量（只使用有效预测）
    similar_cases_counts = [p.get('similar_cases_count', 0) for p in valid_predictions]
    avg_similar_cases = np.mean(similar_cases_counts) if similar_cases_counts else 0
    
    # 分析检索数量与准确率的关系
    print(f"\n平均检索到的相似案例数: {avg_similar_cases:.2f}")
    
    # 按检索数量分组统计准确率
    retrieval_groups = {
        "0个案例": [],
        "1-3个案例": [],
        "4-5个案例": [],
        "5+个案例": []
    }
    
    for pred in valid_predictions:  # 只统计有效预测
        count = pred.get('similar_cases_count', 0)
        is_correct = pred.get('predicted_range') == pred.get('true_range')
        
        if count == 0:
            retrieval_groups["0个案例"].append(is_correct)
        elif 1 <= count <= 3:
            retrieval_groups["1-3个案例"].append(is_correct)
        elif 4 <= count <= 5:
            retrieval_groups["4-5个案例"].append(is_correct)
        else:
            retrieval_groups["5+个案例"].append(is_correct)
    
    for group_name, results in retrieval_groups.items():
        if results:
            acc = sum(results) / len(results)
            print(f"  {group_name}: {sum(results)}/{len(results)} = {acc:.4f} ({acc*100:.2f}%)")
    
    stats_file = Path(args.output_dir) / "detailed_statistics.json"
    
    timeout_predictions = [p for p in all_predictions if p.get('timeout', False)]
    avg_timeout_time = sum(p.get('elapsed_time', 0) for p in timeout_predictions) / len(timeout_predictions) if timeout_predictions else 0
    successful_predictions = [p for p in all_predictions if not p.get('timeout', False) and p.get('elapsed_time', 0) > 0]
    avg_success_time = sum(p.get('elapsed_time', 0) for p in successful_predictions) / len(successful_predictions) if successful_predictions else 0
    
    stats_data = {
        "accuracy": accuracy,
        "mae": float(mae),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_per_class": {class_names[i]: float(precision_scores[i]) for i in range(5)},
        "recall_per_class": {class_names[i]: float(recall_scores[i]) for i in range(5)},
        "f1_per_class": {class_names[i]: float(f1_scores[i]) for i in range(5)},
        "confusion_matrix": cm.tolist(),
        "true_distribution": dict(true_dist),
        "predicted_distribution": dict(predicted_dist),
        "avg_similar_cases": float(avg_similar_cases),
        "retrieval_accuracy_by_group": {
            group_name: {
                "correct": sum(results),
                "total": len(results),
                "accuracy": sum(results) / len(results) if results else 0
            }
            for group_name, results in retrieval_groups.items()
        },
        "timeout_statistics": {
            "timeout_count": timeout_count,
            "timeout_percentage": float(timeout_count / total * 100) if total > 0 else 0,
            "error_count": error_count,
            "error_percentage": float(error_count / total * 100) if total > 0 else 0,
            "avg_timeout_time": float(avg_timeout_time),
            "avg_success_time": float(avg_success_time)
        }
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

