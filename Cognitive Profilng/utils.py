import argparse
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

prompt_module = SourceFileLoader("kg_prompt", str(CURRENT_DIR / "prompt.py")).load_module()
NEW_PROMPTS = prompt_module.NEW_PROMPTS

from data_loader import EduDataLoader
from nano_graphrag.graphrag import GraphRAG
from llm_client_local import LLMClient
from embedding_utils import create_conan_embedding_func
from pgvector_storage import PGVectorStorage
from cache_manager import CacheManager
from hashlib import md5

GRAPH_CONNECTOR = "-" * 10000

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """计算内容的 MD5 hash ID（与 GraphRAG 保持一致）"""
    return prefix + md5(content.encode()).hexdigest()


def chunking_by_chars(content: str, chars: str = None, tiktoken_model: str = "gpt-4o-2024-11-20") -> List[dict]:
    """
    按分隔符分割内容为 chunks（与 GraphRAG 的 chunking_by_chars 保持一致）
    
    Args:
        content: 要分割的内容
        chars: 分隔符（默认为 GRAPH_CONNECTOR）
        tiktoken_model: tiktoken 模型名称
    
    Returns:
        List of chunk dicts with keys: tokens, content, chunk_order_index
    """
    if chars is None:
        chars = GRAPH_CONNECTOR
    
    # 按分隔符分割内容
    content_parts = content.split(chars)
    results = []
    
    # 尝试导入 tiktoken 相关函数，如果失败则使用简单的长度计算
    try:
        from nano_graphrag._utils import encode_string_by_tiktoken
        for i, c in enumerate(content_parts):
            results.append({
                "tokens": len(encode_string_by_tiktoken(c, model_name=tiktoken_model)),
                "content": c.strip(),
                "chunk_order_index": i
            })
    except (ImportError, AttributeError):
        # 如果无法导入，使用简单的字符长度作为 tokens 的近似值
        for i, c in enumerate(content_parts):
            results.append({
                "tokens": len(c),  # 简单近似
                "content": c.strip(),
                "chunk_order_index": i
            })
    
    return results

def compute_chunk_ids_from_content(content: str) -> List[str]:
    """
    从内容计算 chunk IDs（模拟 GraphRAG 的 chunking 过程）
    注意：这里使用简单的字符分割，与 GraphRAG 的 chunking_by_chars 保持一致
    """
    chunks = chunking_by_chars(content, chars=GRAPH_CONNECTOR)
    chunk_ids = [compute_mdhash_id(chunk["content"], prefix="chunk-") for chunk in chunks]
    return chunk_ids


def determine_ability(csv_path: str) -> str:
    if "/低/" in csv_path:
        return "LOW"
    if "/中/" in csv_path:
        return "MEDIUM"
    if "/高/" in csv_path:
        return "HIGH"
    return "UNKNOWN"

def label_from_row(row) -> Tuple[str, float]:
    """
    根据得分和结果判断学生是否答对
    
    优先级：结果字段 > 得分字段
    - 如果结果包含"正确"，标记为正确
    - 如果结果包含"错误"（答案错误、运行错误、编译错误等），标记为错误
    - 否则，如果得分 > 0，标记为正确；否则标记为错误
    
    Returns:
        Tuple[str, float]: (tgt, label)
        - tgt: "Yes" 或 "No"，表示是否答对
        - label: 得分值（从数据集的"得分"字段获取）
    """
    score = row.get("得分", 0)
    # 确保得分是数值类型
    try:
        if score is None or score == "":
            score = 0
        else:
            score = float(score)
    except (ValueError, TypeError):
        score = 0
    
    result = str(row.get("结果", "")).strip()
    
    # 优先根据结果字段判断
    if "正确" in result:
        is_correct = 1
    elif any(keyword in result for keyword in ["错误", "失败", "超时", "编译错误", "运行错误", "答案错误"]):
        is_correct = 0
    else:
        # 如果结果字段不明确，根据得分判断
        # 得分 > 0 且接近满分（比如 >= 90）才认为是正确，否则认为是错误
        # 或者如果结果字段为空，使用得分判断
        if not result:  # 结果字段为空，使用得分判断
            is_correct = 1 if (score and score > 0) else 0
        else:
            # 结果字段有值但不包含明确的关键词，保守处理：得分 >= 90 才认为正确
            is_correct = 1 if (score and score >= 90) else 0
    
    # tgt 使用 Yes/No，label 使用实际的得分值
    return ("Yes" if is_correct else "No"), score


def build_training_strings(train_records: List[dict]) -> List[str]:
    strings = []
    for record in train_records:
        prompt = NEW_PROMPTS["get_library"].format(
            question=record["question"],
            desc=record["desc"],
            program=record["program"],
            error_desc=record["error_desc"],
        )
        strings.append(prompt)
    return strings



def summarise_knowledge(graph: GraphRAG, top_k: int = 5, record_count: int = None, 
                        chunk_timestamp_map: Dict[str, str] = None, 
                        current_timestamp: str = None,
                        knowledge_points: List[str] = None) -> str:
    """
    Args:
        graph: GraphRAG 实例
        top_k: 返回前 k 个强项和弱项
        chunk_timestamp_map: chunk_id -> timestamp 的映射（用于过滤节点，只考虑该时刻之前的数据）
        current_timestamp: 当前时间戳（用于过滤节点，只考虑该时刻之前的数据）
    """
    import re
    nodes = []
    GRAPH_FIELD_SEP = "<SEP>"
    
    valid_chunks = None
    if chunk_timestamp_map and current_timestamp:

        valid_chunks = {chunk_id for chunk_id, ts in chunk_timestamp_map.items() 
                      if ts <= current_timestamp}
        total_chunks = len(chunk_timestamp_map)
        print(f"[Summarise] Filtering chunks: {len(valid_chunks)}/{total_chunks} chunks before timestamp {current_timestamp}")
        if len(valid_chunks) == 0:
            print(f"[Summarise] Warning: No valid chunks found! This may cause empty summary.")
            print(f"[Summarise] Available timestamps in map: {sorted(set(chunk_timestamp_map.values()))[:5]}")
            print(f"[Summarise] Current timestamp: {current_timestamp}")
    
    knowledge_points_set = None
    if knowledge_points and len(knowledge_points) > 0:
        knowledge_points_set = set()
        for kp in knowledge_points:
            kp = kp.strip()
            if kp:
                knowledge_points_set.add(kp)
                knowledge_points_set.add(kp.replace(' ', '_'))
                knowledge_points_set.add(kp.replace('_', ' '))
                knowledge_points_set.add(kp.upper())
                knowledge_points_set.add(kp.replace(' ', '_').upper())
                knowledge_points_set.add(kp.replace('_', ' ').upper())
        print(f"[Summarise] 知识点列表过滤: 将只保留 {len(knowledge_points_set)} 个知识点变体中的实体")
    
    # 需要排除的非知识点类型
    excluded_types = ["PERSON", '"PERSON"', "EVENT", '"EVENT"', "TECHNOLOGY", '"TECHNOLOGY"']
    # 需要排除的明显非知识点节点名称（人名、程序名等）
    excluded_names = ["小红", "学生程序", "XIAO HONG", "STUDENT PROGRAM", "SOLVE FUNCTION", "MAIN FUNCTION"]
    
    for node, data in graph.chunk_entity_relation_graph._graph.nodes(data=True):
        entity_type = data.get("entity_type", "").upper()
        node_name = str(node)
        node_name_upper = node_name.upper()
        
        # 排除明显非知识点的类型
        if entity_type in excluded_types:
            continue
        
        # 排除明显非知识点的节点名称（人名、程序名等）
        if any(excluded_name in node_name or excluded_name in node_name_upper for excluded_name in excluded_names):
            continue
        
        # 如果提供了知识点列表，只保留列表中的知识点
        if knowledge_points_set is not None:
            # 检查节点名称是否在知识点列表中（支持多种格式匹配）
            node_name_variants = [
                node_name,
                node_name.replace(' ', '_'),
                node_name.replace('_', ' '),
                node_name_upper,
                node_name_upper.replace(' ', '_'),
                node_name_upper.replace('_', ' ')
            ]
            # 检查是否有任何变体匹配知识点列表
            if not any(variant in knowledge_points_set for variant in node_name_variants):
                # 如果节点名称不在知识点列表中，跳过
                continue
        
        # 保留所有 CONCEPT 类型的节点，以及其他看起来像知识点的节点
        # 如果节点名称包含常见知识点关键词，或者描述看起来像知识点，都保留
        is_concept = entity_type in ["CONCEPT", '"CONCEPT"']
        has_knowledge_keywords = any(kw in node_name_upper for kw in [
            "STRUCTURE", "ALGORITHM", "THEORY", "SEARCH", "SORT", "ARRAY", "LIST", 
            "STACK", "QUEUE", "GRAPH", "TREE", "HASH", "DYNAMIC", "GREEDY"
        ])
        
        # 如果既不是 CONCEPT，也不包含知识点关键词，但描述看起来像知识点，也保留
        description = data.get("description", "").lower()
        looks_like_knowledge = any(kw in description for kw in [
            "knowledge", "concept", "algorithm", "structure", "technique", "method"
        ])
        
        # 放宽过滤条件：保留更多实体（不仅仅是CONCEPT类型）
        if not (is_concept or has_knowledge_keywords or looks_like_knowledge):
            # 如果实体类型不是明确排除的类型，也保留（特别是包含常见编程术语的）
            if entity_type not in ["PERSON", '"PERSON"', "GEO", '"GEO"', "ORGANIZATION", '"ORGANIZATION"']:
                # 如果节点名称看起来像知识点（包含常见编程术语），也保留
                if any(kw in node_name_upper or kw in node_name for kw in ["数组", "ARRAY", "区间", "INTERVAL", "行列式", "DETERMINANT", "回文", "PALINDROME", "子数组", "SUBARRAY", "极差", "RANGE", "排列", "PERMUTATION"]):
                    pass  # 保留
                else:
                    continue  # 跳过
            else:
                continue  # 跳过明确排除的类型
        
        description = data.get("description", "")
        if not description:
            continue
        
        # 从 description 中解析 master_level
        descriptions = description.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in description else [description]
        
        good_count = 0
        bad_count = 0
        total_count = 0
        
        for desc in descriptions:
            desc = desc.strip()
            if not desc:
                continue
            # 检查是否包含 "Good:" 或 "Bad:"（可能在引号内）
            desc_lower = desc.lower()
            if "good:" in desc_lower or desc_lower.startswith("good"):
                good_count += 1
                total_count += 1
            elif "bad:" in desc_lower or desc_lower.startswith("bad"):
                bad_count += 1
                total_count += 1
        
        # 先统一获取和过滤 source_ids
        source_id = data.get("source_id", "")
        source_ids = source_id.split(GRAPH_FIELD_SEP) if source_id else []
        original_count = len(source_ids)
        
        # 如果提供了 valid_chunks，只计算在 valid_chunks 中的 chunk
        if valid_chunks:
            source_ids = [sid for sid in source_ids if sid in valid_chunks]
            # 如果过滤后没有有效的 chunk，但原来有，说明这个节点是在该时刻之后出现的
            # 这种情况下，我们应该跳过这个节点
            if original_count > 0 and len(source_ids) == 0:
                # 跳过该时刻之后出现的节点
                continue
        
        # 计算过滤后的 source_count
        source_count = len(source_ids) if source_ids else 0
        
        # 如果没有找到 Good/Bad 标记，基于节点出现频率推断
        if total_count == 0:
            # 如果 source_count 为 0，说明这个节点在该时刻之前没有出现，应该跳过
            if source_count == 0:
                continue
            
            # 如果出现频率高，可能是掌握较好的知识点
            # 使用更细粒度的分数计算，避免所有节点都是 0.7
            score = min(0.95, 0.3 + source_count * 0.05)  # 0.3-0.95 之间，更细粒度
            level = "Good" if score >= 0.5 else "Bad"
        else:
            # 计算平均分数：Good=1.0, Bad=0.0
            base_score = good_count / total_count if total_count > 0 else 0.5
            level = "Good" if base_score >= 0.5 else "Bad"
            
            # 根据 source_count 和 record_count 调整分值，确保能反映图谱的变化
            # 这样即使 Good/Bad 比例相同，出现频率不同的知识点也会有不同分值
            if record_count and record_count > 0 and source_count > 0:
                # 计算出现频率因子：source_count / record_count
                frequency_factor = source_count / record_count
                # 对于 Good 的知识点，频率越高分值越高（0.50-1.00）
                if base_score >= 0.5:
                    score = 0.50 + frequency_factor * 0.50
                # 对于 Bad 的知识点，频率越高分值越低（0.00-0.50）
                else:
                    score = 0.50 - frequency_factor * 0.50
            else:
                score = base_score
        
        # 计算相对增长因子（如果提供了 record_count）
        relative_growth = 0.0
        if record_count and record_count > 0:
            # 相对增长 = source_count / record_count
            # 这样，即使 source_count 相同，相对增长更大的节点也会排在前面
            relative_growth = source_count / record_count
        
        nodes.append(
            {
                "name": node,
                "score": score,
                "level": level,
                "description": description,
                "source_count": source_count,  # 添加 source_count 用于排序
                "relative_growth": relative_growth,  # 相对增长因子
            }
        )
    
    print(f"[Summarise] Extracted {len(nodes)} knowledge points after filtering")
    if valid_chunks:
        strengths_count = len([n for n in nodes if n["level"] == "Good"])
        weaknesses_count = len([n for n in nodes if n["level"] == "Bad"])
        print(f"[Summarise] Strengths: {strengths_count}, Weaknesses: {weaknesses_count}")
        
        # 显示 source_count 的分布
        source_counts = [n["source_count"] for n in nodes]
        if source_counts:
            print(f"[Summarise] Source count distribution: min={min(source_counts)}, max={max(source_counts)}, avg={sum(source_counts)/len(source_counts):.2f}")
            # 显示 source_count 的分布（按范围）
            count_1 = sum(1 for c in source_counts if c == 1)
            count_2_5 = sum(1 for c in source_counts if 2 <= c <= 5)
            count_6_10 = sum(1 for c in source_counts if 6 <= c <= 10)
            count_11_plus = sum(1 for c in source_counts if c > 10)
            print(f"[Summarise] Source count ranges: 1={count_1}, 2-5={count_2_5}, 6-10={count_6_10}, 11+={count_11_plus}")
    
    if not nodes:
        return "Student Knowledge Summary:\nNo knowledge points extracted from the graph."
    
    # 排序时，先按 score，再按 relative_growth（如果可用），最后按 source_count
    # 这样能更好地反映知识图谱的增量变化
    if record_count:
        strengths = sorted(
            [n for n in nodes if str(n["level"]).lower() == "good"],
            key=lambda x: (x["score"], x.get("relative_growth", 0), x.get("source_count", 0)),
            reverse=True,
        )
        weaknesses = sorted(
            [n for n in nodes if str(n["level"]).lower() == "bad"],
            key=lambda x: (x["score"], -x.get("relative_growth", 0), -x.get("source_count", 0)),
        )
    else:
        # 如果没有 record_count，使用原来的排序逻辑
        strengths = sorted(
            [n for n in nodes if str(n["level"]).lower() == "good"],
            key=lambda x: (x["score"], x.get("source_count", 0)),
            reverse=True,
        )
        weaknesses = sorted(
            [n for n in nodes if str(n["level"]).lower() == "bad"],
            key=lambda x: (x["score"], -x.get("source_count", 0)),
        )
    
    summary_lines = ["Student Knowledge Summary:"]
    if strengths:
        summary_lines.append("Strengths:")
        for n in strengths[:top_k]:
            summary_lines.append(f"- {n['name']} (score {n['score']:.2f})")
    if weaknesses:
        summary_lines.append("Needs Improvement:")
        for n in weaknesses[:top_k]:
            summary_lines.append(f"- {n['name']} (score {n['score']:.2f})")
    
    if len(summary_lines) == 1:  # 只有标题，没有内容
        # 如果既没有 strengths 也没有 weaknesses，列出所有知识点
        summary_lines.append("All Knowledge Points:")
        for n in sorted(nodes, key=lambda x: x["score"], reverse=True)[:top_k * 2]:
            summary_lines.append(f"- {n['name']} (score {n['score']:.2f})")
    
    return "\n".join(summary_lines)


def build_prompt(question_row, ability: str) -> str:
    instruction = (
        "You are an AI tutor. Based on the student's profile and the problem, "
        "predict whether the student will solve it correctly. "
        "Answer with 'Yes' or 'No'."
    )
    problem = question_row.get("题干", "")
    desc = question_row.get("题目", "")
    prompt_parts = [
        instruction,
        f"Innate ability: {ability}",
        f"Problem title: {desc}",
        f"Problem statement:\n{problem}",
    ]
    return "\n\n".join(part for part in prompt_parts if part)


def records_from_dataframe(df, ability: str, knowledge_summary=None, knowledge_summary_func=None, knowledge_summaries: dict=None) -> List[dict]:

    samples = []
    for _, row in df.iterrows():
        tgt, label = label_from_row(row)
        # 处理 timestamp，确保是字符串类型
        timestamp = row.get("时间", "")
        if timestamp is not None:
            # 如果是 pandas Timestamp 类型，转换为字符串
            if hasattr(timestamp, 'strftime'):
                timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp = str(timestamp)
        else:
            timestamp = ""
        
        # 根据模式选择知识总结（优先级：固定总结 > 时间序列字典 > 动态函数）
        if knowledge_summary:
            # 默认模式：使用固定的知识总结
            current_knowledge_summary = knowledge_summary
        elif knowledge_summaries:
            # 预计算模式：从字典中查找
            # 找到该时刻之前最近的知识总结
            current_knowledge_summary = knowledge_summaries.get(timestamp, "")
            if not current_knowledge_summary:
                # 如果找不到精确匹配，找最接近的时间戳（该时刻之前）
                # 尝试将时间戳转换为可比较的格式
                try:
                    from datetime import datetime
                    target_time = pd.to_datetime(timestamp) if timestamp else None
                    if target_time:
                        # 找到该时刻之前的所有时间戳
                        before_timestamps = []
                        for ts in knowledge_summaries.keys():
                            try:
                                ts_time = pd.to_datetime(ts)
                                if ts_time <= target_time:
                                    before_timestamps.append((ts_time, ts))
                            except:
                                # 如果时间戳无法解析，使用字符串比较
                                if ts <= timestamp:
                                    before_timestamps.append((None, ts))
                        
                        if before_timestamps:
                            # 按时间排序，取最新的
                            before_timestamps.sort(key=lambda x: x[0] if x[0] is not None else pd.Timestamp.min, reverse=True)
                            current_knowledge_summary = knowledge_summaries[before_timestamps[0][1]]
                        else:
                            current_knowledge_summary = "Student Knowledge Summary:\nNo knowledge points extracted yet."
                    else:
                        # 时间戳无法解析，使用字符串比较
                        sorted_timestamps = sorted([ts for ts in knowledge_summaries.keys() if ts <= timestamp], reverse=True)
                        if sorted_timestamps:
                            current_knowledge_summary = knowledge_summaries[sorted_timestamps[0]]
                        else:
                            current_knowledge_summary = "Student Knowledge Summary:\nNo knowledge points extracted yet."
                except Exception:
                    # 如果时间比较失败，使用字符串比较作为兜底
                    sorted_timestamps = sorted([ts for ts in knowledge_summaries.keys() if ts <= timestamp], reverse=True)
                    if sorted_timestamps:
                        current_knowledge_summary = knowledge_summaries[sorted_timestamps[0]]
                    else:
                        current_knowledge_summary = "Student Knowledge Summary:\nNo knowledge points extracted yet."
        elif knowledge_summary_func:
            # 时间序列模式：使用函数动态生成
            current_knowledge_summary = knowledge_summary_func(timestamp)
        else:
            # 兜底：没有提供任何知识总结
            current_knowledge_summary = "Student Knowledge Summary:\nNo knowledge points extracted."
        
        record = {
            "src": build_prompt(row, ability),
            "e": current_knowledge_summary,
            "tgt": tgt,
            "label": label,
            "user_id": str(row.get("用户", "")),
            "problem_name": str(row.get("题目", "")),
            "timestamp": timestamp,
        }
        samples.append(record)
    return samples


