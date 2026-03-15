import argparse
import json
import os
import sys
import re
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent.parent.parent
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
import asyncio
import time

# 导入原脚本中的辅助函数
sys.path.insert(0, str(CURRENT_DIR))
from utils import (
    compute_chunk_ids_from_content,
    determine_ability,
    summarise_knowledge,
    records_from_dataframe
)
from graph_build_stats import GraphBuildStatsCollector

GRAPH_CONNECTOR = "-" * 10000


def fix_json_escape_errors(json_str: str) -> str:
    """
    尝试修复 JSON 字符串中的转义字符错误
    Args:
        json_str: 可能包含转义错误的 JSON 字符串
    Returns:
        修复后的 JSON 字符串
    """
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
    fixed = json_str
    
    # 修复 1: 处理字符串值中的未转义反斜杠（但保留已转义的）
    # 定义正则表达式模式（避免在 f-string 中使用反斜杠）
    invalid_escape_pattern = r'\\(?!["\\/bfnrtu])'
    replacement = r'\\\\'
    
    def fix_backslashes_in_strings(match):
        content = match.group(1)
        # 替换未转义的反斜杠（后面不是引号、n、t、r、u、\等有效转义字符）
        fixed_content = re.sub(invalid_escape_pattern, replacement, content)
        return f'"{fixed_content}"'
    
    # 匹配 JSON 字符串值（在引号之间的内容）
    string_pattern = r'"([^"]*(?:\\.[^"]*)*)"'
    fixed = re.sub(string_pattern, fix_backslashes_in_strings, fixed)
    
    # 修复 2: 处理 \U 或 \u 后跟非十六进制字符的情况
    # 将 \U 或 \u 后跟非十六进制字符的情况转义为 \\U 或 \\u
    fixed = re.sub(r'\\([Uu])(?![0-9A-Fa-f]{4,8})', r'\\\\\1', fixed)
    
    # 修复 3: 处理 \x 后跟非十六进制字符的情况
    fixed = re.sub(r'\\x(?![0-9A-Fa-f]{2})', r'\\\\x', fixed)
    
    return fixed


def convert_response_to_json_with_fix(response: str) -> dict:

    from nano_graphrag._utils import locate_json_string_body_from_string
    import logging
    
    logger = logging.getLogger("nano-graphrag")
    
    json_str = locate_json_string_body_from_string(response)
    if json_str is None:
        raise ValueError(f"Unable to parse JSON from response: {response}")
    
    fixed_json_str = fix_json_escape_errors(json_str)
    
    try:
        data = json.loads(fixed_json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON even after fixing escapes. Original: {json_str[:200]}... Fixed: {fixed_json_str[:200]}...")
        raise e from None


def wrap_clustering_for_stats(original_clustering_method, stats_collector: GraphBuildStatsCollector):
    async def wrapped_clustering(algorithm: str):
        start_time = time.time()
        try:
            # 调用原始的聚类方法
            result = await original_clustering_method(algorithm)
            elapsed_time = time.time() - start_time
            # 记录统计信息
            if stats_collector:
                stats_collector.record_clustering(elapsed_time)
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            # 即使出错也记录时间
            if stats_collector:
                stats_collector.record_clustering(elapsed_time)
            raise e
    
    return wrapped_clustering


def split_program_into_chunks(program: str, max_chars: int) -> List[str]:
    """
    将学生程序按指定字符数切分成多个片段，尽量在换行处切分。
    """
    if not program:
        return []
    if max_chars <= 0 or len(program) <= max_chars:
        return [program]

    chunks = []
    current_lines: List[str] = []
    current_len = 0

    for line in program.splitlines():
        line_with_newline = line + "\n"
        line_len = len(line_with_newline)
        if current_len + line_len > max_chars and current_lines:
            chunks.append("".join(current_lines).rstrip())
            current_lines = [line_with_newline]
            current_len = line_len
        else:
            current_lines.append(line_with_newline)
            current_len += line_len

    if current_lines:
        chunks.append("".join(current_lines).rstrip())

    return [chunk for chunk in chunks if chunk.strip()] or [program]


def build_entity_graph_simple(
    graph: GraphRAG, 
    formatted_train: List[Dict], 
    output_dir: str,
    args,
    stats_collector: GraphBuildStatsCollector = None
) -> Dict[str, str]:
    """
    构建题干实体图谱
    """
    knowledge_summaries = {}
    chunk_timestamp_map = {}
    progress_file = os.path.join(output_dir, "entity_progress.json")
    start_idx = 0
    
    # 加载已有进度
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                knowledge_summaries = progress_data.get("knowledge_summaries", {})
                chunk_timestamp_map = progress_data.get("chunk_timestamp_map", {})
                start_idx = progress_data.get("last_processed_idx", 0)
                print(f"[题干实体图谱] 找到已有进度: 已处理 {start_idx}/{len(formatted_train)} 条记录")
        except Exception as e:
            print(f"[题干实体图谱] 警告: 加载进度文件失败: {e}")
    
    total_records = len(formatted_train)
    
    # 性能优化：减少保存频率（每 SAVE_INTERVAL 条记录保存一次）
    SAVE_INTERVAL = 20  # 每20条记录保存一次图谱
    print(f"[实体图谱] 性能优化：每 {SAVE_INTERVAL} 条记录保存一次图谱，减少序列化开销")
    
    # 包装 NetworkXStorage 以控制保存频率
    original_storage = graph.chunk_entity_relation_graph
    if not hasattr(graph, '_entity_save_counter'):
        original_index_done = original_storage.index_done_callback
        save_counter = {'count': 0, 'enabled': False}
        
        async def delayed_index_done_callback():
            save_counter['count'] += 1
            if save_counter['enabled'] or save_counter['count'] >= SAVE_INTERVAL:
                await original_index_done()
                save_counter['count'] = 0
                save_counter['enabled'] = False
        
        original_storage.index_done_callback = delayed_index_done_callback
        graph._entity_save_counter = save_counter
    
    # 更新统计信息
    if stats_collector:
        stats_collector.update_record_counts(total_records, 0)  # unique_questions 不再需要
    
    # ========== 按时间顺序处理 ==========
    print("\n[处理] 按时间顺序处理记录...")
    
    processed_count = 0
    for idx, record in enumerate(formatted_train):
        if idx < start_idx:
            continue
        
        print(f"\n处理记录 {idx + 1}/{total_records}")
        
        question = record.get("question", "")
        desc = record.get("desc", "")
        program = record.get("program", "")
        error_desc = record.get("error_desc", "")
        
        timestamp = record.get("timestamp", "")
        if timestamp is not None:
            if hasattr(timestamp, 'strftime'):
                timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp = str(timestamp)
        else:
            timestamp = ""
        
        if not timestamp:
            timestamp = f"record_{idx}"
        
        unique_timestamp = f"{timestamp}_record_{idx + 1}"
        
        # 如果程序太长，进行分段处理
        program_chunks = split_program_into_chunks(program, args.program_chunk_chars)
        chunk_count = len(program_chunks)
        
        if chunk_count > 1:
            print(f"  程序分段: {chunk_count} 段 (每段最多 {args.program_chunk_chars} 字符)")
        
        for seg_idx, program_chunk in enumerate(program_chunks, start=1):
            # 重新格式化，使用分段后的程序
            if chunk_count > 1:
                segment_header = f"[Program Segment {seg_idx}/{chunk_count}]\n"
                formatted_input = NEW_PROMPTS['get_library'].format(
                    question=question,
                    desc=desc,
                    program=segment_header + program_chunk,
                    error_desc=error_desc
                )
            else:
                formatted_input = NEW_PROMPTS['get_library'].format(
                    question=question,
                    desc=desc,
                    program=program,
                    error_desc=error_desc
                )
            
            # 添加时间戳和记录索引
            unique_suffix = f"\n\n[Record ID: {idx + 1}-{seg_idx}, Timestamp: {timestamp}]"
            formatted_input = formatted_input + unique_suffix
            
            start_time = time.time()
            
            try:
                if args.insert_timeout:
                    asyncio.run(
                        asyncio.wait_for(
                            graph.ainsert(formatted_input),
                            timeout=args.insert_timeout
                        )
                    )
                else:
                    asyncio.run(graph.ainsert(formatted_input))
                elapsed_time = time.time() - start_time
                print(f"  插入完成，耗时 {elapsed_time:.2f} 秒")
                
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=True)
                    stats_collector.record_llm_call(elapsed_time * 0.8, success=True) 
                
                chunk_ids = compute_chunk_ids_from_content(formatted_input)
                for chunk_id in chunk_ids:
                    chunk_timestamp_map[chunk_id] = unique_timestamp
            except asyncio.TimeoutError:
                elapsed_time = time.time() - start_time
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=False)
                    stats_collector.record_llm_call(elapsed_time, success=False, is_timeout=True)
                continue
            except Exception as e:
                elapsed_time = time.time() - start_time
                error_msg = str(e)
                error_type = type(e).__name__
                is_timeout = ("504" in error_msg or "Gateway Timeout" in error_msg or 
                             "timeout" in error_msg.lower() or "TimeoutError" in error_type)
                if is_timeout:
                    print(f"  LLM 调用超时，跳过此段")
                elif "EmptyNetwork" in error_msg or "empty" in error_msg.lower() or "EmptyNetwork" in error_type:
                    print(f"  警告: 图谱中没有边（只有节点），社区报告生成跳过，但实体已添加")
                    if stats_collector:
                        stats_collector.record_graph_insert(elapsed_time, success=True)  
                    try:
                        chunk_ids = compute_chunk_ids_from_content(formatted_input)
                        for chunk_id in chunk_ids:
                            chunk_timestamp_map[chunk_id] = unique_timestamp
                    except:
                        pass
                    continue
                elif "JSONDecodeError" in error_type or "Invalid" in error_msg and "escape" in error_msg:
                    print(f"  JSON 解析错误（LLM 返回格式问题），跳过此段: {error_msg[:200]}...")
                else:
                    print(f"  插入失败 ({error_type})，跳过此段: {error_msg[:200]}...")
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=False)
                    stats_collector.record_llm_call(elapsed_time, success=False, is_timeout=is_timeout)
                continue
        
        if (idx + 1) % SAVE_INTERVAL == 0 or idx == len(formatted_train) - 1:
            if hasattr(graph, '_entity_save_counter'):
                graph._entity_save_counter['enabled'] = True
                graph._entity_save_counter['count'] = SAVE_INTERVAL 
                try:
                    asyncio.run(graph.chunk_entity_relation_graph.index_done_callback())
                    print(f"[实体图谱] 已保存图谱（记录 {idx + 1}/{total_records}）")
                except Exception as e:
                    print(f"[实体图谱] 警告: 手动保存失败: {e}")
        
        # 生成Summary
        if (idx + 1) % args.snapshot_interval == 0 or idx == len(formatted_train) - 1:
            num_nodes = len(graph.chunk_entity_relation_graph._graph.nodes())
            num_edges = len(graph.chunk_entity_relation_graph._graph.edges())
            start_time = time.time()
            try:
                knowledge_summary = summarise_knowledge(
                    graph,
                    top_k=args.top_k,
                    record_count=idx + 1,
                    chunk_timestamp_map=chunk_timestamp_map,
                    current_timestamp=unique_timestamp
                )
                elapsed_time = time.time() - start_time
                knowledge_summaries[unique_timestamp] = knowledge_summary
                print(f"[题干实体图谱] 生成知识总结快照完成，耗时 {elapsed_time:.2f} 秒")
            except Exception as e:
                print(f"[题干实体图谱] 警告: 生成知识总结失败: {e}")
                knowledge_summaries[unique_timestamp] = f"生成知识总结失败: {e}"
        
        try:
            progress_data = {
                "last_processed_idx": idx,
                "knowledge_summaries": knowledge_summaries,
                "chunk_timestamp_map": chunk_timestamp_map,
                "total_records": len(formatted_train)
            }
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[题干实体图谱] 警告: 保存进度失败: {e}")
        
        # 每20条记录保存一次统计信息
        if stats_collector and (idx + 1) % 10 == 0:
            try:
                stats_collector.save(args.output_dir, "build_stats.json")
                print(f"[统计] 已保存统计信息（处理到第 {idx + 1} 条记录）")
            except Exception as e:
                print(f"[统计] 警告: 保存统计信息失败: {e}")
        
        processed_count += 1
    
    # 更新图谱规模
    if stats_collector:
        num_nodes = len(graph.chunk_entity_relation_graph._graph.nodes())
        num_edges = len(graph.chunk_entity_relation_graph._graph.edges())
        # 累加到已有的图谱规模（实体图谱 + 知识点图谱）
        current_nodes = stats_collector.stats.final_node_count
        current_edges = stats_collector.stats.final_edge_count
        stats_collector.update_graph_size(current_nodes + num_nodes, current_edges + num_edges)
    
    print(f"\n[处理] 完成，共处理 {processed_count} 条记录")
    
    return knowledge_summaries


def build_knowledge_point_graph_simple(
    graph: GraphRAG,
    knowledge_points: List[str] = None,
    formatted_train: List[Dict] = None,
    output_dir: str = None,
    args = None,
    stats_collector: GraphBuildStatsCollector = None
) -> Dict[str, str]:
    knowledge_summaries = {}
    chunk_timestamp_map = {}
    progress_file = os.path.join(output_dir, "knowledge_point_progress.json")
    start_idx = 0
    
    # 确保进度文件目录存在
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # 加载已有进度
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                knowledge_summaries = progress_data.get("knowledge_summaries", {})
                chunk_timestamp_map = progress_data.get("chunk_timestamp_map", {})
                start_idx = progress_data.get("last_processed_idx", 0)
                print(f"[知识点图谱] 找到已有进度: 已处理 {start_idx}/{len(formatted_train)} 条记录")
        except Exception as e:
            print(f"[知识点图谱] 警告: 加载进度文件失败: {e}")
    else:
        # 创建初始进度文件
        try:
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({
                    "last_processed_idx": -1,
                    "knowledge_summaries": {},
                    "chunk_timestamp_map": {},
                    "total_records": len(formatted_train)
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[知识点图谱] 警告: 创建进度文件失败: {e}")
    
    total_records = len(formatted_train)
    
    if stats_collector and stats_collector.stats.unique_questions == 0:
        stats_collector.update_record_counts(total_records, 0)  # unique_questions 不再需要
    
    print("\n[处理] 按时间顺序处理记录...")
    
    processed_count = 0
    for idx, record in enumerate(formatted_train):
        if idx < start_idx:
            continue
        
        print(f"\n处理记录 {idx + 1}/{total_records}")
        
        question = record.get("question", "")
        question_context = f"{question}\n{record.get('desc', '')}"
        desc = record.get("desc", "")
        program_chunks = split_program_into_chunks(record.get("program", ""), args.program_chunk_chars)
        if not program_chunks:
            program_chunks = [""]
        chunk_count = len(program_chunks)
        if chunk_count > 1:
            print(f"[知识点图谱] 记录 {idx + 1} 程序分段: {chunk_count} 段 (每段最多 {args.program_chunk_chars} 字符)")
        
        any_success = False
        for seg_idx, program_chunk in enumerate(program_chunks, start=1):
                segment_program = program_chunk
                if chunk_count > 1:
                    segment_program = f"[Program Segment {seg_idx}/{chunk_count}]\n{program_chunk}"
                
                if knowledge_points and len(knowledge_points) > 0:
                    kp_display = ', '.join(knowledge_points)
                    if len(kp_display) > 500:
                        kp_display = ', '.join(knowledge_points[:20]) + f" ... (and {len(knowledge_points) - 20} more)"
                    knowledge_text = f"""Problem Context (for reference only, do not extract entities from this):
Question: {question_context}

Student Program:
{segment_program}

Error Description:
{record.get('error_desc', '')}

Knowledge Points to Extract (ONLY extract knowledge points from this list):
{kp_display}

Instructions:
- ONLY extract knowledge points that appear in the list above
- Do NOT extract any entities from the problem context or student program (such as variable names, function names, or specific problem elements)
- Focus on identifying which knowledge points from the list are relevant to this problem
- For each relevant knowledge point, evaluate the student's understanding based on the student's program and error description:
  * If the error description indicates the student answered CORRECTLY (e.g., "No error", "无错误", or no errors mentioned): Use "Good: [description]" to indicate the student demonstrates understanding
  * If the error description indicates the student answered INCORRECTLY (e.g., contains error details): Use "Bad: [description]" to indicate misunderstanding or lack of understanding
  * Knowledge points not mentioned in the error description can be considered as areas where the student has a good grasp
- The mastery level should reflect the student's performance on this specific problem and submission, similar to how the original entity extraction prompt works
- Different submissions of the same problem should be evaluated independently based on the student's program and error description"""
                else:
                    knowledge_text = NEW_PROMPTS["get_library"].format(
                        question=question,
                        desc=desc,
                        program=segment_program,
                        error_desc=record.get('error_desc', '')
                    )
                
                timestamp = record.get("timestamp", "")
                if timestamp is not None:
                    if hasattr(timestamp, 'strftime'):
                        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        timestamp = str(timestamp)
                else:
                    timestamp = ""
                
                if not timestamp:
                    timestamp = f"record_{idx}"
                
                unique_timestamp = f"{timestamp}_record_{idx + 1}"
                unique_suffix = f"\n\n[Record ID: {idx + 1}-{seg_idx}, Timestamp: {timestamp}]"
                knowledge_text = knowledge_text + unique_suffix
                
                print(f"[知识点图谱] 准备插入记录 {idx + 1}-{seg_idx}/{chunk_count}...")
                start_time = time.time()
                
                try:
                    if args.insert_timeout:
                        asyncio.run(
                            asyncio.wait_for(
                                graph.ainsert(knowledge_text),
                                timeout=args.insert_timeout
                            )
                        )
                    else:
                        asyncio.run(graph.ainsert(knowledge_text))
                    elapsed_time = time.time() - start_time
                    print(f"[知识点图谱] 插入记录 {idx + 1}-{seg_idx} 完成，耗时 {elapsed_time:.2f} 秒")
                    
                    if stats_collector:
                        stats_collector.record_graph_insert(elapsed_time, success=True)
                        stats_collector.record_llm_call(elapsed_time * 0.8, success=True)
                    
                    chunk_ids = compute_chunk_ids_from_content(knowledge_text)
                    for chunk_id in chunk_ids:
                        chunk_timestamp_map[chunk_id] = unique_timestamp
                    any_success = True
                except asyncio.TimeoutError:
                    elapsed_time = time.time() - start_time
                    print(f"[知识点图谱] 插入操作超时（{args.insert_timeout} 秒），跳过此段")
                    if stats_collector:
                        stats_collector.record_graph_insert(elapsed_time, success=False)
                        stats_collector.record_llm_call(elapsed_time, success=False, is_timeout=True)
                    continue
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    error_msg = str(e)
                    error_type = type(e).__name__
                    is_timeout = ("504" in error_msg or "Gateway Timeout" in error_msg or 
                                 "timeout" in error_msg.lower() or "TimeoutError" in error_type)
                    if is_timeout:
                        print(f"[知识点图谱] LLM 调用超时，跳过此段")
                    elif "EmptyNetwork" in error_msg or "empty" in error_msg.lower() or "EmptyNetwork" in error_type:
                        print(f"[知识点图谱] 警告: 图谱中没有边（只有节点），社区报告生成跳过，但实体已添加")
                        if stats_collector:
                            stats_collector.record_graph_insert(elapsed_time, success=True)
                        try:
                            chunk_ids = compute_chunk_ids_from_content(knowledge_text)
                            for chunk_id in chunk_ids:
                                chunk_timestamp_map[chunk_id] = unique_timestamp
                        except:
                            pass
                        continue
                    elif "JSONDecodeError" in error_type or "Invalid" in error_msg and "escape" in error_msg:
                        print(f"[知识点图谱] JSON 解析错误（LLM 返回格式问题），跳过此段: {error_msg[:200]}...")
                    else:
                        print(f"[知识点图谱] 插入失败 ({error_type})，跳过此段: {error_msg[:200]}...")
                    if stats_collector:
                        stats_collector.record_graph_insert(elapsed_time, success=False)
                        stats_collector.record_llm_call(elapsed_time, success=False, is_timeout=is_timeout)
                    continue
        
        # 生成Summary
        if (idx + 1) % args.snapshot_interval == 0 or idx == len(formatted_train) - 1:
            num_nodes = len(graph.chunk_entity_relation_graph._graph.nodes())
            num_edges = len(graph.chunk_entity_relation_graph._graph.edges())
            print(f"[知识点图谱] 图谱统计: {num_nodes} 个节点, {num_edges} 条边")
            
            print(f"[知识点图谱] 准备生成知识总结快照: {unique_timestamp} (记录 {idx + 1})...")
            start_time = time.time()
            try:
                knowledge_summary = summarise_knowledge(
                    graph,
                    top_k=args.top_k,
                    record_count=idx + 1,
                    chunk_timestamp_map=chunk_timestamp_map,
                    current_timestamp=unique_timestamp,
                    knowledge_points=knowledge_points
                )
                elapsed_time = time.time() - start_time
                knowledge_summaries[unique_timestamp] = knowledge_summary
                print(f"[知识点图谱] 生成知识总结快照完成，耗时 {elapsed_time:.2f} 秒")
            except Exception as e:
                print(f"[知识点图谱] 警告: 生成知识总结失败: {e}")
                knowledge_summaries[unique_timestamp] = f"生成知识总结失败: {e}"
        
        try:
            progress_data = {
                "last_processed_idx": idx,
                "knowledge_summaries": knowledge_summaries,
                "chunk_timestamp_map": chunk_timestamp_map,
                "total_records": len(formatted_train)
            }
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[知识点图谱] 警告: 保存进度失败: {e}")
        
        if stats_collector and (idx + 1) % 10 == 0:
            try:
                stats_collector.save(args.output_dir, "build_stats.json")
                print(f"[统计] 已保存统计信息（处理到第 {idx + 1} 条记录）")
            except Exception as e:
                print(f"[统计] 警告: 保存统计信息失败: {e}")
        
        processed_count += 1
    
    if stats_collector:
        num_nodes = len(graph.chunk_entity_relation_graph._graph.nodes())
        num_edges = len(graph.chunk_entity_relation_graph._graph.edges())
        current_nodes = stats_collector.stats.final_node_count
        current_edges = stats_collector.stats.final_edge_count
        stats_collector.update_graph_size(current_nodes + num_nodes, current_edges + num_edges)
    
    print(f"\n[处理] 完成，共处理 {processed_count} 条记录")
    
    return knowledge_summaries


def merge_summaries(entity_summaries: Dict[str, str], 
                   knowledge_point_summaries: Dict[str, str]) -> Dict[str, str]:
    """
    合并两个图谱的知识总结
    """
    merged_summaries = {}
    
    # 获取所有时间戳
    all_timestamps = set(entity_summaries.keys()) | set(knowledge_point_summaries.keys())
    
    for timestamp in sorted(all_timestamps):
        entity_summary = entity_summaries.get(timestamp, "")
        kp_summary = knowledge_point_summaries.get(timestamp, "")
        
        merged = f"""## Entity Graph Summary
{entity_summary}

## Knowledge Point Graph Summary
{kp_summary}
"""
        merged_summaries[timestamp] = merged
    
    return merged_summaries


def main():
    parser = argparse.ArgumentParser(description="Our Data Preparation")
    parser.add_argument("--csv_path", required=True, help="CSV数据文件路径")
    parser.add_argument("--output_dir", default="./our", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="先构建数据比例")
    parser.add_argument("--top_k", type=int, default=10, help="知识总结中显示的前K个知识点")
    parser.add_argument("--embedding_base_url", default=os.getenv("EMBEDDING_BASE_URL", ""), help="Embedding service base URL")

    parser.add_argument("--embedding_api_key", default=os.getenv("EMBEDDING_API_KEY", ""), help="Embedding service API key")
    parser.add_argument("--embedding_model", default=os.getenv("EMBEDDING_MODEL", "conan"), help="Embedding model name")
    parser.add_argument("--embedding_dim", type=int, default=int(os.getenv("EMBEDDING_DIM", "1792")), help="Embedding vector dimension")
    parser.add_argument("--embedding_timeout", type=float, default=float(os.getenv("EMBEDDING_TIMEOUT", "120")), help="Embedding request timeout")
    parser.add_argument("--embedding_batch_size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")), help="Embedding batch size")
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BASELINE_BATCH_SIZE", "10")), help="Number of records per GraphRAG insert batch")
    parser.add_argument("--llm_api_key", default=os.getenv("LLM_API_KEY"), help="LLM API key")
    parser.add_argument("--llm_base_url", default=os.getenv("LLM_BASE_URL"), help="LLM base URL")
    parser.add_argument("--llm_model", default=os.getenv("LLM_MODEL", ""), help="LLM model name")
    parser.add_argument("--output_model_name", default=None, help="Output directory model name (default: same as llm_model)")
    parser.add_argument("--llm_timeout", type=float, default=float(os.getenv("LLM_TIMEOUT", "60")), help="LLM request timeout (seconds)")
    parser.add_argument("--insert_timeout", type=float, default=None, help="Graph insert operation timeout (seconds)")
    parser.add_argument("--llm_max_retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "3")), help="Maximum retries")
    parser.add_argument("--llm_retry_delay", type=float, default=float(os.getenv("LLM_RETRY_DELAY", "5")), help="Initial retry delay")
    parser.add_argument("--llm_retry_backoff", type=float, default=float(os.getenv("LLM_RETRY_BACKOFF", "2")), help="Retry backoff multiplier")
    parser.add_argument("--pg_host", default=os.getenv("PGVECTOR_HOST", ""), help="PGVector host")
    parser.add_argument("--pg_port", type=int, default=int(os.getenv("PGVECTOR_PORT", "")), help="PGVector port")
    parser.add_argument("--pg_database", default=os.getenv("PGVECTOR_DB", ""), help="PGVector database name")
    parser.add_argument("--pg_user", default=os.getenv("PGVECTOR_USER"), help="PGVector user")
    parser.add_argument("--pg_password", default=os.getenv("PGVECTOR_PASSWORD"), help="PGVector password")
    parser.add_argument("--pg_table", default=os.getenv("PGVECTOR_TABLE", "student_kg_entities"), help="PGVector table name")
    parser.add_argument("--pg_dsn", default=os.getenv("PGVECTOR_DSN"), help="PG connection DSN")
    parser.add_argument("--temporal_mode", action="store_true", help="Enable temporal knowledge graph mode")
    parser.add_argument("--snapshot_interval", type=int, default=1, help="Interval for generating knowledge summary snapshots")
    parser.add_argument("--enable_community_report", action="store_true", help="Whether to generate GraphRAG community reports (default: disabled to avoid long LLM calls)")
    parser.add_argument("--program_chunk_chars", type=int, default=int(os.getenv("PROGRAM_CHUNK_CHARS", "5000")), help="Maximum characters per student program chunk when building prompts")
    parser.add_argument("--clean_cache", action="store_true", help="Clean graph cache before building (removes existing graph_cache directories)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_model_name = args.output_model_name if args.output_model_name else args.llm_model
    model_suffix = output_model_name.replace("/", "_").replace("\\", "_")
    base_output_dir = args.output_dir  # 例如: baseline_output/acl_single
    
    csv_basename = os.path.splitext(os.path.basename(args.csv_path))[0]
    entity_output_dir = os.path.join(base_output_dir, f"graph_{csv_basename}", f"model_{model_suffix}")
    os.makedirs(entity_output_dir, exist_ok=True)
    
    current_output_dir = os.path.join(base_output_dir, f"student_{csv_basename}", f"model_{model_suffix}")
    os.makedirs(current_output_dir, exist_ok=True)
    args.output_dir = current_output_dir
    
    stats_collector = GraphBuildStatsCollector(
        version="optimized",
        model_name=args.llm_model
    )
    

    # 加载数据
    loader = EduDataLoader(args.csv_path)
    loader.load_data()
    ability = determine_ability(args.csv_path)
    
    if "时间" in loader.data.columns:
        try:
            loader.data["时间"] = pd.to_datetime(loader.data["时间"])
            loader.data = loader.data.sort_values(by="时间").reset_index(drop=True)
        except Exception as e:
            print(f"[Warning] Failed to sort by time: {e}")
    
    # 划分先构建数据和第二次构建的数据
    train_df, test_df = loader.split_by_ratio(args.train_ratio)
    
    # 格式化数据
    formatted_train = loader.format_dataframe(train_df, knowledge_points=None)
    formatted_test = loader.format_dataframe(test_df, knowledge_points=None)
    
    formatted_all = formatted_train + formatted_test
    if formatted_all:
        # 检查是否有时间字段（可能是"时间"或"timestamp"）
        first_record = formatted_all[0]
        time_key = None
        if "时间" in first_record:
            time_key = "时间"
        elif "timestamp" in first_record:
            time_key = "timestamp"
        
        if time_key:
            formatted_all.sort(key=lambda x: x.get(time_key, ""))

    local_llm = LLMClient(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout=args.llm_timeout,
        enable_thinking=False  # 关闭 thinking 以提升速度（使用非流式调用）
    )
    cache_manager = CacheManager(cache_dir=os.path.join(args.output_dir, "llm_cache"))
    
    async def local_llm_async(prompt: str, system_prompt: str = None, history_messages=None, **kwargs):
        import asyncio
        message_parts = []
        if history_messages:
            for msg in history_messages:
                message_parts.append(msg.get("content", ""))
        if system_prompt:
            message_parts.append(system_prompt)
        message_parts.append(prompt)
        final_prompt = "\n\n".join(part for part in message_parts if part)
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 4000)
        cached_response = cache_manager.get_llm_response(final_prompt, local_llm.model, temperature, max_tokens)
        if cached_response is not None:
            return cached_response
        delay = args.llm_retry_delay
        for attempt in range(1, args.llm_max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    local_llm.call_llm_async(final_prompt, temperature=temperature, max_tokens=max_tokens),
                    timeout=args.llm_timeout,
                )
                cache_manager.save_llm_response(final_prompt, local_llm.model, temperature, max_tokens, response)
                return response
            except Exception as exc:
                import asyncio as _asyncio_mod
                error_msg = str(exc)
                error_type = type(exc).__name__
                is_timeout_error = (
                    isinstance(exc, _asyncio_mod.TimeoutError)
                    or "504" in error_msg
                    or "Gateway Timeout" in error_msg
                    or "timeout" in error_msg.lower()
                    or "<!DOCTYPE html>" in error_msg
                    or "nginx" in error_msg.lower()
                    or "unavailable" in error_msg.lower()
                    or "InternalServerError" in error_type
                )
                if is_timeout_error:
                    raise
                if attempt >= args.llm_max_retries:
                    raise
                else:
                    await asyncio.sleep(delay)
                    delay *= max(1.0, args.llm_retry_backoff)
    
    embedding_func = create_conan_embedding_func(
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        model_name=args.embedding_model,
        embedding_dim=args.embedding_dim,
        timeout=args.embedding_timeout,
    )
    
    model_suffix_safe = model_suffix.replace("-", "_").replace(".", "_")
    entity_pg_table_name = f"{args.pg_table}_{model_suffix_safe}" if args.pg_table else f"student_kg_entities_{model_suffix_safe}"
    
    if args.pg_dsn:
        entity_pg_kwargs = {
            "dsn": args.pg_dsn,
            "table_name": entity_pg_table_name,
            "embedding_dim": args.embedding_dim,
        }
    else:
        missing_keys = [
            key for key, value in (("host", args.pg_host), ("dbname", args.pg_database), ("user", args.pg_user))
            if not value
        ]
        if missing_keys:
            raise ValueError(f"PGVector 连接配置缺少字段: {missing_keys}")
        entity_pg_kwargs = {
            "host": args.pg_host,
            "port": args.pg_port,
            "dbname": args.pg_database,
            "user": args.pg_user,
            "password": args.pg_password,
            "table_name": entity_pg_table_name,
            "embedding_dim": args.embedding_dim,
        }
    
    # 禁用社区报告
    if not args.enable_community_report:
        try:
            from nano_graphrag import _op as graphrag_ops
        except ImportError:
            graphrag_ops = None
        if graphrag_ops is not None:
            async def _noop_generate_community_report(community_report_kv, knwoledge_graph_inst, global_config):
                return
            graphrag_ops.generate_community_report = _noop_generate_community_report
    
    if args.clean_cache:
        import shutil
        entity_cache_dir = os.path.join(entity_output_dir, "graph_cache")
        if os.path.exists(entity_cache_dir):
            shutil.rmtree(entity_cache_dir)
            print(f"[清理缓存] 已删除: {entity_cache_dir}")
    
    entity_graph = GraphRAG(
        working_dir=entity_output_dir,
        best_model_func=local_llm_async,
        cheap_model_func=local_llm_async,
        embedding_func=embedding_func,
        embedding_batch_num=args.embedding_batch_size,
        vector_db_storage_cls=PGVectorStorage,
        vector_db_storage_cls_kwargs=entity_pg_kwargs,
        enable_llm_cache=True,
        best_model_max_token_size=32768,  
        convert_response_to_json_func=convert_response_to_json_with_fix,
    )
    
    # 禁用聚类
    DISABLE_CLUSTERING = True
    
    if DISABLE_CLUSTERING:
        async def noop_clustering(algorithm: str):
            print("已禁用聚类")
            return

        entity_graph.chunk_entity_relation_graph.clustering = noop_clustering

    else:
        if stats_collector:
            original_entity_clustering = entity_graph.chunk_entity_relation_graph.clustering
            entity_graph.chunk_entity_relation_graph.clustering = wrap_clustering_for_stats(original_entity_clustering, stats_collector)
    
    try:
        entity_summaries = build_entity_graph_simple(
            graph=entity_graph,
            formatted_train=formatted_all, 
            output_dir=entity_output_dir,
            args=args,
            stats_collector=stats_collector
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        entity_summaries = {}
    
    train_data = records_from_dataframe(
        train_df,
        ability=ability,
        knowledge_summaries=entity_summaries
    )
    
    test_data = records_from_dataframe(
        test_df,
        ability=ability,
        knowledge_summaries=entity_summaries
    )
    
    train_file = os.path.join(args.output_dir, "train.jsonl")
    test_file = os.path.join(args.output_dir, "test.jsonl")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for record in train_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    with open(test_file, "w", encoding="utf-8") as f:
        for record in test_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    stats_collector.save(args.output_dir, "build_stats.json")
    
if __name__ == "__main__":
    main()

