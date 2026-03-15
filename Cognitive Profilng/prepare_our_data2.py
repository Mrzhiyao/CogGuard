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
    compute_mdhash_id,
    chunking_by_chars,
    compute_chunk_ids_from_content,
    determine_ability,
    label_from_row,
    build_training_strings,
    summarise_knowledge,
    records_from_dataframe
)
from graph_build_stats import GraphBuildStatsCollector

GRAPH_CONNECTOR = "-" * 10000


def fix_json_escape_errors(json_str: str) -> str:
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
    fixed = json_str
    
    invalid_escape_pattern = r'\\(?!["\\/bfnrtu])'
    replacement = r'\\\\'
    
    def fix_backslashes_in_strings(match):
        content = match.group(1)
        fixed_content = re.sub(invalid_escape_pattern, replacement, content)
        return f'"{fixed_content}"'
    
    string_pattern = r'"([^"]*(?:\\.[^"]*)*)"'
    fixed = re.sub(string_pattern, fix_backslashes_in_strings, fixed)
    
    fixed = re.sub(r'\\([Uu])(?![0-9A-Fa-f]{4,8})', r'\\\\\1', fixed)
    
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


def build_common_prompt_prefix(question: str, desc: str, program: str, error_desc: str) -> str:

    return NEW_PROMPTS['get_library'].format(
        question=question,
        desc=desc,
        program=program,
        error_desc=error_desc
    )


def build_entity_prompt(common_prefix: str, unique_suffix: str) -> str:

    return common_prefix + unique_suffix


def build_knowledge_point_prompt(common_prefix: str, unique_suffix: str, knowledge_points: List[str] = None) -> str:

    if knowledge_points and len(knowledge_points) > 0:
        kp_display = ', '.join(knowledge_points)
        if len(kp_display) > 500:
            kp_display = ', '.join(knowledge_points[:20]) + f" ... (and {len(knowledge_points) - 20} more)"
        
        # 在统一前缀后追加知识点特定指令
        kp_instructions = f"""

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
        return common_prefix + kp_instructions + unique_suffix
    else:
        # 如果没有知识点列表，使用与实体图相同的 prompt
        return common_prefix + unique_suffix


def wrap_clustering_for_stats(original_clustering_method, stats_collector: GraphBuildStatsCollector):

    async def wrapped_clustering(algorithm: str):
        start_time = time.time()
        try:
            result = await original_clustering_method(algorithm)
            elapsed_time = time.time() - start_time
            if stats_collector:
                stats_collector.record_clustering(elapsed_time)
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            if stats_collector:
                stats_collector.record_clustering(elapsed_time)
            raise e
    
    return wrapped_clustering


def split_program_into_chunks(program: str, max_chars: int) -> List[str]:

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


async def process_entity_graph_async(
    entity_graph: GraphRAG,
    formatted_train: List[Dict],
    entity_output_dir: str,
    args,
    stats_collector: GraphBuildStatsCollector = None
) -> Dict[str, str]:

    entity_summaries = {}
    entity_chunk_timestamp_map = {}
    entity_progress_file = os.path.join(entity_output_dir, "entity_progress.json")
    
    # 加载已有的知识总结（不加载处理索引，避免不同 CSV 之间的索引混乱）
    # 实体图谱是全局共享的，GraphRAG 会自动去重，所以不需要断点续传
    if os.path.exists(entity_progress_file):
        try:
            with open(entity_progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                # 只加载知识总结和 chunk 映射，不加载 last_processed_idx
                entity_summaries = progress_data.get("knowledge_summaries", {})
                entity_chunk_timestamp_map = progress_data.get("chunk_timestamp_map", {})
                print(f"[实体图] 加载已有知识总结: {len(entity_summaries)} 条")
        except Exception as e:
            print(f"[实体图] 警告: 加载知识总结失败: {e}")
    
    total_records = len(formatted_train)
    
    for idx, record in enumerate(formatted_train):
        
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
        
        program_chunks = split_program_into_chunks(program, args.program_chunk_chars)
        chunk_count = len(program_chunks)
        
        for seg_idx, program_chunk in enumerate(program_chunks, start=1):
            if chunk_count > 1:
                segment_program = f"[Program Segment {seg_idx}/{chunk_count}]\n{program_chunk}"
            else:
                segment_program = program
            
            common_prefix = build_common_prompt_prefix(question, desc, segment_program, error_desc)
            unique_suffix = f"\n\n[Record ID: {idx + 1}-{seg_idx}, Timestamp: {timestamp}]"
            entity_prompt = build_entity_prompt(common_prefix, unique_suffix)
            
            print(f"[实体图] 插入记录 {idx + 1}-{seg_idx}/{chunk_count}...")
            start_time = time.time()
            
            try:
                if args.insert_timeout:
                    await asyncio.wait_for(
                        entity_graph.ainsert(entity_prompt),
                        timeout=args.insert_timeout
                    )
                else:
                    await entity_graph.ainsert(entity_prompt)
                
                elapsed_time = time.time() - start_time
                print(f"[实体图] 记录 {idx + 1}-{seg_idx} 完成，耗时 {elapsed_time:.2f} 秒")
                
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=True)
                    stats_collector.record_llm_call(elapsed_time * 0.8, success=True)
                
                chunk_ids = compute_chunk_ids_from_content(entity_prompt)
                for chunk_id in chunk_ids:
                    entity_chunk_timestamp_map[chunk_id] = unique_timestamp
                    
            except asyncio.TimeoutError:
                elapsed_time = time.time() - start_time
                print(f"[实体图] 记录 {idx + 1}-{seg_idx} 超时，跳过此段")
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=False)
                    stats_collector.record_llm_call(elapsed_time, success=False, is_timeout=True)
                continue
            except Exception as e:
                elapsed_time = time.time() - start_time
                error_msg = str(e)
                error_type = type(e).__name__
                if "JSONDecodeError" in error_type or ("Invalid" in error_msg and "escape" in error_msg):
                    print(f"[实体图] 记录 {idx + 1}-{seg_idx} JSON 解析错误，跳过此段")
                else:
                    print(f"[实体图] 记录 {idx + 1}-{seg_idx} 插入失败 ({error_type}): {error_msg[:200]}...")
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=False)
                    stats_collector.record_llm_call(elapsed_time, success=False)
                continue
        
        if (idx + 1) % args.snapshot_interval == 0 or idx == len(formatted_train) - 1:
            entity_num_nodes = len(entity_graph.chunk_entity_relation_graph._graph.nodes())
            entity_num_edges = len(entity_graph.chunk_entity_relation_graph._graph.edges())
            
            try:
                entity_summary = summarise_knowledge(
                    entity_graph,
                    top_k=args.top_k,
                    record_count=idx + 1,
                    chunk_timestamp_map=entity_chunk_timestamp_map,
                    current_timestamp=unique_timestamp
                )
                entity_summaries[unique_timestamp] = entity_summary
            except Exception as e:
                print(f"[实体图] 警告: 生成知识总结失败: {e}")
                entity_summaries[unique_timestamp] = f"生成知识总结失败: {e}"
        

        try:
            entity_progress_data = {
                "knowledge_summaries": entity_summaries,
                "chunk_timestamp_map": entity_chunk_timestamp_map,
                "note": "实体图谱是全局共享的，不保存处理索引，GraphRAG 会自动去重"
            }
            with open(entity_progress_file, "w", encoding="utf-8") as f:
                json.dump(entity_progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[实体图] 警告: 保存知识总结失败: {e}")
        
        if stats_collector and (idx + 1) % 10 == 0:
            try:
                pass
            except Exception as e:
                print(f"[实体图] 警告: 保存统计信息失败: {e}")
    
    return entity_summaries


async def process_knowledge_point_graph_async(
    kp_graph: GraphRAG,
    formatted_train: List[Dict],
    kp_output_dir: str,
    args,
    knowledge_points: List[str] = None,
    stats_collector: GraphBuildStatsCollector = None
) -> Dict[str, str]:

    kp_summaries = {}
    kp_chunk_timestamp_map = {}
    kp_progress_file = os.path.join(kp_output_dir, "knowledge_point_progress.json")
    start_idx = 0
    
    if os.path.exists(kp_progress_file):
        try:
            with open(kp_progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                kp_summaries = progress_data.get("knowledge_summaries", {})
                kp_chunk_timestamp_map = progress_data.get("chunk_timestamp_map", {})
                start_idx = progress_data.get("last_processed_idx", 0)
                print(f"[知识图] 找到已有进度: 已处理 {start_idx}/{len(formatted_train)} 条记录")
        except Exception as e:
            print(f"[知识图] 警告: 加载进度失败: {e}")
    
    total_records = len(formatted_train)
    
    for idx, record in enumerate(formatted_train):
        if idx < start_idx:
            continue
        
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
        
        program_chunks = split_program_into_chunks(program, args.program_chunk_chars)
        chunk_count = len(program_chunks)
        
        for seg_idx, program_chunk in enumerate(program_chunks, start=1):
            if chunk_count > 1:
                segment_program = f"[Program Segment {seg_idx}/{chunk_count}]\n{program_chunk}"
            else:
                segment_program = program
            
            common_prefix = build_common_prompt_prefix(question, desc, segment_program, error_desc)
            unique_suffix = f"\n\n[Record ID: {idx + 1}-{seg_idx}, Timestamp: {timestamp}]"
            kp_prompt = build_knowledge_point_prompt(common_prefix, unique_suffix, knowledge_points)
            
            print(f"[知识图] 插入记录 {idx + 1}-{seg_idx}/{chunk_count}...")
            start_time = time.time()
            
            try:
                if args.insert_timeout:
                    await asyncio.wait_for(
                        kp_graph.ainsert(kp_prompt),
                        timeout=args.insert_timeout
                    )
                else:
                    await kp_graph.ainsert(kp_prompt)
                
                elapsed_time = time.time() - start_time
                print(f"[知识图] 记录 {idx + 1}-{seg_idx} 完成，耗时 {elapsed_time:.2f} 秒")
                
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=True)
                    stats_collector.record_llm_call(elapsed_time * 0.8, success=True)
                
                chunk_ids = compute_chunk_ids_from_content(kp_prompt)
                for chunk_id in chunk_ids:
                    kp_chunk_timestamp_map[chunk_id] = unique_timestamp
                    
            except asyncio.TimeoutError:
                elapsed_time = time.time() - start_time
                print(f"[知识图] 记录 {idx + 1}-{seg_idx} 超时，跳过此段")
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=False)
                    stats_collector.record_llm_call(elapsed_time, success=False, is_timeout=True)
                continue
            except Exception as e:
                elapsed_time = time.time() - start_time
                error_msg = str(e)
                error_type = type(e).__name__
                if "JSONDecodeError" in error_type or ("Invalid" in error_msg and "escape" in error_msg):
                    print(f"[知识图] 记录 {idx + 1}-{seg_idx} JSON 解析错误，跳过此段")
                else:
                    print(f"[知识图] 记录 {idx + 1}-{seg_idx} 插入失败 ({error_type}): {error_msg[:200]}...")
                if stats_collector:
                    stats_collector.record_graph_insert(elapsed_time, success=False)
                    stats_collector.record_llm_call(elapsed_time, success=False)
                continue
        
        if (idx + 1) % args.snapshot_interval == 0 or idx == len(formatted_train) - 1:
            kp_num_nodes = len(kp_graph.chunk_entity_relation_graph._graph.nodes())
            kp_num_edges = len(kp_graph.chunk_entity_relation_graph._graph.edges())
            
            try:
                kp_summary = summarise_knowledge(
                    kp_graph,
                    top_k=args.top_k,
                    record_count=idx + 1,
                    chunk_timestamp_map=kp_chunk_timestamp_map,
                    current_timestamp=unique_timestamp
                )
                kp_summaries[unique_timestamp] = kp_summary
            except Exception as e:
                kp_summaries[unique_timestamp] = f"生成知识总结失败: {e}"
        
        try:
            kp_progress_data = {
                "last_processed_idx": idx,
                "knowledge_summaries": kp_summaries,
                "chunk_timestamp_map": kp_chunk_timestamp_map,
                "total_records": len(formatted_train)
            }
            with open(kp_progress_file, "w", encoding="utf-8") as f:
                json.dump(kp_progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[知识图] 警告: 保存进度失败: {e}")
    
    return kp_summaries


def build_double_graph_parallel(
    entity_graph: GraphRAG,
    kp_graph: GraphRAG,
    formatted_train: List[Dict],
    entity_output_dir: str,
    kp_output_dir: str,
    args,
    knowledge_points: List[str] = None,
    stats_collector: GraphBuildStatsCollector = None
) -> Tuple[Dict[str, str], Dict[str, str]]:

    total_records = len(formatted_train)
    if stats_collector:
        stats_collector.update_record_counts(total_records, 0)
    
    # 使用 asyncio.gather 让两个图独立并行处理
    async def run_parallel():
        entity_task = process_entity_graph_async(
            entity_graph, formatted_train, entity_output_dir, args, stats_collector
        )
        kp_task = process_knowledge_point_graph_async(
            kp_graph, formatted_train, kp_output_dir, args, knowledge_points, stats_collector
        )
        return await asyncio.gather(entity_task, kp_task)
    
    entity_summaries, kp_summaries = asyncio.run(run_parallel())
    
    if stats_collector:
        entity_num_nodes = len(entity_graph.chunk_entity_relation_graph._graph.nodes())
        entity_num_edges = len(entity_graph.chunk_entity_relation_graph._graph.edges())
        kp_num_nodes = len(kp_graph.chunk_entity_relation_graph._graph.nodes())
        kp_num_edges = len(kp_graph.chunk_entity_relation_graph._graph.edges())
        stats_collector.update_graph_size(entity_num_nodes + kp_num_nodes, entity_num_edges + kp_num_edges)
    
    return entity_summaries, kp_summaries


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
    parser.add_argument("--knowledge_points_file", default=None, help="预定义知识点文件路径（可选）")
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
    
    model_suffix = args.llm_model.replace("/", "_").replace("\\", "_")
    base_output_dir = args.output_dir  # 例如: baseline_output/acl_double
    

    csv_basename = os.path.splitext(os.path.basename(args.csv_path))[0]
    kp_output_dir = os.path.join(base_output_dir, f"knowledge_point_graph_{csv_basename}", f"model_{model_suffix}")
    os.makedirs(kp_output_dir, exist_ok=True)
    
    if args.global_entity_graph_dir:
        if os.path.isabs(args.global_entity_graph_dir):
            entity_output_dir = os.path.join(args.global_entity_graph_dir, f"model_{model_suffix}")
        else:
            entity_output_dir = os.path.join(base_output_dir, args.global_entity_graph_dir, f"model_{model_suffix}")
        if not os.path.exists(entity_output_dir):
            os.makedirs(entity_output_dir, exist_ok=True)
        else:
            print(f"[实体图谱] 加载已有全局实体图谱")
    else:
        entity_output_dir = os.path.join(base_output_dir, "entity_graph_global", f"model_{model_suffix}")
        os.makedirs(entity_output_dir, exist_ok=True)
    
    current_output_dir = os.path.join(base_output_dir, f"student_{csv_basename}", f"model_{model_suffix}")
    os.makedirs(current_output_dir, exist_ok=True)
    args.output_dir = current_output_dir
    
    stats_collector = GraphBuildStatsCollector(
        version="optimized",
        model_name=args.llm_model
    )
    
    loader = EduDataLoader(args.csv_path)
    loader.load_data()
    ability = determine_ability(args.csv_path)
    
    if args.knowledge_points_file:
        knowledge_points = loader.get_knowledge_points_from_file(args.knowledge_points_file)
        print(f"\n读取了 {len(knowledge_points)} 个预定义知识点")
    else:
        knowledge_points = []
    
    # 确保数据按时间排序
    if "时间" in loader.data.columns:
        try:
            loader.data["时间"] = pd.to_datetime(loader.data["时间"])
            loader.data = loader.data.sort_values(by="时间").reset_index(drop=True)
        except Exception as e:
            print(f"[Warning] Failed to sort by time: {e}")
    
    print(f"能力等级: {ability}")
    

    train_df, test_df = loader.split_by_ratio(args.train_ratio)
    
    
    # 格式化数据
    formatted_train = loader.format_dataframe(train_df, knowledge_points=None)
    formatted_test = loader.format_dataframe(test_df, knowledge_points=None)
    
    formatted_all = formatted_train + formatted_test
    # 按时间戳排序（如果数据中有时间字段）
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

    # 初始化 LLM 客户端和缓存管理器
    local_llm = LLMClient(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        timeout=args.llm_timeout
    )
    cache_manager = CacheManager(cache_dir=os.path.join(args.output_dir, "llm_cache"))
    
    # 定义异步 LLM 调用函数
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
    entity_pg_table_name = f"{args.pg_table}_entity_{model_suffix_safe}" if args.pg_table else f"student_kg_entities_entity_{model_suffix_safe}"
    kp_pg_table_name = f"{args.pg_table}_kp_{model_suffix_safe}" if args.pg_table else f"student_kg_entities_kp_{model_suffix_safe}"
    
    if args.pg_dsn:
        entity_pg_kwargs = {
            "dsn": args.pg_dsn,
            "table_name": entity_pg_table_name,
            "embedding_dim": args.embedding_dim,
        }
        kp_pg_kwargs = {
            "dsn": args.pg_dsn,
            "table_name": kp_pg_table_name,
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
        kp_pg_kwargs = {
            "host": args.pg_host,
            "port": args.pg_port,
            "dbname": args.pg_database,
            "user": args.pg_user,
            "password": args.pg_password,
            "table_name": kp_pg_table_name,
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
                print("[Baseline] Community report generation disabled (noop).")
                return
            graphrag_ops.generate_community_report = _noop_generate_community_report
            print("[Baseline] Community report generation已关闭（可通过 --enable_community_report 开启）")
    
    if args.clean_cache:
        import shutil
        entity_cache_dir = os.path.join(entity_output_dir, "graph_cache")
        kp_cache_dir = os.path.join(kp_output_dir, "graph_cache")
        if os.path.exists(entity_cache_dir):
            shutil.rmtree(entity_cache_dir)
            print(f"[清理缓存] 已删除: {entity_cache_dir}")
        if os.path.exists(kp_cache_dir):
            shutil.rmtree(kp_cache_dir)
            print(f"[清理缓存] 已删除: {kp_cache_dir}")
    

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
    

    kp_graph = GraphRAG(
        working_dir=kp_output_dir,
        best_model_func=local_llm_async,
        cheap_model_func=local_llm_async,
        embedding_func=embedding_func,
        embedding_batch_num=args.embedding_batch_size,
        vector_db_storage_cls=PGVectorStorage,
        vector_db_storage_cls_kwargs=kp_pg_kwargs,
        enable_llm_cache=True,
        best_model_max_token_size=32768, 
        convert_response_to_json_func=convert_response_to_json_with_fix,  
    )
    
    # 临时禁用聚类
    DISABLE_CLUSTERING = True  # 设置为 True 禁用聚类，False 启用聚类
    
    if DISABLE_CLUSTERING:
        async def noop_clustering(algorithm: str):
            return
        
        entity_graph.chunk_entity_relation_graph.clustering = noop_clustering
        kp_graph.chunk_entity_relation_graph.clustering = noop_clustering
    else:
        if stats_collector:
            original_entity_clustering = entity_graph.chunk_entity_relation_graph.clustering
            entity_graph.chunk_entity_relation_graph.clustering = wrap_clustering_for_stats(original_entity_clustering, stats_collector)
            
            original_kp_clustering = kp_graph.chunk_entity_relation_graph.clustering
            kp_graph.chunk_entity_relation_graph.clustering = wrap_clustering_for_stats(original_kp_clustering, stats_collector)
    
    try:
        entity_summaries, knowledge_point_summaries = build_double_graph_parallel(
            entity_graph=entity_graph,
            kp_graph=kp_graph,
            formatted_train=formatted_all,  # 使用合并后的数据
            entity_output_dir=entity_output_dir,
            kp_output_dir=kp_output_dir,
            args=args,
            knowledge_points=knowledge_points,
            stats_collector=stats_collector
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        entity_summaries = {}
        knowledge_point_summaries = {}
    
    merged_summaries = merge_summaries(entity_summaries, knowledge_point_summaries)
    
    train_data = records_from_dataframe(
        train_df,
        ability=ability,
        knowledge_summaries=merged_summaries
    )
    
    test_data = records_from_dataframe(
        test_df,
        ability=ability,
        knowledge_summaries=merged_summaries
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

