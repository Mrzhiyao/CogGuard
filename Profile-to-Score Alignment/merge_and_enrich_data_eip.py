#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并多个 train.jsonl 文件，并补全 entity 和 knowledge 总结
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# 尝试导入 jsonlines
try:
    import jsonlines
    HAS_JSONLINES = True
except ImportError:
    HAS_JSONLINES = False


def find_all_data_files(base_dir: str, data_type: str = "train", max_students: int = None, student_ids: List[str] = None) -> List[str]:
    """
    查找所有 student_*_practice 目录下的指定类型数据文件
    
    Args:
        base_dir: 基础目录路径
        data_type: 数据类型，可选值: "train", "test", "validation", "val", "all"
        max_students: 最大学生数量限制（None 表示不限制）
        
    Returns:
        数据文件路径列表
    """
    # 支持多种文件名
    if data_type == "val":
        data_type = "validation"
    
    # 可能的文件名
    possible_names = [f"{data_type}.jsonl"]
    if data_type == "validation":
        possible_names.append("val.jsonl")
    elif data_type == "all":
        # all 类型优先查找 all.jsonl 文件，如果找不到则查找 train.jsonl 和 test.jsonl
        possible_names = ["all.jsonl", "train.jsonl", "test.jsonl"]
    
    all_files = []
    for name in possible_names:
        pattern = os.path.join(base_dir, "student_*_practice", "**", name)
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    # 去重并排序
    all_files = sorted(list(set(all_files)))
    
    # 如果指定了学生ID列表，只取指定的学生
    if student_ids is not None and len(student_ids) > 0:
        # 按学生目录分组
        student_files = {}
        for file_path in all_files:
            # 提取学生ID
            parts = file_path.split(os.sep)
            for part in parts:
                if part.startswith("student_") and part.endswith("_practice"):
                    student_id = part.replace("student_", "").replace("_practice", "")
                    if student_id in student_ids:
                        if student_id not in student_files:
                            student_files[student_id] = []
                        student_files[student_id].append(file_path)
                    break
        
        # 按指定的学生ID顺序排列
        all_files = []
        for student_id in student_ids:
            if student_id in student_files:
                all_files.extend(student_files[student_id])
    # 如果指定了最大学生数量，只取前N个
    elif max_students is not None and max_students > 0:
        # 按学生目录分组
        student_files = {}
        for file_path in all_files:
            # 提取学生ID
            parts = file_path.split(os.sep)
            for part in parts:
                if part.startswith("student_") and part.endswith("_practice"):
                    student_id = part
                    if student_id not in student_files:
                        student_files[student_id] = []
                    student_files[student_id].append(file_path)
                    break
        
        # 按学生ID排序，只取前N个学生
        sorted_students = sorted(student_files.keys())[:max_students]
        all_files = []
        for student_id in sorted_students:
            all_files.extend(student_files[student_id])
    
    return all_files


def find_all_train_files(base_dir: str) -> List[str]:
    """
    查找所有 student_*_practice 目录下的 train.jsonl 文件（向后兼容）
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        train.jsonl 文件路径列表
    """
    return find_all_data_files(base_dir, "train")


def extract_user_id_from_path(file_path: str) -> Optional[str]:
    """
    从文件路径中提取 user_id
    
    Args:
        file_path: 文件路径，例如 .../student_64904597_practice/.../train.jsonl
        
    Returns:
        user_id 字符串，例如 "64904597"
    """
    parts = file_path.split(os.sep)
    for part in parts:
        if part.startswith("student_") and part.endswith("_practice"):
            user_id = part.replace("student_", "").replace("_practice", "")
            return user_id
    return None


def load_entity_progress(entity_progress_file: str) -> Dict:
    """
    加载 entity_progress.json 文件
    
    Args:
        entity_progress_file: entity_progress.json 文件路径
        
    Returns:
        包含 knowledge_summaries 的字典
    """
    if not os.path.exists(entity_progress_file):
        print(f"警告: entity_progress.json 不存在: {entity_progress_file}")
        return {"knowledge_summaries": {}}
    
    with open(entity_progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_knowledge_progress(knowledge_progress_file: str) -> Dict:
    """
    加载 knowledge_point_progress.json 文件
    
    Args:
        knowledge_progress_file: knowledge_point_progress.json 文件路径
        
    Returns:
        包含 knowledge_summaries 的字典
    """
    if not os.path.exists(knowledge_progress_file):
        print(f"警告: knowledge_point_progress.json 不存在: {knowledge_progress_file}")
        return {"knowledge_summaries": {}}
    
    with open(knowledge_progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_matching_summary(timestamp: str, summaries: Dict, record_id: int = 1, use_nearest: bool = True) -> Optional[str]:
    """
    根据 timestamp 查找匹配的总结
    
    Args:
        timestamp: 时间戳字符串，例如 "2022-11-02 15:08:36"
        summaries: knowledge_summaries 字典
        record_id: 记录 ID，默认为 1
        use_nearest: 如果精确匹配失败，是否使用最接近的时间戳（用于测试集）
        
    Returns:
        匹配的总结文本，如果找不到则返回 None
    """
    # 尝试精确匹配
    key = f"{timestamp}_record_{record_id}"
    if key in summaries:
        return summaries[key]
    
    # 尝试查找所有包含该 timestamp 的 key
    matching_keys = [k for k in summaries.keys() if timestamp in k]
    if matching_keys:
        # 返回第一个匹配的（通常应该只有一个）
        return summaries[matching_keys[0]]
    
    # 如果精确匹配失败且允许使用最接近的时间戳
    if use_nearest and summaries:
        from datetime import datetime
        
        try:
            # 解析目标时间戳
            target_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            
            # 找到最接近的时间戳
            nearest_key = None
            min_diff = None
            
            for key in summaries.keys():
                try:
                    # 从 key 中提取时间戳: "2022-03-26 14:49:07_record_1"
                    key_timestamp = key.split('_record_')[0]
                    key_time = datetime.strptime(key_timestamp, "%Y-%m-%d %H:%M:%S")
                    
                    # 只考虑早于或等于目标时间的时间戳（测试集应该使用训练集的历史数据）
                    if key_time <= target_time:
                        diff = (target_time - key_time).total_seconds()
                        if min_diff is None or diff < min_diff:
                            min_diff = diff
                            nearest_key = key
                except (ValueError, IndexError):
                    continue
            
            if nearest_key:
                return summaries[nearest_key]
        except ValueError:
            # 时间戳格式错误，跳过
            pass
    
    return None


def format_e_field(entity_summary: str) -> str:
    """
    格式化 e 字段（单图谱版本，只包含实体图）
    
    Args:
        entity_summary: 实体图总结
        
    Returns:
        格式化后的 e 字段文本
    """
    # 清理总结文本
    entity_summary = entity_summary.strip() if entity_summary else ""
    
    # 如果 entity_summary 为空
    if not entity_summary or "No knowledge points extracted from the graph" in entity_summary:
        entity_summary = "Student Knowledge Summary:\nNo knowledge points extracted from the graph."
    # 确保 entity_summary 以 "Student Knowledge Summary:" 开头
    elif not entity_summary.startswith("Student Knowledge Summary"):
        entity_summary = "Student Knowledge Summary:\n" + entity_summary
    
    # 单图谱版本：只返回实体图
    return entity_summary


def enrich_record(
    record: Dict,
    user_id: str,
    base_dir: str,
    entity_progress_cache: Dict = None,
    is_test: bool = False
) -> Dict:
    """
    补全单条记录的实体图总结（单图谱版本，只使用实体图）
    
    Args:
        record: 原始记录字典
        user_id: 用户 ID
        base_dir: 基础目录路径
        entity_progress_cache: entity_progress 缓存（可选）
        
    Returns:
        补全后的记录字典
    """
    timestamp = record.get('timestamp', '')
    if not timestamp:
        print(f"警告: 记录缺少 timestamp，user_id={user_id}, problem_name={record.get('problem_name', 'Unknown')}")
        return record
    
    # 加载 entity_progress（单图谱版本，路径为 graph_{user_id}_practice）
    entity_progress_file = os.path.join(
        base_dir,
        f"graph_{user_id}_practice", "model_General", "entity_progress.json"
    )
    
    if entity_progress_cache is None or user_id not in entity_progress_cache:
        entity_data = load_entity_progress(entity_progress_file)
        entity_summaries = entity_data.get('knowledge_summaries', {})
    else:
        entity_summaries = entity_progress_cache[user_id].get('knowledge_summaries', {})
    
    # 查找匹配的总结
    # 对于测试集，使用最接近的时间戳（因为测试集时间可能晚于 entity_progress 的时间范围）
    entity_summary = find_matching_summary(timestamp, entity_summaries, use_nearest=is_test)
    
    # 如果找到了总结，补全 e 字段（单图谱版本，只使用实体图）
    if entity_summary:
        record['e'] = format_e_field(entity_summary)
    else:
        # 如果找不到，使用默认值
        record['e'] = format_e_field("")
        print(f"警告: 未找到匹配的总结，user_id={user_id}, timestamp={timestamp}")
    
    return record


def merge_and_enrich(
    base_dir: str,
    output_file: str,
    data_type: str = "train",
    use_cache: bool = True,
    max_students: int = None,
    student_ids: List[str] = None
):
    """
    合并所有数据文件并补全 entity 和 knowledge 总结
    
    Args:
        base_dir: 基础目录路径
        output_file: 输出文件路径
        data_type: 数据类型，可选值: "train", "test", "validation", "val"
        use_cache: 是否使用缓存来加速
        max_students: 最大学生数量限制（None 表示不限制）
    """
    print(f"开始处理，基础目录: {base_dir}")
    print(f"数据类型: {data_type}")
    if student_ids:
        print(f"指定的学生ID: {student_ids}")
    elif max_students:
        print(f"学生数量限制: {max_students}")
    
    # 查找所有数据文件
    data_files = find_all_data_files(base_dir, data_type, max_students=max_students, student_ids=student_ids)
    print(f"找到 {len(data_files)} 个 {data_type}.jsonl 文件")
    
    if not data_files:
        print(f"错误: 未找到任何 {data_type}.jsonl 文件")
        return
    
    # 单图谱版本：只加载 entity_progress 文件（不加载 knowledge_progress）
    # 预加载所有 entity_progress 文件（使用缓存，路径为 graph_{user_id}_practice）
    entity_progress_cache = {}
    if use_cache:
        print("预加载 entity_progress 文件（单图谱版本）...")
        for data_file in tqdm(data_files, desc="加载 entity_progress"):
            user_id = extract_user_id_from_path(data_file)
            if user_id and user_id not in entity_progress_cache:
                entity_progress_file = os.path.join(
                    base_dir,
                    f"graph_{user_id}_practice", "model_General", "entity_progress.json"
                )
                if os.path.exists(entity_progress_file):
                    entity_data = load_entity_progress(entity_progress_file)
                    entity_progress_cache[user_id] = entity_data
        print(f"预加载了 {len(entity_progress_cache)} 个 entity_progress 文件")
    
    # 合并并补全数据
    total_records = 0
    enriched_records = 0
    missing_summaries = 0
    
    if HAS_JSONLINES:
        writer = jsonlines.open(output_file, 'w')
    else:
        writer = open(output_file, 'w', encoding='utf-8')
    
    try:
        for data_file in tqdm(data_files, desc="处理文件"):
            user_id = extract_user_id_from_path(data_file)
            if not user_id:
                print(f"警告: 无法从路径提取 user_id: {data_file}")
                continue
            
            # 单图谱版本：knowledge_progress 已经在缓存中或由 enrich_record 处理
            
            # 读取数据文件
            if HAS_JSONLINES:
                reader = jsonlines.open(data_file, 'r')
            else:
                reader = open(data_file, 'r', encoding='utf-8')
            
            try:
                for line in reader:
                    if not HAS_JSONLINES:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                    else:
                        record = line
                    
                    total_records += 1
                    
                    # 补全记录（单图谱版本，只使用实体图）
                    is_test = data_type == 'test'
                    enriched_record = enrich_record(
                        record,
                        user_id,
                        base_dir,
                        entity_progress_cache=entity_progress_cache,
                        is_test=is_test
                    )
                    
                    if enriched_record.get('e') and enriched_record.get('e') != "Student Knowledge Summary:\nNo knowledge points extracted from the graph.":
                        enriched_records += 1
                    else:
                        missing_summaries += 1
                    
                    record = enriched_record
                    
                    # 写入输出文件
                    if HAS_JSONLINES:
                        writer.write(record)
                    else:
                        writer.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            finally:
                if HAS_JSONLINES:
                    reader.close()
                else:
                    reader.close()
    
    finally:
        if HAS_JSONLINES:
            writer.close()
        else:
            writer.close()
    
    print(f"\n处理完成！")
    print(f"总记录数: {total_records}")
    print(f"成功补全: {enriched_records}")
    print(f"缺少总结: {missing_summaries}")
    print(f"输出文件: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='合并多个数据文件并补全 entity 和 knowledge 总结')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='基础目录路径（包含 student_*_practice 目录的父目录）')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出文件路径（jsonl格式）')
    parser.add_argument('--data_type', type=str, default='train',
                        choices=['train', 'test', 'validation', 'val', 'all'],
                        help='数据类型：train, test, validation/val, all（默认: train）')
    parser.add_argument('--no_cache', action='store_true',
                        help='不使用缓存（默认使用缓存加速）')
    parser.add_argument('--max_students', type=int, default=None,
                        help='最大学生数量限制（None 表示不限制）')
    parser.add_argument('--student_ids', type=str, default=None,
                        help='指定的学生ID列表（逗号分隔），例如: "134015446,135363688"')
    
    args = parser.parse_args()
    
    # 检查基础目录是否存在
    if not os.path.isdir(args.base_dir):
        print(f"错误: 基础目录不存在: {args.base_dir}")
        return
    
    # 解析学生ID列表
    student_ids = None
    if args.student_ids:
        student_ids = [sid.strip() for sid in args.student_ids.split(',') if sid.strip()]
    
    # 执行合并和补全
    merge_and_enrich(
        args.base_dir,
        args.output_file,
        data_type=args.data_type,
        use_cache=not args.no_cache,
        max_students=args.max_students,
        student_ids=student_ids
    )


if __name__ == '__main__':
    main()

