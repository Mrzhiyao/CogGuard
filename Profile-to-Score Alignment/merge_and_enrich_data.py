import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import jsonlines
    HAS_JSONLINES = True
except ImportError:
    HAS_JSONLINES = False


def find_all_data_files(base_dir: str, data_type: str = "train", max_students: int = None) -> List[str]:
    if data_type == "val":
        data_type = "validation"
    
    possible_names = [f"{data_type}.jsonl"]
    if data_type == "validation":
        possible_names.append("val.jsonl")
    elif data_type == "all":
        possible_names = ["all.jsonl"]
    
    all_files = []
    for name in possible_names:
        pattern = os.path.join(base_dir, "student_*_practice", "**", name)
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    all_files = sorted(list(set(all_files)))
    
    if max_students is not None and max_students > 0:
        student_files = {}
        for file_path in all_files:
            parts = file_path.split(os.sep)
            for part in parts:
                if part.startswith("student_") and part.endswith("_practice"):
                    student_id = part
                    if student_id not in student_files:
                        student_files[student_id] = []
                    student_files[student_id].append(file_path)
                    break
        
        sorted_students = sorted(student_files.keys())[:max_students]
        all_files = []
        for student_id in sorted_students:
            all_files.extend(student_files[student_id])
    
    return all_files


def find_all_train_files(base_dir: str) -> List[str]:
    return find_all_data_files(base_dir, "train")


def extract_user_id_from_path(file_path: str) -> Optional[str]:
    parts = file_path.split(os.sep)
    for part in parts:
        if part.startswith("student_") and part.endswith("_practice"):
            user_id = part.replace("student_", "").replace("_practice", "")
            return user_id
    return None


def load_entity_progress(entity_progress_file: str) -> Dict:
    if not os.path.exists(entity_progress_file):
        print(f"警告: entity_progress.json 不存在: {entity_progress_file}")
        return {"knowledge_summaries": {}}
    
    with open(entity_progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_knowledge_progress(knowledge_progress_file: str) -> Dict:
    if not os.path.exists(knowledge_progress_file):
        print(f"警告: knowledge_point_progress.json 不存在: {knowledge_progress_file}")
        return {"knowledge_summaries": {}}
    
    with open(knowledge_progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_matching_summary(timestamp: str, summaries: Dict, record_id: int = 1, use_nearest: bool = True) -> Optional[str]:
    key = f"{timestamp}_record_{record_id}"
    if key in summaries:
        return summaries[key]
    
    matching_keys = [k for k in summaries.keys() if timestamp in k]
    if matching_keys:
        return summaries[matching_keys[0]]
    
    if use_nearest and summaries:
        from datetime import datetime
        
        try:
            target_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            
            nearest_key = None
            min_diff = None
            
            for key in summaries.keys():
                try:
                    key_timestamp = key.split('_record_')[0]
                    key_time = datetime.strptime(key_timestamp, "%Y-%m-%d %H:%M:%S")
                    
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
            pass
    
    return None


def format_e_field(entity_summary: str, knowledge_summary: str) -> str:
    entity_summary = entity_summary.strip() if entity_summary else ""
    knowledge_summary = knowledge_summary.strip() if knowledge_summary else ""
    
    if not entity_summary or "No knowledge points extracted from the graph" in entity_summary:
        entity_summary = "Student Knowledge Summary:\nNo knowledge points extracted from the graph."
    elif not entity_summary.startswith("Student Knowledge Summary"):
        entity_summary = "Student Knowledge Summary:\n" + entity_summary
    
    if not knowledge_summary or "No knowledge points extracted from the graph" in knowledge_summary:
        knowledge_summary = "Student Knowledge Summary:\nNo knowledge points extracted from the graph."
    elif not knowledge_summary.startswith("Student Knowledge Summary"):
        knowledge_summary = "Student Knowledge Summary:\n" + knowledge_summary
    
    formatted = (
        "Student Knowledge Summary (Combined):\n\n"
        "[Entity Graph - Directly Extracted from Problems]:\n"
        f"{entity_summary}\n\n"
        "[Knowledge Point Graph - Predefined Knowledge Points]:\n"
        f"{knowledge_summary}"
    )
    
    return formatted


def enrich_record(
    record: Dict,
    user_id: str,
    base_dir: str,
    entity_progress_cache: Dict = None,
    knowledge_progress_cache: Dict = None,
    is_test: bool = False
) -> Dict:
    timestamp = record.get('timestamp', '')
    if not timestamp:
        print(f"警告: 记录缺少 timestamp，user_id={user_id}, problem_name={record.get('problem_name', 'Unknown')}")
        return record
    
    entity_progress_file = os.path.join(
        base_dir,
        "baseline_output", "entity_graph_global", "model_General", "entity_progress.json"
    )
    
    if entity_progress_cache is None:
        entity_data = load_entity_progress(entity_progress_file)
        entity_summaries = entity_data.get('knowledge_summaries', {})
    else:
        entity_summaries = entity_progress_cache.get('knowledge_summaries', {})
    
    knowledge_progress_file = os.path.join(
        base_dir,
        f"knowledge_point_graph_{user_id}_practice", "model_General", "knowledge_point_progress.json"
    )
    
    if knowledge_progress_cache is None or user_id not in knowledge_progress_cache:
        knowledge_data = load_knowledge_progress(knowledge_progress_file)
        knowledge_summaries = knowledge_data.get('knowledge_summaries', {})
    else:
        knowledge_summaries = knowledge_progress_cache[user_id].get('knowledge_summaries', {})
    
    entity_summary = find_matching_summary(timestamp, entity_summaries, use_nearest=is_test)
    knowledge_summary = find_matching_summary(timestamp, knowledge_summaries, use_nearest=is_test)
    
    if entity_summary or knowledge_summary:
        record['e'] = format_e_field(
            entity_summary or "",
            knowledge_summary or ""
        )
    else:
        record['e'] = format_e_field("", "")
        print(f"警告: 未找到匹配的总结，user_id={user_id}, timestamp={timestamp}")
    
    return record


def merge_and_enrich(
    base_dir: str,
    output_file: str,
    data_type: str = "train",
    use_cache: bool = True,
    max_students: int = None
):
    print(f"开始处理，基础目录: {base_dir}")
    print(f"数据类型: {data_type}")
    if max_students:
        print(f"学生数量限制: {max_students}")
    
    data_files = find_all_data_files(base_dir, data_type, max_students=max_students)
    print(f"找到 {len(data_files)} 个 {data_type}.jsonl 文件")
    
    if not data_files:
        print(f"错误: 未找到任何 {data_type}.jsonl 文件")
        return
    
    entity_progress_file = os.path.join(
        base_dir,
        "baseline_output", "entity_graph_global", "model_General", "entity_progress.json"
    )
    print(f"加载 entity_progress: {entity_progress_file}")
    entity_data = load_entity_progress(entity_progress_file)
    entity_summaries = entity_data.get('knowledge_summaries', {})
    print(f"Entity summaries 数量: {len(entity_summaries)}")
    
    knowledge_progress_cache = {}
    if use_cache:
        print("预加载 knowledge_progress 文件...")
        for data_file in tqdm(data_files, desc="加载 knowledge_progress"):
            user_id = extract_user_id_from_path(data_file)
            if user_id and user_id not in knowledge_progress_cache:
                knowledge_progress_file = os.path.join(
                    base_dir,
                    f"knowledge_point_graph_{user_id}_practice", "model_General", "knowledge_point_progress.json"
                )
                if os.path.exists(knowledge_progress_file):
                    knowledge_data = load_knowledge_progress(knowledge_progress_file)
                    knowledge_progress_cache[user_id] = knowledge_data
        print(f"预加载了 {len(knowledge_progress_cache)} 个 knowledge_progress 文件")
    
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
            
            if user_id in knowledge_progress_cache:
                user_knowledge_data = knowledge_progress_cache[user_id]
            else:
                knowledge_progress_file = os.path.join(
                    base_dir,
                    f"knowledge_point_graph_{user_id}_practice", "model_General", "knowledge_point_progress.json"
                )
                user_knowledge_data = load_knowledge_progress(knowledge_progress_file)
            
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
                    
                    timestamp = record.get('timestamp', '')
                    is_test = data_type == 'test'
                    entity_summary = find_matching_summary(timestamp, entity_summaries, use_nearest=is_test)
                    knowledge_summaries = user_knowledge_data.get('knowledge_summaries', {})
                    knowledge_summary = find_matching_summary(timestamp, knowledge_summaries, use_nearest=is_test)
                    
                    if entity_summary or knowledge_summary:
                        record['e'] = format_e_field(
                            entity_summary or "",
                            knowledge_summary or ""
                        )
                        enriched_records += 1
                    else:
                        record['e'] = format_e_field("", "")
                        missing_summaries += 1
                    
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
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.base_dir):
        print(f"错误: 基础目录不存在: {args.base_dir}")
        return
    
    merge_and_enrich(
        args.base_dir,
        args.output_file,
        data_type=args.data_type,
        use_cache=not args.no_cache,
        max_students=args.max_students
    )


if __name__ == '__main__':
    main()

