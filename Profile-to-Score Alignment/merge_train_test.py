import json
import os
import glob
from pathlib import Path
from typing import List
from datetime import datetime
from tqdm import tqdm


def find_all_student_dirs(base_dir: str) -> List[str]:
    pattern = os.path.join(base_dir, "student_*_practice")
    dirs = glob.glob(pattern)
    return sorted(dirs)


def merge_train_test_for_student(student_dir: str) -> bool:
    model_dir = os.path.join(student_dir, "model_General")
    train_file = os.path.join(model_dir, "train.jsonl")
    test_file = os.path.join(model_dir, "test.jsonl")
    all_file = os.path.join(model_dir, "all.jsonl")
    
    # 检查文件是否存在
    if not os.path.exists(train_file) and not os.path.exists(test_file):
        return False
    
    # 读取所有记录
    all_records = []
    
    # 读取 train.jsonl
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"警告: 解析 JSON 失败 ({train_file}): {e}")
                        continue
    
    # 读取 test.jsonl
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"警告: 解析 JSON 失败 ({test_file}): {e}")
                        continue
    
    if not all_records:
        return False
    
    # 按时间戳排序
    def get_timestamp(record):
        timestamp_str = record.get('timestamp', '')
        if not timestamp_str:
            return datetime.min
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.min
    
    all_records.sort(key=get_timestamp)
    
    # 写入 all.jsonl
    with open(all_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成 all.jsonl')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='基础目录路径')
    parser.add_argument('--max_students', type=int, default=None,
                        help='最大学生数量限制（None 表示不限制）')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    # 查找所有学生目录
    student_dirs = find_all_student_dirs(base_dir)
    if args.max_students and args.max_students > 0:
        student_dirs = student_dirs[:args.max_students]
        print(f"找到 {len(find_all_student_dirs(base_dir))} 个学生目录，限制为前 {args.max_students} 个")
    
    success_count = 0
    fail_count = 0
    
    # 合并每个学生的文件
    for student_dir in tqdm(student_dirs, desc="合并文件"):
        if merge_train_test_for_student(student_dir):
            success_count += 1
        else:
            fail_count += 1

if __name__ == '__main__':
    main()

