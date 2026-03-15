import json
import random
import os
import glob
import argparse
from tqdm import tqdm


def load_jsonl(file_path: str) -> list:
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告: 解析JSON失败 ({file_path}): {e}")
                    continue
    return records


def save_jsonl(records: list, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        raise IOError(f"保存文件失败: {file_path}, 错误: {e}")


def create_no_embedding_file(input_file: str, output_file: str):
    records = load_jsonl(input_file)
    # 直接使用原始记录，保留所有字段（包括'e'字段）
    save_jsonl(records, output_file)
    return records


def split_dataset(records: list, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """划分数据集（使用相同的随机种子）"""
    random.seed(seed)
    
    records_shuffled = records.copy()
    random.shuffle(records_shuffled)
    
    total = len(records_shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_records = records_shuffled[:train_end]
    val_records = records_shuffled[train_end:val_end]
    test_records = records_shuffled[val_end:]
    
    return train_records, val_records, test_records


def get_record_id(record: dict) -> tuple:
    return (
        record.get('user_id', ''),
        record.get('problem_name', ''),
        record.get('timestamp', '')
    )


def verify_split_match(original_split_file: str, new_records: list, split_name: str):
    if not os.path.exists(original_split_file):
        return True
    
    original_records = load_jsonl(original_split_file)
    
    original_ids = {get_record_id(r) for r in original_records}
    new_ids = {get_record_id(r) for r in new_records}
    
    if original_ids == new_ids:
        print(f"  ✓ {split_name} 匹配成功: {len(original_records)} 条记录")
        return True
    else:
        missing = original_ids - new_ids
        extra = new_ids - original_ids
        print(f"  ✗ {split_name} 匹配失败:")
        print(f"    缺失记录: {len(missing)} 条")
        print(f"    多余记录: {len(extra)} 条")
        if missing:
            print(f"    缺失示例: {list(missing)[:3]}")
        if extra:
            print(f"    多余示例: {list(extra)[:3]}")
        return False


def process_file(
    input_file: str,
    base_output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    verify: bool = True
):
    # 1. 加载原始记录（保留所有字段）
    records = load_jsonl(input_file)
    
    # 2. 使用相同的随机种子切分
    train_records, val_records, test_records = split_dataset(
        records, train_ratio, val_ratio, test_ratio, seed
    )
    
    # 3. 构建输出文件路径（匹配原始切分文件的结构）
    base_name = os.path.basename(input_file)
    name_parts = base_name.replace('all_merged_', '').replace('_new.jsonl', '').replace('.jsonl', '')
    student_dir = f"students_{name_parts}_new"
    
    # 4. 保存切分后的文件
    train_dir = os.path.join(base_output_dir, student_dir, "training set")
    val_dir = os.path.join(base_output_dir, student_dir, "validation set")
    test_dir = os.path.join(base_output_dir, student_dir, "test set")
    
    train_file = os.path.join(train_dir, f"train_processed_{name_parts}_new_no_embedding.jsonl")
    val_file = os.path.join(val_dir, f"validation_processed_{name_parts}_new_no_embedding.jsonl")
    test_file = os.path.join(test_dir, f"test_processed_{name_parts}_new_no_embedding.jsonl")
    
    save_jsonl(train_records, train_file)
    save_jsonl(val_records, val_file)
    save_jsonl(test_records, test_file)
    
    print(f"    训练集: {train_file} ({len(train_records)} 条)")
    print(f"    验证集: {val_file} ({len(val_records)} 条)")
    print(f"    测试集: {test_file} ({len(test_records)} 条)")
    
    # 5. 验证：检查是否与原始切分匹配
    if verify:
        original_train_pattern = os.path.join("datasets", student_dir, "training set", f"train_processed_{name_parts}_new_*.jsonl")
        original_val_pattern = os.path.join("datasets", student_dir, "validation set", f"validation_processed_{name_parts}_new_*.jsonl")
        original_test_pattern = os.path.join("datasets", student_dir, "test set", f"test_processed_{name_parts}_new_*.jsonl")
        
        original_train_files = glob.glob(original_train_pattern)
        original_val_files = glob.glob(original_val_pattern)
        original_test_files = glob.glob(original_test_pattern)
        
        if original_train_files:
            verify_split_match(original_train_files[0], train_records, "训练集")
        if original_val_files:
            verify_split_match(original_val_files[0], val_records, "验证集")
        if original_test_files:
            verify_split_match(original_test_files[0], test_records, "测试集")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='从原始文件创建无embedding版本并切分')
    parser.add_argument('--input_files', type=str, nargs='+',
                        default=[
                            '/datasets/all_merged_40_new.jsonl',
                            '/datasets/all_merged_30_new.jsonl',
                            '/datasets/all_merged_20_new.jsonl'
                        ],
                        help='输入文件列表')
    parser.add_argument('--output_dir', type=str, default='datasets',
                        help='输出目录（默认: datasets）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认: 0.8）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例（默认: 0.1）')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例（默认: 0.1）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42，必须与原始切分相同）')
    parser.add_argument('--verify', action='store_true',
                        help='是否验证切分结果与原始切分匹配')
    
    args = parser.parse_args()
    
    # 检查比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和为 {total_ratio}，不等于1.0，将自动归一化")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    success_count = 0
    fail_count = 0
    
    for i, input_file in enumerate(args.input_files, 1):
        if not os.path.exists(input_file):
            print(f"[{i}/{len(args.input_files)}] ✗ 文件不存在: {input_file}")
            fail_count += 1
            continue
        
        print(f"[{i}/{len(args.input_files)}] 处理: {input_file}")
        try:
            success = process_file(
                input_file, args.output_dir,
                args.train_ratio, args.val_ratio, args.test_ratio,
                args.seed, args.verify
            )
            if success:
                success_count += 1
                print(f"  ✓ 成功")
            else:
                fail_count += 1
                print(f"  ✗ 失败")
        except Exception as e:
            fail_count += 1
            print(f"  ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 70)
    print("处理完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print("=" * 70)


if __name__ == '__main__':
    main()

