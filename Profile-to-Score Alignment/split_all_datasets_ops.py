import json
import random
import os
import glob
import argparse
from collections import defaultdict
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
        # 验证文件是否成功创建
        if not os.path.exists(file_path):
            raise IOError(f"文件保存后不存在: {file_path}")
        return True
    except Exception as e:
        raise IOError(f"保存文件失败: {file_path}, 错误: {e}")


def split_dataset(records: list, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    # 设置随机种子
    random.seed(seed)
    
    # 随机shuffle
    records_shuffled = records.copy()
    random.shuffle(records_shuffled)
    
    # 计算切分点
    total = len(records_shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 切分
    train_records = records_shuffled[:train_end]
    val_records = records_shuffled[train_end:val_end]
    test_records = records_shuffled[val_end:]
    
    return train_records, val_records, test_records


def analyze_split(records: list, split_name: str):
    task_set = set()
    label_distribution = defaultdict(int)
    
    for record in records:
        # OPS数据集可能使用不同的字段名
        task_id = record.get('task_id', record.get('task_name', record.get('id', '')))
        label = record.get('label', '')
        
        task_set.add(task_id)
        label_distribution[label] += 1
    
    return {
        'count': len(records),
        'tasks': len(task_set),
        'label_dist': dict(label_distribution)
    }


def process_file(input_file: str, output_dir: str, train_ratio: float, val_ratio: float, 
                 test_ratio: float, seed: int, analyze: bool = False):
    # 加载数据
    records = load_jsonl(input_file)
    
    if not records:
        print(f"  警告: {input_file} 没有记录，跳过")
        return False
    
    # 划分数据集
    train_records, val_records, test_records = split_dataset(
        records, train_ratio, val_ratio, test_ratio, seed
    )
    
    if analyze:
        train_stats = analyze_split(train_records, "训练集")
        val_stats = analyze_split(val_records, "验证集")
        test_stats = analyze_split(test_records, "测试集")
        
        print(f"    训练集: {train_stats['count']} 条, {train_stats['tasks']} 任务")
        print(f"    验证集: {val_stats['count']} 条, {val_stats['tasks']} 任务")
        print(f"    测试集: {test_stats['count']} 条, {test_stats['tasks']} 任务")
    
    # 构建输出文件路径
    base_name = os.path.basename(input_file)
    name_parts = base_name.replace('ops_processed_', '').replace('.jsonl', '')
    
    # 保存到不同的子目录
    train_dir = os.path.join(output_dir, 'training set')
    val_dir = os.path.join(output_dir, 'validation set')
    test_dir = os.path.join(output_dir, 'test set')
    
    train_file = os.path.join(train_dir, f"train_processed_ops_{name_parts}.jsonl")
    val_file = os.path.join(val_dir, f"validation_processed_ops_{name_parts}.jsonl")
    test_file = os.path.join(test_dir, f"test_processed_ops_{name_parts}.jsonl")
    
    try:
        save_jsonl(train_records, train_file)
        save_jsonl(val_records, val_file)
        save_jsonl(test_records, test_file)
        
        # 验证所有文件都已成功创建
        if not os.path.exists(train_file):
            raise IOError(f"训练集文件未创建: {train_file}")
        if not os.path.exists(val_file):
            raise IOError(f"验证集文件未创建: {val_file}")
        if not os.path.exists(test_file):
            raise IOError(f"测试集文件未创建: {test_file}")
        
        return True
    except Exception as e:
        print(f"  ✗ 保存文件时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='批量对 OPS 数据集进行随机划分（8:1:1）')
    parser.add_argument('--datasets_dir', type=str, default='datasets',
                        help='datasets目录路径（默认: datasets）')
    parser.add_argument('--pattern', type=str, default='ops/ops_processed_*.jsonl',
                        help='文件匹配模式（默认: ops/ops_processed_*.jsonl）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认: 0.8）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例（默认: 0.1）')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例（默认: 0.1）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--analyze', action='store_true',
                        help='是否分析划分后的数据分布')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='输出目录后缀（例如: _random_split）')
    
    args = parser.parse_args()
    
    # 检查比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和为 {total_ratio}，不等于1.0，将自动归一化")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    # 查找所有文件
    pattern = os.path.join(args.datasets_dir, args.pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"错误: 没有找到匹配的文件: {pattern}")
        return
    
    # 处理每个文件
    success_count = 0
    fail_count = 0
    
    for i, input_file in enumerate(tqdm(files, desc="处理文件"), 1):
        # 确定输出目录（保持原有目录结构）
        rel_path = os.path.relpath(input_file, args.datasets_dir)
        subdir = os.path.dirname(rel_path)
        
        if args.output_suffix:
            # 如果有后缀，创建新的子目录
            output_dir = os.path.join(args.datasets_dir, subdir + args.output_suffix)
        else:
            # 否则保存在原目录
            output_dir = os.path.join(args.datasets_dir, subdir)
        
        print(f"[{i}/{len(files)}] 处理: {input_file}")
        
        try:
            success = process_file(
                input_file, output_dir, args.train_ratio, args.val_ratio, 
                args.test_ratio, args.seed, args.analyze
            )
            if success:
                success_count += 1
                print(f"  ✓ 成功: 保存到 {output_dir}")
            else:
                fail_count += 1
                print(f"  ✗ 失败")
        except Exception as e:
            fail_count += 1
            print(f"  ✗ 错误: {e}")
        
        print()
    
    print("=" * 70)
    print("处理完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print("=" * 70)


if __name__ == '__main__':
    main()

