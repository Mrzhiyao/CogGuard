import json
import random
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np


def load_jsonl(file_path: str) -> List[Dict]:
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"警告: 解析JSON失败: {e}")
                continue
    return records


def save_jsonl(records: List[Dict], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def replace_entity_graph_in_input_ids(
    input_ids: List[int],
    old_eg_ids: List[int],
    new_eg_ids: List[int],
    mark_eg_id: int
) -> List[int]:

    try:
        # 找到第一个MARK_EG的位置
        first_mark_eg_idx = input_ids.index(mark_eg_id)
        # 找到第二个MARK_EG的位置
        second_mark_eg_idx = input_ids.index(mark_eg_id, first_mark_eg_idx + 1)
        
        # 构建新的input_ids
        new_input_ids = (
            input_ids[:first_mark_eg_idx + 1] +  # 保留到第一个MARK_EG（包含）
            new_eg_ids +  # 新的实体图
            input_ids[second_mark_eg_idx:]  # 从第二个MARK_EG开始（包含）到结尾
        )
        
        return new_input_ids
    except ValueError:
        # 如果找不到MARK_EG，返回原input_ids
        return input_ids


def collect_entity_graphs_from_training(training_file: str) -> List[Tuple[List[int], int]]:
    training_records = load_jsonl(training_file)
    
    entity_graphs = []
    seen_eg = set()  # 用于去重
    
    for record in training_records:
        eg_ids = record.get('entity_graph_ids', [])
        if not eg_ids:
            continue
        
        # 使用tuple作为key去重
        eg_tuple = tuple(eg_ids)
        if eg_tuple not in seen_eg:
            seen_eg.add(eg_tuple)
            entity_graphs.append((eg_ids, len(eg_ids)))
    
    return entity_graphs


def generate_validation_with_random_eg(
    validation_file: str,
    training_file: str,
    output_file: str,
    mark_eg_id: int,
    strategy: str = 'random',
    length_tolerance: float = 0.2
):

    # 1. 从训练集收集实体图
    training_egs = collect_entity_graphs_from_training(training_file)
    if not training_egs:
        print("错误: 无法从训练集中收集到实体图")
        return
    
    # 按长度分组（用于长度相似性匹配）
    eg_by_length = defaultdict(list)
    for eg_ids, length in training_egs:
        eg_by_length[length].append(eg_ids)
    
    print(f"实体图长度分布: min={min(len(eg) for eg, _ in training_egs)}, "
          f"max={max(len(eg) for eg, _ in training_egs)}, "
          f"avg={np.mean([len(eg) for eg, _ in training_egs]):.1f}")
    print()
    
    # 2. 加载验证集
    print(f"加载验证集: {validation_file}")
    validation_records = load_jsonl(validation_file)
    print(f"验证集记录数: {len(validation_records)}")
    print()
    
    # 3. 为每条验证集记录随机替换实体图
    print("替换实体图...")
    modified_count = 0
    
    for i, record in enumerate(validation_records):
        original_eg_ids = record.get('entity_graph_ids', [])
        if not original_eg_ids:
            print(f"警告: 记录 {i} 没有实体图，跳过")
            continue
        
        original_length = len(original_eg_ids)
        
        # 选择新的实体图
        if strategy == 'random':
            # 完全随机选择
            new_eg_ids, _ = random.choice(training_egs)
        elif strategy == 'length_similar':
            # 选择长度相似的实体图
            target_length = original_length
            tolerance = int(target_length * length_tolerance)
            
            # 找到长度在容忍范围内的实体图
            candidate_lengths = [
                length for length in eg_by_length.keys()
                if abs(length - target_length) <= tolerance
            ]
            
            if candidate_lengths:
                # 从候选长度中随机选择一个长度
                chosen_length = random.choice(candidate_lengths)
                # 从该长度的实体图中随机选择一个
                new_eg_ids = random.choice(eg_by_length[chosen_length])
            else:
                # 如果没有找到相似长度的，使用完全随机
                new_eg_ids, _ = random.choice(training_egs)
        else:
            raise ValueError(f"未知的策略: {strategy}")
        
        # 替换entity_graph_ids字段
        record['entity_graph_ids'] = new_eg_ids
        
        # 替换input_ids中的实体图部分
        input_ids = record.get('input_ids', [])
        if input_ids:
            new_input_ids = replace_entity_graph_in_input_ids(
                input_ids, original_eg_ids, new_eg_ids, mark_eg_id
            )
            record['input_ids'] = new_input_ids
        
        modified_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(validation_records)} 条记录")
    
    print(f"完成！共修改 {modified_count} 条记录")
    print()
    
    # 4. 保存结果
    print(f"保存到: {output_file}")
    save_jsonl(validation_records, output_file)
    print("完成！")


def main():
    parser = argparse.ArgumentParser(
        description='为验证集生成随机实体图版本，用于测试模型对实体图的敏感度'
    )
    parser.add_argument(
        '--validation_file',
        type=str,
        required=True,
        help='验证集文件路径'
    )
    parser.add_argument(
        '--training_file',
        type=str,
        required=True,
        help='训练集文件路径（用于收集实体图）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='输出文件路径'
    )
    parser.add_argument(
        '--mark_eg_id',
        type=int,
        required=True,
        help='MARK_EG的token ID'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='random',
        choices=['random', 'length_similar'],
        help='替换策略: random（完全随机）或 length_similar（长度相似）'
    )
    parser.add_argument(
        '--length_tolerance',
        type=float,
        default=0.2,
        help='长度相似性的容忍度（比例，仅用于length_similar策略）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generate_validation_with_random_eg(
        validation_file=args.validation_file,
        training_file=args.training_file,
        output_file=args.output_file,
        mark_eg_id=args.mark_eg_id,
        strategy=args.strategy,
        length_tolerance=args.length_tolerance
    )


if __name__ == '__main__':
    main()

