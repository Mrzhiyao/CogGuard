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


def extract_knowledge_graph_from_input_ids(input_ids: List[int], mark_kg_id: int, mark_r_id: int) -> Tuple[List[int], int, int]:
    try:
        # 找到第一个MARK_KG的位置
        first_mark_kg_idx = input_ids.index(mark_kg_id)
        # 找到第二个MARK_KG的位置（从第一个之后开始找）
        second_mark_kg_idx = input_ids.index(mark_kg_id, first_mark_kg_idx + 1)
        
        # 知识图在两个MARK_KG之间（不包含MARK_KG本身）
        knowledge_graph_ids = input_ids[first_mark_kg_idx + 1:second_mark_kg_idx]
        
        return knowledge_graph_ids, first_mark_kg_idx, second_mark_kg_idx
    except ValueError:
        # 如果找不到MARK_KG，返回空列表
        return [], -1, -1


def find_mark_token_ids(records: List[Dict]) -> Dict[str, int]:
    # 尝试从第一条记录中找到MARK_R（通常在分数之前）
    if not records:
        return {}
    
    first_record = records[0]
    input_ids = first_record.get('input_ids', [])
    
    # MARK_R应该在score_ids之前
    score_ids = first_record.get('score_ids', [])
    if score_ids and len(input_ids) > len(score_ids):
        # 找到score_ids在input_ids中的位置
        score_start = len(input_ids) - len(score_ids)
        # MARK_R应该在score_start之前
        if score_start > 0:
            mark_r_id = input_ids[score_start - 1]
        else:
            mark_r_id = None
    else:
        mark_r_id = None

    knowledge_graph_ids = first_record.get('knowledge_graph_ids', [])
    if knowledge_graph_ids:
        kg_str = str(knowledge_graph_ids)
        input_str = str(input_ids)

        entity_graph_ids = first_record.get('entity_graph_ids', [])
        if entity_graph_ids:

            pass
    
    return {}


def collect_knowledge_graphs_from_training(training_file: str) -> List[Tuple[List[int], int]]:

    training_records = load_jsonl(training_file)
    
    knowledge_graphs = []
    seen_kg = set()  # 用于去重
    
    for record in training_records:
        kg_ids = record.get('knowledge_graph_ids', [])
        if not kg_ids:
            continue
        
        # 使用tuple作为key去重
        kg_tuple = tuple(kg_ids)
        if kg_tuple not in seen_kg:
            seen_kg.add(kg_tuple)
            knowledge_graphs.append((kg_ids, len(kg_ids)))
    
    print(f"收集到 {len(knowledge_graphs)} 个唯一的知识图")
    return knowledge_graphs


def replace_knowledge_graph_in_input_ids(
    input_ids: List[int],
    old_kg_ids: List[int],
    new_kg_ids: List[int],
    mark_kg_id: int
) -> List[int]:

    try:
        # 找到第一个MARK_KG的位置
        first_mark_kg_idx = input_ids.index(mark_kg_id)
        # 找到第二个MARK_KG的位置
        second_mark_kg_idx = input_ids.index(mark_kg_id, first_mark_kg_idx + 1)
        
        # 构建新的input_ids
        new_input_ids = (
            input_ids[:first_mark_kg_idx + 1] +  # 保留到第一个MARK_KG（包含）
            new_kg_ids +  # 新的知识图
            input_ids[second_mark_kg_idx:]  # 从第二个MARK_KG开始（包含）到结尾
        )
        
        return new_input_ids
    except ValueError:
        # 如果找不到MARK_KG，返回原input_ids
        print("警告: 无法找到MARK_KG，保持原input_ids不变")
        return input_ids


def generate_validation_with_random_kg(
    validation_file: str,
    training_file: str,
    output_file: str,
    mark_kg_id: int,
    strategy: str = 'random',
    length_tolerance: float = 0.2
):
    # 1. 从训练集收集知识图
    training_kgs = collect_knowledge_graphs_from_training(training_file)
    if not training_kgs:
        print("错误: 无法从训练集中收集到知识图")
        return
    
    # 按长度分组（用于长度相似性匹配）
    kg_by_length = defaultdict(list)
    for kg_ids, length in training_kgs:
        kg_by_length[length].append(kg_ids)
    
    print(f"知识图长度分布: min={min(len(kg) for kg, _ in training_kgs)}, "
          f"max={max(len(kg) for kg, _ in training_kgs)}, "
          f"avg={np.mean([len(kg) for kg, _ in training_kgs]):.1f}")
    print()
    
    # 2. 加载验证集
    print(f"加载验证集: {validation_file}")
    validation_records = load_jsonl(validation_file)
    print(f"验证集记录数: {len(validation_records)}")
    print()
    
    # 3. 为每条验证集记录随机替换知识图
    print("替换知识图...")
    modified_count = 0
    
    for i, record in enumerate(validation_records):
        original_kg_ids = record.get('knowledge_graph_ids', [])
        if not original_kg_ids:
            print(f"警告: 记录 {i} 没有知识图，跳过")
            continue
        
        original_length = len(original_kg_ids)
        
        # 选择新的知识图
        if strategy == 'random':
            # 完全随机选择
            new_kg_ids, _ = random.choice(training_kgs)
        elif strategy == 'length_similar':
            # 选择长度相似的知识图
            target_length = original_length
            tolerance = int(target_length * length_tolerance)
            
            # 找到长度在容忍范围内的知识图
            candidate_lengths = [
                length for length in kg_by_length.keys()
                if abs(length - target_length) <= tolerance
            ]
            
            if candidate_lengths:
                # 从候选长度中随机选择一个长度
                chosen_length = random.choice(candidate_lengths)
                # 从该长度的知识图中随机选择一个
                new_kg_ids = random.choice(kg_by_length[chosen_length])
            else:
                # 如果没有找到相似长度的，使用完全随机
                new_kg_ids, _ = random.choice(training_kgs)
        else:
            raise ValueError(f"未知的策略: {strategy}")
        
        # 替换knowledge_graph_ids字段
        record['knowledge_graph_ids'] = new_kg_ids
        
        # 替换input_ids中的知识图部分
        input_ids = record.get('input_ids', [])
        if input_ids:
            new_input_ids = replace_knowledge_graph_in_input_ids(
                input_ids, original_kg_ids, new_kg_ids, mark_kg_id
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
        description='为验证集生成随机知识图版本，用于测试模型对知识summary的敏感度'
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
        help='训练集文件路径（用于收集知识图）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='输出文件路径'
    )
    parser.add_argument(
        '--mark_kg_id',
        type=int,
        required=True,
        help='MARK_KG的token ID'
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
    
    generate_validation_with_random_kg(
        validation_file=args.validation_file,
        training_file=args.training_file,
        output_file=args.output_file,
        mark_kg_id=args.mark_kg_id,
        strategy=args.strategy,
        length_tolerance=args.length_tolerance
    )


if __name__ == '__main__':
    main()

