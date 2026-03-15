import json
from typing import Dict, List, Tuple
import re
from pathlib import Path
from transformers import AutoTokenizer

# 尝试导入 jsonlines，如果没有则使用标准库
try:
    import jsonlines
    HAS_JSONLINES = True
except ImportError:
    HAS_JSONLINES = False


# 定义标识符
MARK_L = '<MARK_L>'
MARK_R = '<MARK_R>'
MARK_EG = '<MARK_EG>'  # Entity Graph 标识符
MARK_KG = '<MARK_KG>'  # Knowledge Point Graph 标识符


def parse_entity_and_knowledge_graph(e_text: str) -> Tuple[str, str]:

    # 使用正则表达式分离两种图
    entity_pattern = r'\[Entity Graph - Directly Extracted from Problems\]:\s*(.*?)(?=\[Knowledge Point Graph|$)'
    knowledge_pattern = r'\[Knowledge Point Graph - Predefined Knowledge Points\]:\s*(.*?)$'
    
    entity_match = re.search(entity_pattern, e_text, re.DOTALL)
    knowledge_match = re.search(knowledge_pattern, e_text, re.DOTALL)
    
    entity_graph = entity_match.group(1).strip() if entity_match else ""
    knowledge_graph = knowledge_match.group(1).strip() if knowledge_match else ""
    
    # 清理文本，移除多余的空白
    entity_graph = re.sub(r'\n\s*\n', '\n', entity_graph).strip()
    knowledge_graph = re.sub(r'\n\s*\n', '\n', knowledge_graph).strip()
    
    return entity_graph, knowledge_graph


def discretize_score(score: float, bins: List[Tuple[float, float, str]] = None) -> str:

    if bins is None:
        # 默认区间：0-20, 20-40, 40-60, 60-80, 80-100
        bins = [
            (0, 20, "0-20"),
            (20, 40, "20-40"),
            (40, 60, "40-60"),
            (60, 80, "60-80"),
            (80, 100, "80-100"),
        ]
    
    for min_val, max_val, label in bins:
        if min_val <= score < max_val:
            return label
    
    # 如果超出范围，返回最后一个区间
    return bins[-1][2]


def process_single_example(example: Dict, tokenizer, discretize: bool = True, is_test: bool = False) -> Dict:

    src = example.get('src', '')
    e = example.get('e', '')
    label = example.get('label', 0.0)
    
    # 解析实体图和知识图
    entity_graph, knowledge_graph = parse_entity_and_knowledge_graph(e)
    
    # Tokenize 各个部分
    src_ids = tokenizer(src, add_special_tokens=False)['input_ids']
    entity_graph_ids = tokenizer(entity_graph, add_special_tokens=False)['input_ids']
    knowledge_graph_ids = tokenizer(knowledge_graph, add_special_tokens=False)['input_ids']
    
    # 获取标识符的 token id
    mark_l_id = tokenizer.convert_tokens_to_ids(MARK_L)
    mark_r_id = tokenizer.convert_tokens_to_ids(MARK_R)
    mark_eg_id = tokenizer.convert_tokens_to_ids(MARK_EG)
    mark_kg_id = tokenizer.convert_tokens_to_ids(MARK_KG)
    
    # 如果标识符不在词汇表中，使用 UNK token（后续需要添加）
    if mark_l_id == tokenizer.unk_token_id:
        mark_l_id = tokenizer.unk_token_id
    if mark_r_id == tokenizer.unk_token_id:
        mark_r_id = tokenizer.unk_token_id
    if mark_eg_id == tokenizer.unk_token_id:
        mark_eg_id = tokenizer.unk_token_id
    if mark_kg_id == tokenizer.unk_token_id:
        mark_kg_id = tokenizer.unk_token_id
    
    # 构建 input_ids
    # 格式：题目 + <MARK_L> + <MARK_EG> + 实体图 + <MARK_EG> + <MARK_KG> + 知识图 + <MARK_KG> + <MARK_R> ]
    input_ids = (
        src_ids +
        [mark_l_id] +
        [mark_eg_id] +
        entity_graph_ids +
        [mark_eg_id] +
        [mark_kg_id] +
        knowledge_graph_ids +
        [mark_kg_id] +
        [mark_r_id]
    )
    
    # 处理分数
    if discretize:
        score_range = discretize_score(label)
        score_ids = tokenizer(score_range, add_special_tokens=False)['input_ids']
    else:
        # 直接使用分数值（需要特殊处理）
        score_str = f"{label:.2f}"
        score_ids = tokenizer(score_str, add_special_tokens=False)['input_ids']
    
    # 训练集：input_ids 包含分数；测试集：input_ids 不包含分数
    if not is_test:
        input_ids = input_ids + score_ids
    

    if is_test:
        labels = [-100] * len(input_ids)
    else:
        labels = (
            [-100] * len(src_ids) +
            [-100] * 5 +  # MARK_L, MARK_EG (2个), MARK_KG (2个), MARK_R
            [-100] * len(entity_graph_ids) +
            [-100] * len(knowledge_graph_ids) +
            score_ids  # 只计算分数部分的 loss
        )
    
    # 返回处理后的样本
    return {
        'input_ids': input_ids,
        'labels': labels,
        'src_ids': src_ids,
        'entity_graph_ids': entity_graph_ids,
        'knowledge_graph_ids': knowledge_graph_ids,
        'score_ids': score_ids,
        'label': label,
        'user_id': example.get('user_id', ''),
        'problem_name': example.get('problem_name', ''),
        'timestamp': example.get('timestamp', ''),
    }


def process_dataset(input_file: str, output_file: str, model_dir: str, discretize: bool = True, is_test: bool = False):

    print(f"开始处理数据集: {input_file}")
    print(f"模型目录: {model_dir}")
    print(f"分数离散化: {discretize}")
    print(f"测试集模式: {is_test} {'(input_ids 不包含分数)' if is_test else '(input_ids 包含分数)'}")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # 添加新的标识符 token
    new_tokens = [MARK_L, MARK_R, MARK_EG, MARK_KG]
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
    print(f"添加了 {num_added} 个新 token")
    print(f"新 token IDs: {[tokenizer.convert_tokens_to_ids(t) for t in new_tokens]}")
    
    processed_count = 0
    error_count = 0
    
    if HAS_JSONLINES:
        # 使用 jsonlines 库
        with jsonlines.open(input_file, 'r') as reader, \
             jsonlines.open(output_file, 'w') as writer:
            
            for idx, example in enumerate(reader):
                try:
                    processed_example = process_single_example(example, tokenizer, discretize, is_test)
                    writer.write(processed_example)
                    processed_count += 1
                    
                    if (idx + 1) % 100 == 0:
                        print(f"已处理 {idx + 1} 条数据...")
                        
                except Exception as e:
                    error_count += 1
                    print(f"处理第 {idx + 1} 条数据时出错: {e}")
                    print(f"问题数据: {example.get('problem_name', 'Unknown')}")
                    import traceback
                    traceback.print_exc()
                    continue
    else:
        # 使用标准库
        with open(input_file, 'r', encoding='utf-8') as reader, \
             open(output_file, 'w', encoding='utf-8') as writer:
            
            for idx, line in enumerate(reader):
                try:
                    example = json.loads(line.strip())
                    processed_example = process_single_example(example, tokenizer, discretize, is_test)
                    writer.write(json.dumps(processed_example, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                    if (idx + 1) % 100 == 0:
                        print(f"已处理 {idx + 1} 条数据...")
                        
                except Exception as e:
                    error_count += 1
                    print(f"处理第 {idx + 1} 条数据时出错: {e}")
                    if 'example' in locals():
                        print(f"问题数据: {example.get('problem_name', 'Unknown')}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 条")
    print(f"错误: {error_count} 条")
    print(f"输出文件: {output_file}")
    print(f"\n注意: 需要保存更新后的 tokenizer，或者在使用时重新添加这些 token")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='处理数据集用于训练，生成 tokenized 数据')
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入文件路径（jsonl格式）')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出文件路径（jsonl格式）')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='模型目录路径（用于加载 tokenizer）')
    parser.add_argument('--discretize', action='store_true', default=True,
                        help='是否将分数离散化（默认: True）')
    parser.add_argument('--no_discretize', dest='discretize', action='store_false',
                        help='不将分数离散化')
    parser.add_argument('--is_test', action='store_true',
                        help='测试集模式：input_ids 不包含分数（用于评估）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input_file).exists():
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    # 处理数据集
    process_dataset(args.input_file, args.output_file, args.model_dir, args.discretize, args.is_test)


if __name__ == '__main__':
    main()

