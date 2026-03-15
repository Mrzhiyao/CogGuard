import json
import os
import re
from typing import Dict, List
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer

# 模型配置
MODEL_CONFIG = {
    'qwen25_3b': {
        'path': './Qwen2.5-3B-Instruct',
        'mark_r_token_id': 151666  
    },
    'qwen3_06b': {
        'path': './Qwen3-0.6B',
        'mark_r_token_id': 151670
    },
    'qwen3_17b': {
        'path': './Qwen3-1.7B',
        'mark_r_token_id': 151670
    },
    'qwen3_8b': {
        'path': './Qwen3-8B',
        'mark_r_token_id': 151670
    },
    'llama3_1_8b': {
        'path': './Meta-Llama-3.1-8B-Instruct',
        'mark_r_token_id': 128257
    },
    'gemma2_2b': {
        'path': './gemma-2-2b-it',
        'mark_r_token_id': 256001
    },
    'gemma2_9b': {
        'path': './gemma-2-9b-it',
        'mark_r_token_id': 256001
    }
}

MARK_L = '<MARK_L>'
MARK_R = '<MARK_R>'
MARK_EG = '<MARK_EG>'
MARK_KG = '<MARK_KG>'


def preprocess_function(examples: Dict[str, List], tokenizer=None, score_start_token: int = None) -> Dict[str, List]:

    processed_input_ids = []
    processed_attention_masks = []
    processed_labels = []
    
    if score_start_token is None:
        raise ValueError("score_start_token must be provided")
    
    for idx, input_ids in enumerate(examples['input_ids']):
        # 1. 删除MARK_R之后的分数部分，只保留到MARK_R（包含MARK_R）
        try:
            # 找到最后一个MARK_R的位置
            idx_mark_r = len(input_ids) - 1 - input_ids[::-1].index(score_start_token)
            # 只保留到MARK_R（包含MARK_R）
            truncated_input_ids = input_ids[:idx_mark_r + 1]
            # 提取分数部分的token ids（用于解析类别）
            score_tokens = input_ids[idx_mark_r + 1:]
            processed_input_ids.append(truncated_input_ids)
        except ValueError:
            # 如果没有找到MARK_R，保持原样（但这种情况应该很少见）
            print(f"Warning: No {score_start_token} (MARK_R) found in sequence, keeping input_ids as is")
            processed_input_ids.append(input_ids)
            truncated_input_ids = input_ids
            score_tokens = []
        
        # 2. attention_mask全为1，长度与truncated_input_ids一致
        attention_mask = [1] * len(truncated_input_ids)
        processed_attention_masks.append(attention_mask)
        
        # 3. 将分数token转换为类别索引(0-4)
        class_label = None
        
        if 'score_ids' in examples and idx < len(examples['score_ids']):
            score_ids = examples['score_ids'][idx]
            if score_ids and len(score_ids) > 0:
                if tokenizer is not None:
                    # 使用tokenizer解码分数token ids
                    try:
                        score_str = tokenizer.decode(score_ids, skip_special_tokens=True).strip()
                        match = re.search(r'(\d+)-(\d+)', score_str)
                        if match:
                            score_low = int(match.group(1))
                            score_high = int(match.group(2))
                            if 0 <= score_low < 20 and 0 <= score_high <= 20:
                                class_label = 0
                            elif 20 <= score_low < 40 and 20 < score_high <= 40:
                                class_label = 1
                            elif 40 <= score_low < 60 and 40 < score_high <= 60:
                                class_label = 2
                            elif 60 <= score_low < 80 and 60 < score_high <= 80:
                                class_label = 3
                            elif 80 <= score_low < 100 and 80 < score_high <= 100:
                                class_label = 4
                    except Exception as e:
                        pass  
        
        if class_label is None and len(score_tokens) > 0 and tokenizer is not None:
            try:
                score_str = tokenizer.decode(score_tokens, skip_special_tokens=True).strip()
                match = re.search(r'(\d+)-(\d+)', score_str)
                if match:
                    score_low = int(match.group(1))
                    score_high = int(match.group(2))
                    if 0 <= score_low < 20 and 0 <= score_high <= 20:
                        class_label = 0
                    elif 20 <= score_low < 40 and 20 < score_high <= 40:
                        class_label = 1
                    elif 40 <= score_low < 60 and 40 < score_high <= 60:
                        class_label = 2
                    elif 60 <= score_low < 80 and 60 < score_high <= 80:
                        class_label = 3
                    elif 80 <= score_low < 100 and 80 < score_high <= 100:
                        class_label = 4
            except Exception as e:
                pass
        
        if class_label is None:
            if 'label' in examples and idx < len(examples['label']):
                score_value = examples['label'][idx]
                if isinstance(score_value, (int, float)):
                    if 0 <= score_value < 20:
                        class_label = 0
                    elif 20 <= score_value < 40:
                        class_label = 1
                    elif 40 <= score_value < 60:
                        class_label = 2
                    elif 60 <= score_value < 80:
                        class_label = 3
                    elif 80 <= score_value <= 100:
                        class_label = 4
        
        if class_label is None:
            class_label = 0
        
        processed_labels.append(class_label)
    
    return {
        "input_ids": processed_input_ids,
        "attention_mask": processed_attention_masks,
        "labels": processed_labels  # 现在是整数列表，不是token序列
    }


def reformat_dataset(input_file: str, output_file: str, tokenizer_path: str, score_start_token: int):
    print(f"  处理文件: {input_file}")
    print(f"  输出文件: {output_file}")
    
    # 加载数据集
    dataset = load_dataset('json', data_files=input_file)['train']
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # 添加特殊token（如果还没有添加）
    new_tokens = [MARK_L, MARK_R, MARK_EG, MARK_KG]
    tokenizer.add_tokens(new_tokens, special_tokens=False)
    
    # 创建预处理函数
    preprocess_func = partial(
        preprocess_function,
        tokenizer=tokenizer,
        score_start_token=score_start_token
    )
    
    # 应用预处理
    processed_dataset = dataset.map(
        preprocess_func,
        batched=True,
        batch_size=1000,
        desc="  格式化数据"
    )
    
    # 保存
    processed_dataset.to_json(output_file, orient='records', lines=True)
    print(f"  ✓ 完成: {output_file} ({len(processed_dataset)} 条)")


def process_all_datasets():
    datasets = ['students_20', 'students_30', 'students_40']
    
    base_dir = 'datasets'
    
    total_files = 0
    processed_files = 0
    
    for dataset_name in datasets:

        dataset_dir = os.path.join(base_dir, dataset_name)
        
        processed_files_pattern = os.path.join(dataset_dir, "**", "*_processed_*.jsonl")
        import glob
        all_files = glob.glob(processed_files_pattern, recursive=True)
        
        all_files = [f for f in all_files if 'ray' not in f and 'worker' not in f]
        
        print(f"找到 {len(all_files)} 个文件需要处理")
        
        for input_file in all_files:
            total_files += 1
            
            # 从文件名提取模型名称
            filename = os.path.basename(input_file)
            parts = filename.replace('.jsonl', '').split('_')
            
            model_name = None
            for part in parts:
                if part in MODEL_CONFIG:
                    model_name = part
                    break
            
            if model_name is None:
                continue
            
            model_config = MODEL_CONFIG[model_name]
            tokenizer_path = model_config['path']
            score_start_token = model_config['mark_r_token_id']
            
            # 生成输出文件名（在同一目录下，添加_reformatted后缀）
            dir_name = os.path.dirname(input_file)
            base_name = os.path.basename(input_file).replace('.jsonl', '')
            output_file = os.path.join(dir_name, f"{base_name}_reformatted.jsonl")
            
            if os.path.exists(output_file):
                print(f"  跳过（已存在）: {output_file}")
                continue
            
            try:
                reformat_dataset(input_file, output_file, tokenizer_path, score_start_token)
                processed_files += 1
            except Exception as e:
                print(f"  错误: 处理 {input_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*70}")
    print(f"处理完成")
    print(f"{'='*70}")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files}")
    print(f"失败/跳过: {total_files - processed_files}")


if __name__ == "__main__":
    process_all_datasets()

