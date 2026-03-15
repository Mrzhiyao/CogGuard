import json
import os
import glob
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import AutoTokenizer
from functools import partial
import re


# 模型名称到路径和token ID的映射
MODEL_CONFIG = {
    'qwen25_3b': {
        'path': './Qwen2.5-3B-Instruct',
        'mark_r_token_id': None  # 将动态获取
    },
    'qwen3_06b': {
        'path': './Qwen3-0.6B',
        'mark_r_token_id': None
    },
    'qwen3_17b': {
        'path': './Qwen3-1.7B',
        'mark_r_token_id': None
    },
    'qwen3_8b': {
        'path': './Qwen3-8B',
        'mark_r_token_id': None
    },
    'llama3_1_8b': {
        'path': './Meta-Llama-3.1-8B-Instruct',
        'mark_r_token_id': None
    },
    'gemma2_2b': {
        'path': './gemma-2-2b-it',
        'mark_r_token_id': None
    },
    'gemma2_9b': {
        'path': './gemma-2-9b-it',
        'mark_r_token_id': None
    }
}

# 特殊token定义
MARK_R = '<MARK_R>'
MARK_L = '<MARK_L>'
MARK_EG = '<MARK_EG>'
MARK_KG = '<MARK_KG>'


def get_mark_r_token_id(tokenizer: AutoTokenizer) -> int:
    """获取 MARK_R 的 token ID"""
    # 确保token已添加到tokenizer
    new_tokens = [MARK_L, MARK_R, MARK_EG, MARK_KG]
    tokenizer.add_tokens(new_tokens, special_tokens=False)
    return tokenizer.convert_tokens_to_ids(MARK_R)


def preprocess_function(examples: Dict[str, List], tokenizer=None, score_start_token: int = None) -> Dict[str, List]:
    processed_input_ids = []
    processed_attention_masks = []
    processed_labels = []
    
    if score_start_token is None:
        raise ValueError("score_start_token must be provided")
    
    # 处理每个样本
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
            processed_input_ids.append(input_ids)
            truncated_input_ids = input_ids
            score_tokens = []
        
        # 2. attention_mask全为1，长度与truncated_input_ids一致
        attention_mask = [1] * len(truncated_input_ids)
        processed_attention_masks.append(attention_mask)
        
        # 3. 将分数token转换为类别索引(0-4)
        class_label = None
        
        # 方法1：优先从score_ids字段获取（如果存在），并用tokenizer解码
        if 'score_ids' in examples and idx < len(examples['score_ids']):
            score_ids = examples['score_ids'][idx]
            if score_ids and len(score_ids) > 0:
                if tokenizer is not None:
                    # 使用tokenizer解码分数token ids
                    try:
                        score_str = tokenizer.decode(score_ids, skip_special_tokens=True).strip()
                        # 解析分数区间（例如"0-20"）
                        match = re.search(r'(\d+)-(\d+)', score_str)
                        if match:
                            score_low = int(match.group(1))
                            score_high = int(match.group(2))
                            # 转换为类别索引：0-20->0, 20-40->1, 40-60->2, 60-80->3, 80-100->4
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
                        pass  # 解码失败，继续尝试其他方法
        
        # 方法2：如果从score_ids获取失败，使用从input_ids提取的score_tokens
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
        
        # 方法3：如果仍然无法确定，尝试从原始数据中的score字段获取（原始分数值）
        if class_label is None:
            if 'label' in examples and idx < len(examples['label']):
                score_value = examples['label'][idx]
                if isinstance(score_value, (int, float)):
                    # 根据分数值转换为类别：0-20->0, 20-40->1, 40-60->2, 60-80->3, 80-100->4
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
        
        # 如果还是无法确定，打印警告并使用默认值（0）
        if class_label is None:
            print(f"Warning: Could not determine class label for sample {idx}, using default 0")
            class_label = 0
        
        processed_labels.append(class_label)
    
    return {
        "input_ids": processed_input_ids,
        "attention_mask": processed_attention_masks,
        "labels": processed_labels  # 现在是整数列表，不是token序列
    }


def split_dataset_by_src_length(
    input_file: str, 
    num_workers: int,
    output_dir: str,
    model_name: str,
    tokenizer_path: str
) -> List[str]:
    # 1. 加载数据集和tokenizer
    print(f"  Loading dataset from {input_file}...")
    dataset = load_dataset('json', data_files=input_file)['train']
    
    # 加载tokenizer用于解码分数token和获取MARK_R token ID
    print(f"  Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        # 添加特殊token
        new_tokens = [MARK_L, MARK_R, MARK_EG, MARK_KG]
        tokenizer.add_tokens(new_tokens, special_tokens=False)
        score_start_token = get_mark_r_token_id(tokenizer)
        print(f"  MARK_R token ID: {score_start_token}")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    # 创建带tokenizer和score_start_token的preprocess_function
    preprocess_func = partial(
        preprocess_function, 
        tokenizer=tokenizer, 
        score_start_token=score_start_token
    )
    
    # 2. 计算每个样本的input_ids长度（去掉分数部分，即训练时的实际输入长度）
    print(f"  Calculating input_ids lengths (without score, actual training input length)...")
    lengths = []
    for i in range(len(dataset)):
        input_ids = dataset[i]['input_ids']
        # 找到MARK_R的位置，计算去掉分数后的长度（即训练时的实际输入长度）
        try:
            idx_mark_r = len(input_ids) - 1 - input_ids[::-1].index(score_start_token)
            # 只保留到MARK_R（包含MARK_R），这是训练时的实际输入长度
            training_input_length = idx_mark_r + 1
        except ValueError:
            # 如果没有找到MARK_R，使用完整长度（但这种情况应该很少见）
            training_input_length = len(input_ids)
        lengths.append((i, training_input_length))
    
    # 3. 按长度从长到短排序
    print(f"  Sorting by length (longest first)...")
    sorted_indices = sorted(lengths, key=lambda x: x[1], reverse=True)
    
    # 4. 打印长度分布信息
    all_lengths = [x[1] for x in sorted_indices]
    print(f"  Input length statistics (training input, without score):")
    print(f"    min={min(all_lengths)}, max={max(all_lengths)}, "
          f"mean={np.mean(all_lengths):.2f}, median={np.median(all_lengths):.2f}")
    
    # 5. 分层采样策略：在保持样本数相同的前提下，让长度差异更极端
    print(f"  Splitting into {num_workers} workers using stratified sampling...")
    total_samples = len(sorted_indices)
    samples_per_worker = total_samples // num_workers
    remainder = total_samples % num_workers
    
    worker_samples = []
    
    if num_workers == 3:
        worker0_end = samples_per_worker + (1 if remainder > 0 else 0)
        worker1_end = worker0_end + samples_per_worker + (1 if remainder > 1 else 0)
        worker0_samples = [sorted_indices[j][0] for j in range(0, worker0_end)]
        worker1_samples = [sorted_indices[j][0] for j in range(worker0_end, worker1_end)]
        worker2_samples = [sorted_indices[j][0] for j in range(worker1_end, total_samples)]
        min_samples = min(len(worker0_samples), len(worker1_samples), len(worker2_samples))
        worker_samples = [
            worker0_samples[:min_samples],
            worker1_samples[:min_samples],
            worker2_samples[:min_samples]
        ]

        if min_samples * 3 < total_samples:
            print(f"  Warning: {total_samples - min_samples * 3} samples discarded to ensure sync")
    else:
        # 其他worker数量，使用平均拆分策略
        start_idx = 0
        for i in range(num_workers):
            end_idx = start_idx + samples_per_worker + (1 if i < remainder else 0)
            worker_indices = [sorted_indices[j][0] for j in range(start_idx, end_idx)]
            worker_samples.append(worker_indices)
            start_idx = end_idx
    
    # 6. 保存处理后的数据集到不同文件
    output_files = []
    os.makedirs(output_dir, exist_ok=True)
    
    for worker_id in range(num_workers):
        # 获取该worker的样本
        worker_indices = worker_samples[worker_id]
        worker_dataset = dataset.select(worker_indices)
        
        # 应用预处理
        processed_dataset = worker_dataset.map(
            preprocess_func,
            batched=True,
            batch_size=1000,
            desc=f"  Processing worker {worker_id} dataset"
        )
        
        # 保存
        output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")
        processed_dataset.to_json(output_file, orient='records', lines=True)
        output_files.append(output_file)
        
        print(f"  Saved worker {worker_id} data to: {output_file} ({len(processed_dataset)} samples)")
    
    return output_files


def process_all_datasets():
    """处理所有数据集"""
    # 定义数据集和模型
    datasets = ['students_20_new', 'students_30_new', 'students_40_new']
    num_workers_list = [2, 3, 4]  
    
    # 模型名称映射（从文件名提取）
    model_file_mapping = {
        'qwen25_3b': 'qwen25_3b',
        'qwen3_06b': 'qwen3_06b',
        'qwen3_17b': 'qwen3_17b',
        'qwen3_8b': 'qwen3_8b',
        'llama3_1_8b': 'llama3_1_8b',
        'gemma2_9b': 'gemma2_9b',
        'gemma2_2b': 'gemma2_2b'
    }
    
    base_dir = 'datasets'
    
    total_tasks = 0
    completed_tasks = 0
    
    # 统计总任务数
    for dataset in datasets:
        dataset_num = dataset.replace('students_', '')
        training_set_dir = os.path.join(base_dir, dataset, 'training set')
        pattern = os.path.join(training_set_dir, f'train_processed_{dataset_num}_*.jsonl')
        files = glob.glob(pattern)
        total_tasks += len(files) * len(num_workers_list)
    
    
    # 处理每个数据集
    for dataset in datasets:
        dataset_num = dataset.replace('students_', '')
        training_set_dir = os.path.join(base_dir, dataset, 'training set')
        
        # 查找所有训练文件
        pattern = os.path.join(training_set_dir, f'train_processed_{dataset_num}_*.jsonl')
        train_files = sorted(glob.glob(pattern))
        
        for train_file in train_files:
            # 从文件名提取模型名
            basename = os.path.basename(train_file)
            # 例如: train_processed_20_qwen25_3b.jsonl -> qwen25_3b
            model_name = basename.replace(f'train_processed_{dataset_num}_', '').replace('.jsonl', '')
            
            if model_name not in MODEL_CONFIG:
                print(f"   跳过未知模型: {model_name}")
                continue
            
            model_config = MODEL_CONFIG[model_name]
            tokenizer_path = model_config['path']
            
            print(f"\n处理模型: {model_name}")
            print(f"  输入文件: {train_file}")
            print(f"  Tokenizer: {tokenizer_path}")
            
            # 为每个worker数量划分数据集
            for num_workers in num_workers_list:
                output_dir = os.path.join(training_set_dir, f'ray{num_workers}', model_name)
                
                print(f"\n  Worker数量: {num_workers}")
                print(f"  输出目录: {output_dir}")
                
                try:
                    output_files = split_dataset_by_src_length(
                        input_file=train_file,
                        num_workers=num_workers,
                        output_dir=output_dir,
                        model_name=model_name,
                        tokenizer_path=tokenizer_path
                    )
                    completed_tasks += 1
                    print(f"  ✓ 完成 ({completed_tasks}/{total_tasks})")
                except Exception as e:
                    print(f"  ✗ 失败: {e}")
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"处理完成！完成 {completed_tasks}/{total_tasks} 个任务")
    print("=" * 70)


if __name__ == "__main__":
    process_all_datasets()

