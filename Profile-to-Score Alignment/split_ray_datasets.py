#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量对所有数据集进行随机划分（不做长度排序）
将数据集随机打乱后平均分成2、3、4份，用于分布式训练
"""

import json
import os
import glob
from datasets import load_dataset
from typing import Dict, List
import numpy as np
from transformers import AutoTokenizer
from functools import partial
import re
import random

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
        
        # 方法3：如果仍然无法确定，尝试从原始数据中的label字段获取（原始分数值）
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


def split_dataset_random(
    input_file: str, 
    num_workers: int,
    output_dir: str,
    model_name: str,
    tokenizer_path: str,
    score_start_token: int,
    seed: int = 42
):

    # 1. 加载数据集和tokenizer
    print(f"  Loading dataset from {input_file}...")
    dataset = load_dataset('json', data_files=input_file)['train']
    print(f"  Dataset size: {len(dataset)} samples")
    
    # 加载tokenizer用于解码分数token
    print(f"  Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        # 添加特殊token
        new_tokens = [MARK_L, MARK_R, MARK_EG, MARK_KG]
        tokenizer.add_tokens(new_tokens, special_tokens=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    # 创建带tokenizer和score_start_token的preprocess_function
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
        desc="  Processing dataset"
    )
    
    # 2. 随机打乱数据集
    print(f"  Shuffling dataset (seed={seed})...")
    processed_dataset = processed_dataset.shuffle(seed=seed)
    
    # 3. 平均分成num_workers份
    total_samples = len(processed_dataset)
    samples_per_worker = total_samples // num_workers
    remainder = total_samples % num_workers
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = []
    
    # 分配样本到各个worker
    start_idx = 0
    for worker_id in range(num_workers):
        # 计算这个worker的样本数（余数分配给前几个worker）
        worker_size = samples_per_worker + (1 if worker_id < remainder else 0)
        end_idx = start_idx + worker_size
        
        # 提取这个worker的数据
        worker_dataset = processed_dataset.select(range(start_idx, end_idx))
        
        # 保存
        output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")
        worker_dataset.to_json(output_file, orient='records', lines=True)
        output_files.append(output_file)
        
        print(f"  Saved worker {worker_id} data to: {output_file} ({len(worker_dataset)} samples)")
        
        start_idx = end_idx
    
    return output_files


def process_all_datasets():
    # 定义数据集和模型
    datasets = ['students_20', 'students_30', 'students_40']
    num_workers_list = [2, 3, 4]  # ray2_random, ray3_random, ray4_random
    
    # 模型名称映射（从文件名提取）
    model_file_mapping = {
        'qwen25_3b': 'qwen25_3b',
        'qwen3_1.7b': 'qwen3_1.7b',
        'qwen3_06b': 'qwen3_06b',
        'qwen3_8b': 'qwen3_8b',
        'llama3_1_8b': 'llama3_1_8b',
        'gemma2_9b': 'gemma2_9b',
        'gemma2_2b': 'gemma2_2b'
    }
    
    base_dir = 'datasets'
    seed = 42  # 固定随机种子，确保可复现
    
    total_tasks = 0
    completed_tasks = 0
    failed_tasks = 0
    
    for dataset_name in datasets:

        # 查找训练集文件
        training_set_dir = os.path.join(base_dir, dataset_name, "training set")
        train_files = glob.glob(os.path.join(training_set_dir, "train_processed_*.jsonl"))
        
        # 过滤掉reformatted和ray目录下的文件
        train_files = [f for f in train_files if 'reformatted' not in f and 'ray' not in os.path.dirname(f)]
        
        if not train_files:
            print(f"  警告: 未找到训练集文件，跳过 {dataset_name}")
            continue
        
        print(f"  找到 {len(train_files)} 个训练集文件")
        
        for train_file in train_files:
            filename = os.path.basename(train_file)
            
            # 从文件名提取模型名称
            model_name = None
            for key in model_file_mapping.keys():
                if key in filename:
                    model_name = key
                    break
            
            if model_name is None:
                print(f"  警告: 无法从文件名识别模型: {filename}，跳过")
                continue
            
            model_config = MODEL_CONFIG[model_name]
            tokenizer_path = model_config['path']
            score_start_token = model_config['mark_r_token_id']
            
            # 对每个num_workers进行处理
            for num_workers in num_workers_list:
                total_tasks += 1
                
                # 构建输出目录
                output_dir = os.path.join(training_set_dir, f"ray{num_workers}_random", model_name)
                
                print(f"\n  {'-'*65}")
                print(f"  文件: {filename}")
                print(f"  模型: {model_name}")
                print(f"  Workers: {num_workers}")
                print(f"  输出目录: {output_dir}")
                
                try:
                    output_files = split_dataset_random(
                        input_file=train_file,
                        num_workers=num_workers,
                        output_dir=output_dir,
                        model_name=model_name,
                        tokenizer_path=tokenizer_path,
                        score_start_token=score_start_token,
                        seed=seed
                    )
                    
                    # 验证所有worker文件都存在
                    all_exist = all(os.path.exists(f) for f in output_files)
                    if all_exist:
                        completed_tasks += 1
                        print(f"  ✓ 成功")
                    else:
                        failed_tasks += 1
                        print(f"  ✗ 失败: 部分输出文件不存在")
                        
                except Exception as e:
                    failed_tasks += 1
                    print(f"  ✗ 错误: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("处理完成")
    print(f"{'='*70}")
    print(f"总任务数: {total_tasks}")
    print(f"成功: {completed_tasks}")
    print(f"失败: {failed_tasks}")
    
    if failed_tasks > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    process_all_datasets()

