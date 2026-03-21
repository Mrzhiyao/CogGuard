from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import json
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.join(_SCRIPT_DIR, '..')

# 配置路径
mode_path = os.path.join(_DEMO_DIR, 'base_model', 'Qwen3-1.7B')
lora_paths = [
    os.path.join(_DEMO_DIR, 'checkpoints', 'edu', 'Qwen3-1.7B', 'stu40')
]

# 20学生测试集路径
test_file = os.path.join(_DEMO_DIR, 'checkpoints', 'test_data', 'validation_processed_40_new_qwen3_17b.jsonl')
# 去重以后的验证集
# 标记token ID（用于识别分数部分的起始位置）
SCORE_START_TOKEN = 151670

# 类别到分数区间的映射
CLASS_TO_SCORE_RANGE = {
    0: "0-20",
    1: "20-40",
    2: "40-60",
    3: "60-80",
    4: "80-100"
}

# 分数区间到类别的反向映射
SCORE_RANGE_TO_CLASS = {v: k for k, v in CLASS_TO_SCORE_RANGE.items()}

# 类别到数值的映射（用于计算MAE，使用区间中点）
CLASS_TO_VALUE = {
    0: 10,   # 0-20的中点
    1: 30,   # 20-40的中点
    2: 50,   # 40-60的中点
    3: 70,   # 60-80的中点
    4: 90    # 80-100的中点
}


def extract_input_without_score(input_ids):
    """
    从input_ids中提取除去分数部分的输入
    
    Args:
        input_ids: 完整的input_ids列表
        
    Returns:
        去除分数部分的input_ids（只保留到256001，包含256001）
    """
    if SCORE_START_TOKEN in input_ids:
        # 找到最后一个256001的位置
        idx = len(input_ids) - 1 - input_ids[::-1].index(SCORE_START_TOKEN)
        # 返回256001及之前的部分（包含256001）
        return input_ids[:idx + 1]
    else:
        # 如果没有找到标记，返回整个input_ids
        print(f"Warning: No {SCORE_START_TOKEN} found in input_ids")
        return input_ids


def extract_ground_truth_score(input_ids, score_ids=None):
    """
    从input_ids或score_ids中提取真实的分数部分
    
    Args:
        input_ids: 完整的input_ids列表
        score_ids: score_ids字段（如果存在）
        
    Returns:
        分数区间字符串（如"0-20"）
    """
    # 优先使用score_ids字段
    if score_ids and len(score_ids) > 0:
        # score_ids是token ids，需要解码
        return None  # 需要在外部用tokenizer解码
    
    # 如果score_ids不存在，从input_ids中提取
    if SCORE_START_TOKEN in input_ids:
        # 找到最后一个256001的位置
        idx = len(input_ids) - 1 - input_ids[::-1].index(SCORE_START_TOKEN)
        # 返回256001之后的部分
        return input_ids[idx + 1:]
    else:
        return []


def run_inference_on_dataset(lora_path, test_file, tokenizer, output_file=None):
    """
    在整个测试集上运行推理（使用SequenceClassification模型）
    
    Args:
        lora_path: LoRA权重路径
        test_file: 测试集文件路径
        tokenizer: tokenizer对象
        output_file: 可选的输出文件，保存预测结果
    """
    print(f"Testing LoRA checkpoint: {lora_path}")
    
    # 加载模型（使用SequenceClassification）
    print("Loading base model (SequenceClassification)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        mode_path, 
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        num_labels=5  # 5个类别：0-20, 20-40, 40-60, 60-80, 80-100
    )
    
    # 设置padding token（Qwen模型需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 调整模型词汇表大小（如果tokenizer添加了新token）
    # 这必须与训练时的操作一致：训练时添加了MARK token后调用了resize_token_embeddings
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print(f"Resizing model embeddings from {embedding_size} to {len(tokenizer)} to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))
        print(f"Model embeddings resized successfully.")
    
    # 加载LoRA权重
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    model.eval()
    
    # 加载测试集
    print("Loading test dataset...")
    test_dataset = load_dataset('json', data_files=test_file)['train']
    # 只测试前200个样本
    test_dataset = test_dataset.select(range(len(test_dataset)))
    
    print(f"Running inference on {len(test_dataset)} samples...")
    count = 0
    total_samples = 0
    
    # 用于存储所有预测和真实标签（用于计算混淆矩阵、MAE、F1）
    all_predictions = []
    all_ground_truths = []
    all_predicted_classes = []
    all_ground_truth_classes = []
    
    # 用于存储极端错误样本（0-20预测为80-100）
    extreme_error_samples = []
    
    for idx, sample in enumerate(tqdm(test_dataset)):
        # 提取输入（不包含分数，只保留到256001）
        input_ids = sample['input_ids']
        input_without_score = extract_input_without_score(input_ids)
        
        # 提取真实分数标签：优先从score_ids字段获取
        ground_truth = None
        if 'score_ids' in sample and sample['score_ids']:
            score_ids = sample['score_ids']
            score_str = tokenizer.decode(score_ids, skip_special_tokens=True).strip()
            # 从分数字符串中提取区间（如"0-20"）
            match = re.search(r'(\d+)-(\d+)', score_str)
            if match:
                ground_truth = match.group(0)  # "0-20"
        else:
            # 如果没有score_ids，从input_ids中提取
            score_ids = extract_ground_truth_score(input_ids)
            if score_ids:
                score_str = tokenizer.decode(score_ids, skip_special_tokens=True).strip()
                match = re.search(r'(\d+)-(\d+)', score_str)
                if match:
                    ground_truth = match.group(0)
        
        # 如果没有找到ground truth，跳过
        if not ground_truth:
            print(f"Warning: Could not extract ground truth for sample {idx}")
            continue
        
        # 准备模型输入
        input_tensor = torch.tensor([input_without_score]).to(model.device)
        
        # 创建attention_mask（全为1，因为输入不包含padding）
        attention_mask = torch.ones_like(input_tensor).to(model.device)
        
        # 分类预测（不使用generate，直接forward）
        with torch.no_grad():
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attention_mask
            )
            logits = outputs.logits  # [batch_size, 5]
            predicted_class = torch.argmax(logits, dim=-1).item()  # 获取预测的类别索引（0-4）
            # 获取所有类别的概率（用于分析）
            # 注意：需要先转换为float，因为bfloat16不能直接转换为numpy
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()[0]  # [5]
        
        # 将类别索引转换为分数区间字符串
        predicted_score_range = CLASS_TO_SCORE_RANGE.get(predicted_class, "Unknown")
        
        # 将ground_truth转换为类别索引
        ground_truth_class = SCORE_RANGE_TO_CLASS.get(ground_truth, -1)
        if ground_truth_class == -1:
            # 如果无法映射，跳过这个样本（不应该发生）
            print(f"Warning: 无法映射ground_truth '{ground_truth}' 到类别，跳过样本 {idx}")
            continue
        
        # 收集预测和真实标签（用于后续分析）
        all_predictions.append(predicted_score_range)
        all_ground_truths.append(ground_truth)
        all_predicted_classes.append(predicted_class)
        all_ground_truth_classes.append(ground_truth_class)
        
        # ========== 记录极端错误样本：0-20预测为80-100 ==========
        if ground_truth_class == 0 and predicted_class == 4:  # 0-20预测为80-100
            # 解码input_ids为文本（用于查看）
            input_text = tokenizer.decode(input_without_score, skip_special_tokens=True)
            
            # 保存样本信息（包含完整信息用于导出）
            error_sample = {
                'sample_idx': idx,
                'problem_name': sample.get('problem_name', 'N/A'),
                'user_id': sample.get('user_id', 'N/A'),
                'timestamp': sample.get('timestamp', 'N/A'),
                'ground_truth': ground_truth,
                'predicted': predicted_score_range,
                'ground_truth_class': ground_truth_class,
                'predicted_class': predicted_class,
                'input_text': input_text,  # 完整文本
                'input_ids': input_without_score,  # 完整的input_ids（用于后续分析）
                'input_length': len(input_without_score),
                'logits': logits.float().cpu().numpy()[0].tolist(),  # 所有类别的logits（需要先转换为float）
                'probs': probs.tolist(),  # 所有类别的概率
            }
            extreme_error_samples.append(error_sample)
        
        # 计算准确率
        total_samples += 1
        if predicted_score_range == ground_truth:
            count += 1
        
        # 打印结果
        print(f"标签分数: {ground_truth}")
        print(f"预测分数: {predicted_score_range} (类别: {predicted_class})")
        print(f"当前匹配数: {count}/{total_samples}")
    
    # ========== 计算评估指标 ==========
    accuracy = count / total_samples if total_samples > 0 else 0.0
    print(f"\n{'='*70}")
    print("基础统计:")
    print(f"{'='*70}")
    print(f"完全匹配: {count}/{total_samples}")
    print(f"准确率: {accuracy:.2%}")
    
    # 检查是否有有效样本
    if len(all_ground_truth_classes) == 0:
        print("Warning: 没有有效的样本用于计算评估指标！")
        return {
            'accuracy': accuracy,
            'mae': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'f1_per_class': {},
            'confusion_matrix': []
        }
    
    # 确保所有标签都已收集（应该已经一致了，但为了安全再检查一次）
    if len(all_ground_truth_classes) != len(all_predicted_classes):
        print(f"Warning: 标签数量不匹配！真实标签: {len(all_ground_truth_classes)}, 预测标签: {len(all_predicted_classes)}")
        min_len = min(len(all_ground_truth_classes), len(all_predicted_classes))
        all_ground_truth_classes = all_ground_truth_classes[:min_len]
        all_predicted_classes = all_predicted_classes[:min_len]
    
    # 转换为numpy数组
    y_true = np.array(all_ground_truth_classes)
    y_pred = np.array(all_predicted_classes)
    
    # ========== 1. 混淆矩阵 ==========
    print(f"\n{'='*70}")
    print("混淆矩阵 (5x5):")
    print(f"{'='*70}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    # 打印混淆矩阵（带标签）
    class_names = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    print("\n预测类别 ->")
    print(f"{'真实类别':<12} {'0-20':<8} {'20-40':<8} {'40-60':<8} {'60-80':<8} {'80-100':<8}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:<12}"
        for j in range(5):
            row_str += f"{cm[i, j]:<8}"
        print(row_str)
    
    # 打印每行的总数
    print("-" * 70)
    row_sums = cm.sum(axis=1)
    print(f"{'总计':<12}", end="")
    for sum_val in row_sums:
        print(f"{sum_val:<8}", end="")
    print()
    
    # ========== 2. MAE (平均绝对误差) ==========
    print(f"\n{'='*70}")
    print("MAE (平均绝对误差):")
    print(f"{'='*70}")
    
    # 将类别转换为数值（使用区间中点）
    y_true_values = np.array([CLASS_TO_VALUE[c] for c in y_true])
    y_pred_values = np.array([CLASS_TO_VALUE[c] for c in y_pred])
    
    mae = np.mean(np.abs(y_true_values - y_pred_values))
    print(f"MAE: {mae:.2f} 分")
    print(f"说明: 平均预测误差为 {mae:.2f} 分（基于区间中点：10, 30, 50, 70, 90）")
    
    # ========== 3. Precision, Recall, F1分数 ==========
    print(f"\n{'='*70}")
    print("Precision, Recall, F1分数:")
    print(f"{'='*70}")
    
    # 计算每个类别的Precision, Recall, F1分数
    precision_scores = precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    recall_scores = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    f1_scores = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    
    # 计算宏平均和加权平均
    precision_macro = precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average='weighted', zero_division=0)
    
    print("\n每个类别的指标:")
    print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12} {precision_scores[i]:<12.4f} {recall_scores[i]:<12.4f} {f1_scores[i]:<12.4f}")
    
    print(f"\n宏平均 (Macro Average):")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1: {f1_macro:.4f}")
    
    print(f"\n加权平均 (Weighted Average):")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1: {f1_weighted:.4f}")
    
    # ========== 详细的分类报告 ==========
    print(f"\n{'='*70}")
    print("详细分类报告:")
    print(f"{'='*70}")
    report = classification_report(
        y_true, y_pred, 
        labels=[0, 1, 2, 3, 4],
        target_names=class_names,
        zero_division=0
    )
    print(report)
    
    # ========== 极端错误样本分析：0-20预测为80-100 ==========
    print(f"\n{'='*70}")
    print(f"极端错误样本分析：真实标签为'0-20'但预测为'80-100'")
    print(f"{'='*70}")
    print(f"找到 {len(extreme_error_samples)} 个极端错误样本\n")
    
    # ========== 导出极端错误样本到JSON文件 ==========
    if len(extreme_error_samples) > 0:
        # 生成输出文件名（基于lora_path和test_file）
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lora_name = os.path.basename(os.path.dirname(lora_path))
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"extreme_errors_0-20_to_80-100_{test_name}_{timestamp}.json")
        
        # 准备导出的数据（包含完整信息）
        export_data = {
            'summary': {
                'total_extreme_errors': len(extreme_error_samples),
                'lora_path': lora_path,
                'test_file': test_file,
                'timestamp': timestamp,
                'description': '真实标签为0-20但预测为80-100的极端错误样本'
            },
            'samples': []
        }
        
        for error_sample in extreme_error_samples:
            # 准备完整的样本信息
            sample_data = {
                'sample_idx': error_sample['sample_idx'],
                'problem_name': error_sample['problem_name'],
                'user_id': error_sample['user_id'],
                'ground_truth': error_sample['ground_truth'],
                'predicted': error_sample['predicted'],
                'ground_truth_class': error_sample['ground_truth_class'],
                'predicted_class': error_sample['predicted_class'],
                'input_text': error_sample['input_text'],  # 完整文本
                'input_ids': error_sample['input_ids'],  # 完整的input_ids
                'input_length': error_sample['input_length'],
                'probabilities': {
                    class_names[i]: {
                        'prob': float(error_sample['probs'][i]),
                        'prob_percent': float(error_sample['probs'][i] * 100),
                        'logit': float(error_sample['logits'][i])
                    }
                    for i in range(5)
                },
                'predicted_prob': float(error_sample['probs'][error_sample['predicted_class']]),
                'true_prob': float(error_sample['probs'][error_sample['ground_truth_class']]),
                'confidence_diff': float(error_sample['probs'][error_sample['predicted_class']] - 
                                       error_sample['probs'][error_sample['ground_truth_class']])
            }
            export_data['samples'].append(sample_data)
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*70}")
        print(f"极端错误样本已导出到文件:")
        print(f"  {output_file}")
        print(f"{'='*70}\n")
    
    # if len(extreme_error_samples) > 0:
    #     for i, error_sample in enumerate(extreme_error_samples, 1):
    #         print(f"\n{'='*70}")
    #         print(f"极端错误样本 #{i}/{len(extreme_error_samples)}")
    #         print(f"{'='*70}")
    #         print(f"样本索引: {error_sample['sample_idx']}")
    #         print(f"题目名称: {error_sample['problem_name']}")
    #         print(f"用户ID: {error_sample['user_id']}")
    #         print(f"真实标签: {error_sample['ground_truth']} (类别 {error_sample['ground_truth_class']})")
    #         print(f"预测标签: {error_sample['predicted']} (类别 {error_sample['predicted_class']})")
    #         print(f"\n所有类别的概率分布:")
    #         for j, class_name in enumerate(class_names):
    #             prob = error_sample['probs'][j]
    #             logit = error_sample['logits'][j]
    #             marker = " <-- 预测" if j == error_sample['predicted_class'] else ""
    #             marker2 = " <-- 真实" if j == error_sample['ground_truth_class'] else ""
    #             print(f"  {class_name}: 概率={prob:.4f} ({prob*100:.2f}%), logit={logit:.2f}{marker}{marker2}")
            
    #         # 显示预测置信度
    #         predicted_prob = error_sample['probs'][error_sample['predicted_class']]
    #         true_prob = error_sample['probs'][error_sample['ground_truth_class']]
    #         print(f"\n预测置信度分析:")
    #         print(f"  预测类别(80-100)的置信度: {predicted_prob:.4f} ({predicted_prob*100:.2f}%)")
    #         print(f"  真实类别(0-20)的置信度: {true_prob:.4f} ({true_prob*100:.2f}%)")
    #         print(f"  置信度差异: {predicted_prob - true_prob:.4f} ({(predicted_prob - true_prob)*100:.2f}%)")
            
            # print(f"\n输入文本（前800字符，便于查看完整上下文）:")
            # input_text = error_sample['input_text']
            # if len(input_text) > 800:
            #     print(f"{input_text[:800]}...")
            # else:
            #     print(input_text)
            # print(f"\n输入长度: {len(error_sample['input_ids'])} tokens")
    else:
        print("没有找到极端错误样本（真实标签为'0-20'但预测为'80-100'）")
    
    # ========== 返回所有指标 ==========
    results = {
        'accuracy': accuracy,
        'mae': mae,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': {class_names[i]: precision_scores[i] for i in range(5)},
        'recall_per_class': {class_names[i]: recall_scores[i] for i in range(5)},
        'f1_per_class': {class_names[i]: f1_scores[i] for i in range(5)},
        'confusion_matrix': cm.tolist(),
        'extreme_error_samples': extreme_error_samples  # 包含极端错误样本
    }
    
    return results


def main():
    # 加载tokenizer（所有checkpoint共享）
    tokenizer = AutoTokenizer.from_pretrained(mode_path)
    
    # 添加MARK tokens（如果tokenizer中没有）
    MARK_L = "<MARK_L>"
    MARK_R = "<MARK_R>"
    MARK_EG = "<MARK_EG>"
    MARK_KG = "<MARK_KG>"
    
    new_tokens = [MARK_L, MARK_R, MARK_EG, MARK_KG]
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
    if num_added > 0:
        print(f"Added {num_added} new tokens to tokenizer")
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 对每个LoRA checkpoint进行测试
    all_results = {}
    
    for idx, lora_path in enumerate(lora_paths):
        print(f"\n{'='*70}")
        print(f"Testing checkpoint {idx+1}/{len(lora_paths)}")
        print(f"{'='*70}")
        
        results = run_inference_on_dataset(
            lora_path=lora_path,
            test_file=test_file,
            tokenizer=tokenizer,
            output_file=''
        )
        
        all_results[lora_path] = results
        
        # 释放显存
        torch.cuda.empty_cache()
    
    # 打印所有结果
    print(f"\n{'='*70}")
    print("所有checkpoint的测试结果汇总:")
    print(f"{'='*70}")
    for lora_path, results in all_results.items():
        print(f"\n{lora_path}:")
        print(f"  准确率 (Accuracy): {results['accuracy']:.2%}")
        print(f"  MAE: {results['mae']:.2f} 分")
        print(f"  宏平均 Precision: {results['precision_macro']:.4f}")
        print(f"  宏平均 Recall: {results['recall_macro']:.4f}")
        print(f"  宏平均 F1: {results['f1_macro']:.4f}")
        print(f"  加权平均 Precision: {results['precision_weighted']:.4f}")
        print(f"  加权平均 Recall: {results['recall_weighted']:.4f}")
        print(f"  加权平均 F1: {results['f1_weighted']:.4f}")
        print(f"  各类别指标:")
        print(f"    {'类别':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print(f"    {'-'*50}")
        for class_name in ["0-20", "20-40", "40-60", "60-80", "80-100"]:
            print(f"    {class_name:<12} {results['precision_per_class'][class_name]:<12.4f} {results['recall_per_class'][class_name]:<12.4f} {results['f1_per_class'][class_name]:<12.4f}")


if __name__ == "__main__":
    main()
