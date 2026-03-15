import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from collections import defaultdict
import os
from datetime import datetime
import re

train_file = './datasets/train_processed_40_new_no_embedding.jsonl'
val_file = './datasets/validation_processed_40_new_no_embedding.jsonl'

# 类别映射
CLASS_TO_SCORE_RANGE = {
    0: "0-20",
    1: "20-40",
    2: "40-60",
    3: "60-80",
    4: "80-100"
}

SCORE_RANGE_TO_CLASS = {v: k for k, v in CLASS_TO_SCORE_RANGE.items()}

CLASS_TO_VALUE = {
    0: 10, 1: 30, 2: 50, 3: 70, 4: 90
}

# 将分数转换为类别
def score_to_class(score):
    """将分数 (0-100) 转换为类别 (0-4)"""
    if score < 20:
        return 0
    elif score < 40:
        return 1
    elif score < 60:
        return 2
    elif score < 80:
        return 3
    else:
        return 4

# 将分数转换为二分类（正确/错误），用于 SimpleKT 的交互编码
def score_to_binary(score, threshold=60.0):
    """将分数转换为二分类：>= threshold 为正确(1)，否则为错误(0)"""
    return 1 if score >= threshold else 0


def extract_kcs_from_e_field(e_content):
    if not e_content or not isinstance(e_content, str):
        return []
    
    kcs = []
    
    # 方法1: 从 "Knowledge Point Graph" 部分提取
    kc_section_match = re.search(
        r'\[Knowledge Point Graph[^\]]*\]:\s*(.*?)(?=\n\n\[|\Z)', 
        e_content, 
        re.DOTALL
    )
    
    if kc_section_match:
        kc_section = kc_section_match.group(1)
        # 提取所有带引号的知识点（格式: "KC_NAME" (score X.XX)）
        kc_matches = re.findall(r'"([^"]+)"\s*\(score', kc_section)
        kcs.extend(kc_matches)
    
    # 去重并保持顺序
    seen = set()
    unique_kcs = []
    for kc in kcs:
        kc_upper = kc.upper().strip()
        if kc_upper and kc_upper not in seen:
            seen.add(kc_upper)
            unique_kcs.append(kc_upper)
    
    return unique_kcs


def build_kc_mapping_from_dataset(dataset, use_e_field=True):
    problem_kcs = defaultdict(set)
    
    if use_e_field:
        print("从 'e' 字段提取知识点...")
        for sample in dataset:
            problem_name = sample.get('problem_name')
            if not problem_name:
                continue
            
            # 从 'e' 字段提取知识点
            e_content = sample.get('e', '')
            kcs = extract_kcs_from_e_field(e_content)
            
            if kcs:
                problem_kcs[problem_name].update(kcs)
    
    # 转换为列表格式
    kc_mapping = {prob: sorted(list(kcs)) for prob, kcs in problem_kcs.items()}
    
    print(f"构建了 {len(kc_mapping)} 个题目的知识点映射")
    total_kcs = len(set(kc for kcs in kc_mapping.values() for kc in kcs))
    print(f"总知识点数: {total_kcs}")
    
    return kc_mapping


class SimpleKTDataset(Dataset):
    def __init__(self, data_file, max_seq_len=200, train_file=None, remove_duplicates=True, 
                 use_rasch=False, kc_mapping=None):
        self.max_seq_len = max_seq_len
        self.use_rasch = use_rasch
        self.kc_mapping = kc_mapping or {}
        
        # 加载数据
        print(f"加载数据文件: {data_file}")
        dataset = load_dataset('json', data_files=data_file)['train']
        
        # 如果使用 Rasch 分解但没有提供 kc_mapping，尝试从数据中提取
        if use_rasch and not self.kc_mapping:
            print("未提供知识点映射，尝试从数据中提取...")
            # 检查是否有 'e' 字段
            has_e_field = any('e' in sample for sample in dataset)
            if has_e_field:
                print("检测到 'e' 字段，从中提取知识点...")
                self.kc_mapping = build_kc_mapping_from_dataset(dataset, use_e_field=True)
                if self.kc_mapping:
                    print(f"成功提取 {len(self.kc_mapping)} 个题目的知识点映射")
                else:
                    print("警告: 未能从 'e' 字段提取知识点，将使用简化版（不做 Rasch 分解）")
                    use_rasch = False
            else:
                print("警告: 数据中没有 'e' 字段，无法提取知识点，将使用简化版（不做 Rasch 分解）")
                use_rasch = False
        
        self.use_rasch = use_rasch  # 更新 use_rasch 标志
        
        if train_file and remove_duplicates:
            train_dataset = load_dataset('json', data_files=train_file)['train']
            train_keys_set = set()
            for sample in train_dataset:
                key = (
                    sample.get('problem_name'),
                    sample.get('user_id'),
                    sample.get('timestamp')
                )
                if all(key):
                    train_keys_set.add(key)
            print(f"训练集唯一样本数: {len(train_keys_set)}")
            
            unique_samples = []
            for sample in dataset:
                key = (
                    sample.get('problem_name'),
                    sample.get('user_id'),
                    sample.get('timestamp')
                )
                if all(key) and key not in train_keys_set:
                    unique_samples.append(sample)
            
            print(f"验证集原始样本数: {len(dataset)}")
            print(f"去重后唯一样本数: {len(unique_samples)}")
            dataset = unique_samples
        
        # 按用户分组，构建交互序列
        user_interactions = defaultdict(list)
        
        for sample in dataset:
            user_id = sample.get('user_id')
            problem_name = sample.get('problem_name')
            label = sample.get('label', 0.0)
            
            timestamp = sample.get('timestamp', 0)
            if isinstance(timestamp, str):
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    timestamp = int(dt.timestamp())  # 转换为时间戳整数
                except:
                    timestamp = hash(timestamp) if timestamp else 0
            elif timestamp is None:
                timestamp = 0
            elif not isinstance(timestamp, (int, float)):
                timestamp = 0
            
            if user_id is None or problem_name is None:
                continue
            
            score = float(label) if label is not None else 0.0
            class_label = score_to_class(score)
            binary_label = score_to_binary(score)
            
            user_interactions[user_id].append({
                'problem_name': problem_name,
                'score': score,
                'class_label': class_label,
                'binary_label': binary_label,
                'timestamp': timestamp
            })
        
        # 对每个用户的交互按时间戳排序
        for user_id in user_interactions:
            user_interactions[user_id].sort(key=lambda x: x['timestamp'])
        
        # 构建序列样本
        self.sequences = []
        
        # 统计题目和知识点（用于构建 embedding）
        all_problems = set()
        all_kcs = set()
        
        for user_id, interactions in user_interactions.items():
            # 为每个交互位置生成一个训练样本
            # 使用前 t 个交互预测第 t+1 个交互的结果
            for t in range(len(interactions)):
                # 历史交互序列（前 t 个）
                history = interactions[:t]
                # 当前交互（要预测的）
                current = interactions[t]
                
                # 跳过空历史（至少需要 1 个历史交互）
                if len(history) == 0:
                    continue
                
                self.sequences.append({
                    'user_id': user_id,
                    'history': history,
                    'current': current
                })
                
                # 收集题目和知识点
                all_problems.add(current['problem_name'])
                for inter in history + [current]:
                    all_problems.add(inter['problem_name'])
                    if self.use_rasch and inter['problem_name'] in self.kc_mapping:
                        all_kcs.update(self.kc_mapping[inter['problem_name']])
        
        # 构建 ID 映射
        self.problem_to_id = {prob: idx for idx, prob in enumerate(sorted(all_problems))}
        self.id_to_problem = {idx: prob for prob, idx in self.problem_to_id.items()}
        self.num_problems = len(self.problem_to_id)
        
        if self.use_rasch:
            # 从 kc_mapping 中收集所有知识点
            all_kcs_from_mapping = set()
            for prob in all_problems:
                if prob in self.kc_mapping:
                    all_kcs_from_mapping.update(self.kc_mapping[prob])
            
            # 如果没有从映射中收集到知识点，使用之前收集的
            if not all_kcs_from_mapping:
                all_kcs_from_mapping = all_kcs
            
            self.kc_to_id = {kc: idx for idx, kc in enumerate(sorted(all_kcs_from_mapping))}
            self.id_to_kc = {idx: kc for kc, idx in self.kc_to_id.items()}
            self.num_kcs = len(self.kc_to_id)
            
            # 构建题目ID到知识点ID列表的映射（用于 Rasch 分解）
            self.problem_id_to_kcs = {}
            for prob_name, prob_id in self.problem_to_id.items():
                if prob_name in self.kc_mapping:
                    kc_names = self.kc_mapping[prob_name]
                    kc_ids = [self.kc_to_id[kc] for kc in kc_names if kc in self.kc_to_id]
                    self.problem_id_to_kcs[prob_id] = kc_ids
                else:
                    self.problem_id_to_kcs[prob_id] = []
            
            print(f"知识点数: {self.num_kcs}")
            print(f"有知识点映射的题目数: {sum(1 for kcs in self.problem_id_to_kcs.values() if len(kcs) > 0)}")
        else:
            self.num_kcs = 0
            self.problem_id_to_kcs = {}
            print("不使用 Rasch 分解（简化版）")
        
        print(f"题目数: {self.num_problems}")
        print(f"学生数: {len(user_interactions)}")
        print(f"序列样本数: {len(self.sequences)}")
        
        # 计算类别分布
        class_dist = defaultdict(int)
        for seq in self.sequences:
            class_dist[seq['current']['class_label']] += 1
        print(f"\n类别分布:")
        for cls in range(5):
            print(f"  类别{cls} ({CLASS_TO_SCORE_RANGE[cls]}): {class_dist[cls]}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        history = seq_data['history']
        current = seq_data['current']
        
        # 限制历史序列长度
        if len(history) > self.max_seq_len:
            history = history[-self.max_seq_len:]  # 取最近的 max_seq_len 个
        
        problem_ids = []
        response_labels = []  # 二分类：0=错误, 1=正确
        kc_ids_list = []  # 每个题目对应的知识点列表（如果使用 Rasch）
        
        for inter in history:
            prob_id = self.problem_to_id[inter['problem_name']]
            problem_ids.append(prob_id)
            response_labels.append(inter['binary_label'])
            
            if self.use_rasch:
                kcs = self.kc_mapping.get(inter['problem_name'], [])
                kc_ids = [self.kc_to_id[kc] for kc in kcs if kc in self.kc_to_id]
                kc_ids_list.append(kc_ids)
        
        current_prob_id = self.problem_to_id[current['problem_name']]
        current_class_label = current['class_label']
        
        return {
            'problem_ids': torch.tensor(problem_ids, dtype=torch.long),
            'response_labels': torch.tensor(response_labels, dtype=torch.long),
            'current_prob_id': torch.tensor(current_prob_id, dtype=torch.long),
            'current_class_label': torch.tensor(current_class_label, dtype=torch.long),
            'kc_ids_list': kc_ids_list if self.use_rasch else None,
            'seq_len': len(problem_ids)
        }


class SimpleKTModel(nn.Module):
    """SimpleKT 模型"""
    def __init__(self, num_problems, embed_dim=128, num_heads=4, num_layers=2, 
                 dropout=0.3, num_classes=5, use_rasch=False, num_kcs=0, 
                 kc_mapping=None, problem_id_to_kcs=None):
        super().__init__()
        self.num_problems = num_problems
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_rasch = use_rasch
        self.kc_mapping = kc_mapping or {}
        self.problem_id_to_kcs = problem_id_to_kcs or {}
        
        self.problem_embedding = nn.Embedding(num_problems, embed_dim)
        
        # Rasch 分解
        if use_rasch and num_kcs > 0:
            self.kc_embedding = nn.Embedding(num_kcs, embed_dim)
            self.problem_delta = nn.Embedding(num_problems, embed_dim)  # 题目特异偏移
            print(f"使用 Rasch 分解: 知识点数={num_kcs}")
        else:
            self.kc_embedding = None
            self.problem_delta = None
        
        self.response_embedding = nn.Embedding(2, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 2,  # 问题embedding + 作答embedding 拼接
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输入维度：历史状态(embed_dim*2) + 当前题目embedding(embed_dim) = embed_dim*3
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def get_problem_embedding(self, problem_ids):
        if self.use_rasch and self.kc_embedding is not None and self.problem_delta is not None:
            # Rasch 分解：q_j = mean(kc_embeddings) + delta_j
            if problem_ids.dim() == 1:
                batch_size = problem_ids.shape[0]
                problem_emb = torch.zeros(batch_size, self.embed_dim, 
                                         device=problem_ids.device, dtype=torch.float32)
                
                for b in range(batch_size):
                    prob_id = problem_ids[b].item()
                    if prob_id in self.problem_id_to_kcs:
                        kc_ids = self.problem_id_to_kcs[prob_id]
                        if len(kc_ids) > 0:
                            kc_ids_tensor = torch.tensor(kc_ids, device=problem_ids.device, dtype=torch.long)
                            kc_embs = self.kc_embedding(kc_ids_tensor)  # [num_kcs, embed_dim]
                            kc_mean = kc_embs.mean(dim=0)  # [embed_dim]
                            delta = self.problem_delta(problem_ids[b])  # [embed_dim]
                            problem_emb[b] = kc_mean + delta
                        else:
                            problem_emb[b] = self.problem_embedding(problem_ids[b])
                    else:
                        problem_emb[b] = self.problem_embedding(problem_ids[b])
            else:
                batch_size, seq_len = problem_ids.shape
                problem_emb = torch.zeros(batch_size, seq_len, self.embed_dim, 
                                         device=problem_ids.device, dtype=torch.float32)
                
                for b in range(batch_size):
                    for t in range(seq_len):
                        prob_id = problem_ids[b, t].item()
                        if prob_id in self.problem_id_to_kcs:
                            kc_ids = self.problem_id_to_kcs[prob_id]
                            if len(kc_ids) > 0:
                                kc_ids_tensor = torch.tensor(kc_ids, device=problem_ids.device, dtype=torch.long)
                                kc_embs = self.kc_embedding(kc_ids_tensor)  # [num_kcs, embed_dim]
                                kc_mean = kc_embs.mean(dim=0)  # [embed_dim]
                                delta = self.problem_delta(problem_ids[b, t])  # [embed_dim]
                                problem_emb[b, t] = kc_mean + delta
                            else:
                                problem_emb[b, t] = self.problem_embedding(problem_ids[b, t])
                        else:
                            problem_emb[b, t] = self.problem_embedding(problem_ids[b, t])
        else:
            problem_emb = self.problem_embedding(problem_ids)
        
        return problem_emb
    
    def forward(self, problem_ids, response_labels, current_prob_id, seq_lens=None):
        batch_size, seq_len = problem_ids.shape
        
        # 1. 获取历史题目 embedding
        hist_problem_emb = self.get_problem_embedding(problem_ids)  # [batch_size, seq_len, embed_dim]
        
        # 2. 获取当前题目 embedding
        curr_problem_emb = self.get_problem_embedding(current_prob_id)  # [batch_size, embed_dim]
        
        # 3. 获取作答 embedding
        response_emb = self.response_embedding(response_labels)  # [batch_size, seq_len, embed_dim]
        
        # 4. 拼接历史输入：问题 + 作答
        input_emb = torch.cat([hist_problem_emb, response_emb], dim=-1)  # [batch_size, seq_len, embed_dim*2]
        
        # 5. 创建 padding mask（如果提供了 seq_lens）
        if seq_lens is not None:
            mask = torch.arange(seq_len, device=problem_ids.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
            mask = mask  # [batch_size, seq_len]
        else:
            mask = None
        
        # 6. Transformer Encoder 编码历史序列
        encoded = self.transformer(input_emb, src_key_padding_mask=mask)  # [batch_size, seq_len, embed_dim*2]
        
        # 7. 聚合历史状态：取最后一个非padding位置的表示
        if seq_lens is not None:
            # 取每个序列的最后一个有效位置
            batch_indices = torch.arange(batch_size, device=problem_ids.device)
            last_indices = (seq_lens - 1).clamp(min=0)
            aggregated = encoded[batch_indices, last_indices]  # [batch_size, embed_dim*2]
        else:
            # 如果没有 mask，取最后一个位置
            aggregated = encoded[:, -1, :]  # [batch_size, embed_dim*2]
        
        # 8. 融合当前题目信息：将历史状态和当前题目embedding拼接
        combined = torch.cat([aggregated, curr_problem_emb], dim=-1)  # [batch_size, embed_dim*3]
        
        # 9. 输出层预测
        logits = self.output_layer(combined)  # [batch_size, num_classes]
        
        return logits


def collate_fn(batch):
    batch_size = len(batch)
    # 找到最大序列长度
    max_len = max(item['seq_len'] for item in batch)
    
    # Padding
    problem_ids_list = []
    response_labels_list = []
    current_prob_ids = []
    current_class_labels = []
    seq_lens = []
    
    for item in batch:
        seq_len = item['seq_len']
        problem_ids = item['problem_ids']
        response_labels = item['response_labels']
        
        # Padding 到 max_len
        pad_len = max_len - seq_len
        if pad_len > 0:
            problem_ids = torch.cat([problem_ids, torch.zeros(pad_len, dtype=torch.long)])
            response_labels = torch.cat([response_labels, torch.zeros(pad_len, dtype=torch.long)])
        
        problem_ids_list.append(problem_ids)
        response_labels_list.append(response_labels)
        current_prob_ids.append(item['current_prob_id'])
        current_class_labels.append(item['current_class_label'])
        seq_lens.append(seq_len)
    
    return {
        'problem_ids': torch.stack(problem_ids_list),  # [batch_size, max_len]
        'response_labels': torch.stack(response_labels_list),  # [batch_size, max_len]
        'current_prob_id': torch.stack(current_prob_ids),  # [batch_size]
        'current_class_label': torch.stack(current_class_labels),  # [batch_size]
        'seq_lens': torch.tensor(seq_lens, dtype=torch.long)  # [batch_size]
    }


def train_simplekt(model, train_loader, val_loader, num_epochs=30, lr=1e-3, 
                   device='cuda', use_class_weights=True):
    model = model.to(device)
    
    if use_class_weights:
        from collections import Counter
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['current_class_label'].cpu().numpy().tolist())
        label_counts = Counter(all_labels)
        total = len(all_labels)
        num_classes = 5
        class_weights = torch.tensor([
            total / (num_classes * label_counts.get(i, 1)) 
            for i in range(num_classes)
        ], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            problem_ids = batch['problem_ids'].to(device)
            response_labels = batch['response_labels'].to(device)
            current_prob_ids = batch['current_prob_id'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            labels = batch['current_class_label'].to(device)
            
            optimizer.zero_grad()
            logits = model(problem_ids, response_labels, current_prob_id=current_prob_ids, seq_lens=seq_lens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                problem_ids = batch['problem_ids'].to(device)
                response_labels = batch['response_labels'].to(device)
                current_prob_ids = batch['current_prob_id'].to(device)
                seq_lens = batch['seq_lens'].to(device)
                labels = batch['current_class_label'].to(device)
                
                logits = model(problem_ids, response_labels, current_prob_id=current_prob_ids, seq_lens=seq_lens)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        old_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        lr_info = ""
        if old_lr != current_lr:
            lr_info = f" (LR reduced: {old_lr:.6f} -> {current_lr:.6f})"
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}{lr_info}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'simplekt_baseline_best.pth')
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")
    
    return model, all_preds, all_labels


def evaluate_simplekt(model, val_loader, device='cuda'):
    """评估 SimpleKT 模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            problem_ids = batch['problem_ids'].to(device)
            response_labels = batch['response_labels'].to(device)
            current_prob_ids = batch['current_prob_id'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            labels = batch['current_class_label'].to(device)
            
            logits = model(problem_ids, response_labels, current_prob_id=current_prob_ids, seq_lens=seq_lens)
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # MAE
    y_true_values = np.array([CLASS_TO_VALUE[c] for c in all_labels])
    y_pred_values = np.array([CLASS_TO_VALUE[c] for c in all_preds])
    mae = np.mean(np.abs(y_true_values - y_pred_values))
    
    # F1
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Precision
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Recall
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': {CLASS_TO_SCORE_RANGE[i]: f1_per_class[i] for i in range(5)},
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'precision_per_class': {CLASS_TO_SCORE_RANGE[i]: precision_per_class[i] for i in range(5)},
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'recall_per_class': {CLASS_TO_SCORE_RANGE[i]: recall_per_class[i] for i in range(5)},
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    from datasets import load_dataset
    sample_data = load_dataset('json', data_files=train_file)['train']
    has_e_field = any('e' in sample for sample in sample_data)
    
    if has_e_field:
        print("检测到 'e' 字段，将尝试提取知识点并启用 Rasch 分解")
        use_rasch = True
    else:
        print("未检测到 'e' 字段，不做 Rasch 分解")
        use_rasch = False
    
    train_dataset = SimpleKTDataset(
        train_file, 
        max_seq_len=200,
        remove_duplicates=False,
        use_rasch=use_rasch,  
        kc_mapping=None  
    )
    
    from datasets import load_dataset
    val_raw = load_dataset('json', data_files=val_file)['train']
    val_problems = set()
    for sample in val_raw:
        problem_name = sample.get('problem_name')
        if problem_name:
            val_problems.add(problem_name)
    
    max_problem_id = max(train_dataset.problem_to_id.values()) if train_dataset.problem_to_id else -1
    for prob in val_problems:
        if prob not in train_dataset.problem_to_id:
            max_problem_id += 1
            train_dataset.problem_to_id[prob] = max_problem_id
            train_dataset.id_to_problem[max_problem_id] = prob
    
    train_dataset.num_problems = len(train_dataset.problem_to_id)
    
    val_dataset = SimpleKTDataset(
        val_file,
        max_seq_len=200,
        train_file=train_file,
        remove_duplicates=False,
        use_rasch=train_dataset.use_rasch,  
        kc_mapping=train_dataset.kc_mapping  
    )
    
    # 确保验证集使用相同的题目ID映射和知识点映射（从训练集继承）
    val_dataset.problem_to_id = train_dataset.problem_to_id
    val_dataset.id_to_problem = train_dataset.id_to_problem
    val_dataset.num_problems = train_dataset.num_problems
    val_dataset.kc_mapping = train_dataset.kc_mapping
    val_dataset.kc_to_id = train_dataset.kc_to_id
    val_dataset.id_to_kc = train_dataset.id_to_kc
    val_dataset.num_kcs = train_dataset.num_kcs
    val_dataset.problem_id_to_kcs = train_dataset.problem_id_to_kcs
    
    if train_dataset.use_rasch:
        print(f"知识点数: {train_dataset.num_kcs}")
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    use_rasch = train_dataset.use_rasch and train_dataset.num_kcs > 0
    print(f"使用 Rasch 分解: {use_rasch}")
    if use_rasch:
        print(f"  知识点数: {train_dataset.num_kcs}")
        print(f"  有知识点映射的题目数: {sum(1 for kcs in train_dataset.problem_id_to_kcs.values() if len(kcs) > 0)}")
    
    model = SimpleKTModel(
        num_problems=train_dataset.num_problems,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        num_classes=5,
        use_rasch=use_rasch,
        num_kcs=train_dataset.num_kcs if use_rasch else 0,
        kc_mapping=train_dataset.kc_mapping if use_rasch else None,
        problem_id_to_kcs=train_dataset.problem_id_to_kcs if use_rasch else None
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    model, val_preds, val_labels = train_simplekt(
        model, train_loader, val_loader,
        num_epochs=60,
        lr=1e-3,
        device='cuda',
        use_class_weights=True
    )
    
    # 加载最佳模型并评估
    model.load_state_dict(torch.load('simplekt_baseline_best.pth'))
    results = evaluate_simplekt(model, val_loader, device='cuda')
    
    # 打印结果
    print(f"\n{'='*70}")
    print("SimpleKT Baseline 结果:")
    print(f"{'='*70}")
    print(f"准确率 (Accuracy): {results['accuracy']:.2%}")
    print(f"MAE: {results['mae']:.2f} 分")
    
    print(f"\n宏平均指标:")
    print(f"  Precision (Macro): {results['precision_macro']:.4f}")
    print(f"  Recall (Macro): {results['recall_macro']:.4f}")
    print(f"  F1 (Macro): {results['f1_macro']:.4f}")
    
    print(f"\n加权平均指标:")
    print(f"  Precision (Weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (Weighted): {results['recall_weighted']:.4f}")
    print(f"  F1 (Weighted): {results['f1_weighted']:.4f}")
    
    print(f"\n各类别详细指标:")
    print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    for class_name in CLASS_TO_SCORE_RANGE.values():
        i = SCORE_RANGE_TO_CLASS[class_name]
        print(f"{class_name:<12} {results['precision_per_class'][class_name]:<12.4f} "
              f"{results['recall_per_class'][class_name]:<12.4f} "
              f"{results['f1_per_class'][class_name]:<12.4f}")
    
    print(f"\n混淆矩阵:")
    class_names = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    print(f"{'真实类别':<12} {'0-20':<8} {'20-40':<8} {'40-60':<8} {'60-80':<8} {'80-100':<8}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:<12}"
        for j in range(5):
            row_str += f"{results['confusion_matrix'][i, j]:<8}"
        print(row_str)


if __name__ == "__main__":
    main()

