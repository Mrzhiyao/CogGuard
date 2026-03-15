import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 配置
model_path = './Qwen2.5-3B-Instruct'
train_file = './datasets/train_processed_40_new_qwen25_3b_reformatted.jsonl'
val_file = './datasets/validation_processed_40_new_qwen25_3b.jsonl'

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

SCORE_START_TOKEN = 151666


class MLPDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=1024, train_file=None, remove_duplicates=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        dataset = load_dataset('json', data_files=data_file)['train']
        
        if train_file and remove_duplicates:
            train_dataset = load_dataset('json', data_files=train_file)['train']
            train_input_ids_set = set()
            for sample in train_dataset:
                if 'input_ids' in sample:
                    train_input_ids_set.add(tuple(sample['input_ids']))            
            unique_samples = []
            for sample in dataset:
                if 'input_ids' in sample:
                    input_ids_tuple = tuple(sample['input_ids'])
                    if input_ids_tuple not in train_input_ids_set:
                        unique_samples.append(sample)
            

            dataset = unique_samples
        elif train_file and not remove_duplicates:
            print(" 警告: 未启用去重，验证集可能包含与训练集重复的样本（可能导致数据泄露）")
            print(f"验证集样本数: {len(dataset)}")
        
        self.samples = []
        for sample in dataset:
            input_ids = sample['input_ids']
            if SCORE_START_TOKEN in input_ids:
                idx = len(input_ids) - 1 - input_ids[::-1].index(SCORE_START_TOKEN)
                input_ids = input_ids[:idx + 1]  # 包含151666
            
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            
            if 'label' in sample and sample['label'] is not None:
                score = float(sample['label'])
                if score < 20:
                    label = 0
                elif score < 40:
                    label = 1
                elif score < 60:
                    label = 2
                elif score < 80:
                    label = 3
                else:
                    label = 4
            else:
                label = 0  # 默认值
            
            self.samples.append({
                'input_ids': input_ids,
                'label': label
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class MLPClassifier(nn.Module):
    def __init__(self, qwen_model, hidden_dims=[512, 256, 128], num_classes=5, dropout=0.3, freeze_qwen=True, pooling='last'):
        super().__init__()
        self.qwen_model = qwen_model
        self.pooling = pooling
        
        if freeze_qwen:
            for param in self.qwen_model.parameters():
                param.requires_grad = False
            print("Qwen模型已冻结，只训练MLP分类头")

        
        hidden_size = qwen_model.config.hidden_size
        
        mlp_layers = []
        input_dim = hidden_size
        
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, input_ids, attention_mask=None):
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.qwen_model.parameters())):
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        if self.pooling == 'last':
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1因为索引从0开始
                sentence_embedding = last_hidden_state[torch.arange(last_hidden_state.size(0)), seq_lengths]
            else:
                sentence_embedding = last_hidden_state[:, -1, :]
        elif self.pooling == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
                sentence_embedding = sum_embeddings / sum_mask
            else:
                sentence_embedding = last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        sentence_embedding = sentence_embedding.float()
        
        logits = self.mlp(sentence_embedding)  # [batch_size, num_classes]
        
        return logits


def collate_fn(batch, tokenizer):
    input_ids_list = [item['input_ids'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    max_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids = []
    attention_masks = []
    
    for input_ids in input_ids_list:
        pad_len = max_len - len(input_ids)
        padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * pad_len)
        attention_masks.append([1] * len(input_ids) + [0] * pad_len)
    
    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'label': labels
    }


def train_mlp(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cuda', use_class_weights=True):
    model = model.to(device)
    
    if use_class_weights:
        from collections import Counter
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['label'].cpu().numpy().tolist())
        label_counts = Counter(all_labels)
        total = len(all_labels)
        num_classes = 5
        
        max_count = max(label_counts.values())
        class_weights = torch.tensor([
            (max_count / label_counts.get(i, 1)) ** 0.75  # 使用0.75次方，比线性更强但不过度
            for i in range(num_classes)
        ], dtype=torch.float32, device=device)
        
        class_weights = class_weights / class_weights.mean()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 只优化MLP的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-6, betas=(0.9, 0.999))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=3, min_lr=1e-6
    )
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        old_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        lr_info = ""
        if old_lr != current_lr:
            lr_info = f" (LR reduced: {old_lr:.6f} -> {current_lr:.6f})"
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}{lr_info}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.mlp.state_dict(), 'mlp_baseline_best.pth')
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")
    
    return model, all_preds, all_labels


def evaluate_mlp(model, val_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask=attention_mask)
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 已经提前完成去重，所以False
    remove_duplicates = False
    
    train_dataset = MLPDataset(train_file, tokenizer, max_length=1024, remove_duplicates=False)  # 训练集不需要去重
    
    val_dataset = MLPDataset(val_file, tokenizer, max_length=1024, train_file=train_file, remove_duplicates=remove_duplicates)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # 减小batch size，因为要加载完整模型
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    model = MLPClassifier(
        qwen_model=qwen_model,
        hidden_dims=[512, 256, 128],
        num_classes=5,
        dropout=0.2,  # 降低dropout（从0.3改为0.2）
        freeze_qwen=True,  # 冻结Qwen，只训练MLP
        pooling='mean'  # 改为mean pooling（通常比last token更好）
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    model, val_preds, val_labels = train_mlp(
        model, train_loader, val_loader,
        num_epochs=150,  # 增加训练epoch数（从30改为100）
        lr=1e-3,  # 稍微提高学习率（从5e-4改为8e-4），加快学习速度
        device='cuda',
        use_class_weights=True  # 启用类别权重
    )
    
    # 加载最佳模型并评估
    print("\n加载最佳模型进行评估...")
    model.mlp.load_state_dict(torch.load('mlp_baseline_best.pth'))
    results = evaluate_mlp(model, val_loader, device='cuda')
    
    # 打印结果
    print(f"\n{'='*70}")
    print("SLM_Probe Baseline 结果:")
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
        print(f"{class_name:<12} {results['precision_per_class'][class_name]:<12.4f} {results['recall_per_class'][class_name]:<12.4f} {results['f1_per_class'][class_name]:<12.4f}")
    
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
