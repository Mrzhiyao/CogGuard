import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
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


def build_graph_from_tokens(entity_graph_ids, knowledge_graph_ids, use_type_embedding=True):
    # 从token序列构建图结构
    all_tokens = []
    token_types = []  # 0 for entity, 1 for knowledge
    
    for token in entity_graph_ids:
        all_tokens.append(token)
        token_types.append(0)  # entity
    
    for token in knowledge_graph_ids:
        all_tokens.append(token)
        token_types.append(1)  # knowledge
    
    if len(all_tokens) == 0:
        return Data(
            node_tokens=torch.tensor([0], dtype=torch.long),  # 使用0作为虚拟token
            node_types=torch.tensor([0], dtype=torch.long),  # 默认type为0
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=1
        )
    
    unique_tokens = []
    token_to_node = {}
    node_types_list = []
    
    for token, token_type in zip(all_tokens, token_types):
        if token not in token_to_node:
            unique_tokens.append(token)
            token_to_node[token] = len(unique_tokens) - 1
            node_types_list.append(token_type)
        else:
            # 如果token已存在，如果当前是entity类型（0），则更新type（entity优先）
            node_idx = token_to_node[token]
            if token_type == 0 and node_types_list[node_idx] == 1:
                node_types_list[node_idx] = 0
    
    num_nodes = len(unique_tokens)
    
    # 构建边（基于序列中的相邻关系）
    edges = []
    sequences = [
        (list(entity_graph_ids), 0),      # entity序列，type=0
        (list(knowledge_graph_ids), 1)    # knowledge序列，type=1
    ]
    
    for seq, seq_type in sequences:
        for i in range(len(seq) - 1):
            token1 = seq[i]
            token2 = seq[i + 1]
            if token1 in token_to_node and token2 in token_to_node:
                node1 = token_to_node[token1]
                node2 = token_to_node[token2]
                # 添加无向边（两个方向）
                edges.append([node1, node2])
                edges.append([node2, node1])
    
    # 去重边
    if len(edges) > 0:
        edges = list(set(tuple(e) for e in edges))
        edges = [list(e) for e in edges]
    else:
        edges = [[0, 0]]
    
    # 存储token ids（用于后续通过Embedding获取特征）
    node_tokens = torch.tensor(unique_tokens, dtype=torch.long)
    node_types = torch.tensor(node_types_list, dtype=torch.long)  # [num_nodes]
    
    # 构建边索引
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(
        node_tokens=node_tokens, 
        node_types=node_types,  # 新增：节点类型
        edge_index=edge_index, 
        num_nodes=num_nodes
    )


class GraphDataset(Dataset):
    def __init__(self, data_file, train_file=None, remove_duplicates=True):
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
            entity_graph_ids = sample.get('entity_graph_ids', [])
            knowledge_graph_ids = sample.get('knowledge_graph_ids', [])
            
            graph = build_graph_from_tokens(
                entity_graph_ids,
                knowledge_graph_ids,
                use_type_embedding=True  # 启用type embedding
            )
            
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
                'graph': graph,
                'label': label
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class GraphClassifier(nn.Module):
    def __init__(self, qwen_model, gnn_hidden=256, mlp_hidden=[256, 128], num_classes=5, dropout=0.3, freeze_embedding=True, use_type_embedding=True, type_embedding_dim=8):
        super().__init__()
        self.qwen_model = qwen_model
        self.use_type_embedding = use_type_embedding
        
        # 获取embedding layer
        self.embedding = qwen_model.get_input_embeddings()
        self.embedding_dim = self.embedding.embedding_dim
        
        # 冻结embedding layer
        if freeze_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False
            print("Embedding Layer已冻结")
        else:
            print("Embedding Layer可训练")
        
        if use_type_embedding:
            self.type_embedding = nn.Embedding(2, type_embedding_dim)  # 2种类型：0=entity, 1=knowledge
            gnn_input_dim = self.embedding_dim + type_embedding_dim
        else:
            self.type_embedding = None
            gnn_input_dim = self.embedding_dim
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            GCNConv(gnn_input_dim, gnn_hidden),
            GCNConv(gnn_hidden, gnn_hidden),
        ])
        
        # MLP分类头
        mlp_layers = []
        input_dim = gnn_hidden * 2  # mean + max pooling
        
        for hidden_dim in mlp_hidden:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, batch):

        node_tokens = batch.node_tokens  # [num_nodes] - 所有节点的token ids
        edge_index = batch.edge_index  # [2, num_edges]
        batch_idx = batch.batch  # [num_nodes] - 指示每个节点属于哪个图
        
        # 通过Embedding获取节点特征
        with torch.set_grad_enabled(self.embedding.weight.requires_grad):
            x = self.embedding(node_tokens)  # [num_nodes, embedding_dim]
        
        # 添加Type Embedding
        if self.use_type_embedding and hasattr(batch, 'node_types'):
            node_types = batch.node_types  # [num_nodes] - 节点类型（0=entity, 1=knowledge）
            type_emb = self.type_embedding(node_types)  # [num_nodes, type_embedding_dim]
            x = torch.cat([x, type_emb], dim=1)  # [num_nodes, embedding_dim + type_embedding_dim]
        elif self.use_type_embedding:
            node_types = torch.zeros(node_tokens.size(0), dtype=torch.long, device=node_tokens.device)
            type_emb = self.type_embedding(node_types)
            x = torch.cat([x, type_emb], dim=1)
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
        
        x_mean = global_mean_pool(x, batch_idx)  
        x_max = global_max_pool(x, batch_idx)  
        x_pooled = torch.cat([x_mean, x_max], dim=1) 
        
        # MLP分类
        logits = self.mlp(x_pooled) 
        
        return logits


def collate_graphs(batch):
    graphs = [item['graph'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    batch_graph = Batch.from_data_list(graphs)
    
    return {
        'graph': batch_graph,
        'label': labels
    }


def train_graph(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            graph = batch['graph'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(graph)
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
                graph = batch['graph'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(graph)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = {
                'gnn': model.gnn_layers.state_dict(),
                'mlp': model.mlp.state_dict()
            }
            if model.use_type_embedding and model.type_embedding is not None:
                state_dict['type_embedding'] = model.type_embedding.state_dict()
            torch.save(state_dict, 'graph_baseline_best.pth')
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")
    
    return model, all_preds, all_labels


def evaluate_graph(model, val_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            graph = batch['graph'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(graph)
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
    # 加载tokenizer和模型（只用于获取embedding layer）
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
    
    # 已经提前完成去重
    remove_duplicates = False  # 设置为False可以不去重
    
    train_dataset = GraphDataset(train_file, remove_duplicates=False)  # 训练集不需要去重
    
    val_dataset = GraphDataset(val_file, train_file=train_file, remove_duplicates=remove_duplicates)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_graphs)
    
    # 创建模型
    model = GraphClassifier(
        qwen_model=qwen_model,
        gnn_hidden=256,
        mlp_hidden=[256, 128],
        num_classes=5,
        dropout=0.3,
        freeze_embedding=True, 
        use_type_embedding=True,  
        type_embedding_dim=8  
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    model, val_preds, val_labels = train_graph(
        model, train_loader, val_loader,
        num_epochs=600,
        lr=1e-3,
        device='cuda'
    )
    
    state_dict = torch.load('graph_baseline_best.pth')
    model.gnn_layers.load_state_dict(state_dict['gnn'])
    model.mlp.load_state_dict(state_dict['mlp'])
    if 'type_embedding' in state_dict and model.use_type_embedding and model.type_embedding is not None:
        model.type_embedding.load_state_dict(state_dict['type_embedding'])
    
    results = evaluate_graph(model, val_loader, device='cuda')
    
    # 打印加载的模型信息
    print(f"\n模型加载信息:")
    print(f"  - GNN层数: {len(model.gnn_layers)}")
    print(f"  - MLP层数: {len(model.mlp)}")
    print(f"  - 使用Type Embedding: {model.use_type_embedding}")
    if model.use_type_embedding and model.type_embedding is not None:
        print(f"  - Type Embedding维度: {model.type_embedding.embedding_dim}")
    
    # 打印结果
    print(f"\n{'='*70}")
    print("Graph-only Baseline 结果:")
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
