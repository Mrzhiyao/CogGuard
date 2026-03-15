"""
图谱构建统计模块
用于记录和对比不同版本的图谱构建性能指标

适用于 ACL 投稿的性能对比
"""
import json
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil
import sys


@dataclass
class GraphBuildStats:
    """
    图谱构建统计信息
    """
    # 基本信息
    version: str = "unknown"  # 版本标识（如 "baseline", "optimized"）
    model_name: str = ""
    total_records: int = 0
    unique_questions: int = 0
    
    # 时间统计
    total_build_time: float = 0.0  # 总构建时间（秒）
    avg_time_per_record: float = 0.0  # 平均每条记录处理时间（秒）
    avg_time_per_question: float = 0.0  # 平均每个 question 处理时间（秒）
    
    # LLM 调用统计
    total_llm_calls: int = 0  # 总 LLM 调用次数
    total_llm_time: float = 0.0  # 总 LLM 调用时间（秒）
    avg_llm_time_per_call: float = 0.0  # 平均每次 LLM 调用时间（秒）
    llm_timeout_count: int = 0  # LLM 超时次数
    llm_error_count: int = 0  # LLM 错误次数
    
    # 图谱操作统计
    total_graph_inserts: int = 0  # 总插入次数
    total_graph_insert_time: float = 0.0  # 总插入时间（秒）
    avg_graph_insert_time: float = 0.0  # 平均每次插入时间（秒）
    clustering_count: int = 0  # 聚类次数
    total_clustering_time: float = 0.0  # 总聚类时间（秒）
    avg_clustering_time: float = 0.0  # 平均每次聚类时间（秒）
    
    # 图谱规模统计
    final_node_count: int = 0  # 最终节点数
    final_edge_count: int = 0  # 最终边数
    avg_nodes_per_record: float = 0.0  # 平均每条记录产生的节点数
    avg_edges_per_record: float = 0.0  # 平均每条记录产生的边数
    
    # 吞吐量统计
    throughput_records_per_sec: float = 0.0  # 记录吞吐量（条/秒）
    throughput_tokens_per_sec: float = 0.0  # Token 吞吐量（token/秒，如果可获取）
    
    # 缓存统计
    question_cache_hits: int = 0  # Question 缓存命中次数
    question_cache_misses: int = 0  # Question 缓存未命中次数
    question_cache_hit_rate: float = 0.0  # Question 缓存命中率
    
    # 优化相关统计
    pre_extract_time: float = 0.0  # 预提取时间（秒）
    batch_processing_time: float = 0.0  # 批处理时间（秒）
    batch_count: int = 0  # 批处理批次数量
    
    # 资源使用统计
    peak_memory_mb: float = 0.0  # 峰值内存使用（MB）
    avg_cpu_percent: float = 0.0  # 平均 CPU 使用率（%）
    
    # 错误统计
    total_errors: int = 0  # 总错误数
    skipped_records: int = 0  # 跳过的记录数
    success_rate: float = 0.0  # 成功率
    
    # 详细时间记录
    insert_times: List[float] = None  # 每次插入的时间记录
    llm_call_times: List[float] = None  # 每次 LLM 调用的时间记录
    clustering_times: List[float] = None  # 每次聚类的时间记录
    
    # 时间戳
    start_time: str = ""
    end_time: str = ""
    
    def __post_init__(self):
        if self.insert_times is None:
            self.insert_times = []
        if self.llm_call_times is None:
            self.llm_call_times = []
        if self.clustering_times is None:
            self.clustering_times = []
    
    def to_dict(self) -> Dict:
        """转换为字典（用于 JSON 序列化）"""
        data = asdict(self)
        # 确保列表字段存在
        if data.get('insert_times') is None:
            data['insert_times'] = []
        if data.get('llm_call_times') is None:
            data['llm_call_times'] = []
        if data.get('clustering_times') is None:
            data['clustering_times'] = []
        return data
    
    def calculate_derived_metrics(self):
        """计算派生指标"""
        # 平均时间
        if self.total_records > 0:
            self.avg_time_per_record = self.total_build_time / self.total_records
        if self.unique_questions > 0:
            self.avg_time_per_question = self.total_build_time / self.unique_questions
        
        # LLM 统计
        if self.total_llm_calls > 0:
            self.avg_llm_time_per_call = self.total_llm_time / self.total_llm_calls
        
        # 图谱操作统计
        if self.total_graph_inserts > 0:
            self.avg_graph_insert_time = self.total_graph_insert_time / self.total_graph_inserts
        if self.clustering_count > 0:
            self.avg_clustering_time = self.total_clustering_time / self.clustering_count
        
        # 图谱规模统计
        if self.total_records > 0:
            self.avg_nodes_per_record = self.final_node_count / self.total_records
            self.avg_edges_per_record = self.final_edge_count / self.total_records
        
        # 吞吐量
        if self.total_build_time > 0:
            self.throughput_records_per_sec = self.total_records / self.total_build_time
        
        # 缓存命中率
        total_cache_requests = self.question_cache_hits + self.question_cache_misses
        if total_cache_requests > 0:
            self.question_cache_hit_rate = self.question_cache_hits / total_cache_requests
        
        # 成功率
        total_attempts = self.total_records + self.skipped_records
        if total_attempts > 0:
            self.success_rate = self.total_records / total_attempts


class GraphBuildStatsCollector:
    """
    图谱构建统计收集器
    """
    def __init__(self, version: str = "unknown", model_name: str = ""):
        self.stats = GraphBuildStats(version=version, model_name=model_name)
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.cpu_samples = []
        self.start_time = time.time()
        self.stats.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def record_llm_call(self, call_time: float, success: bool = True, is_timeout: bool = False):
        """记录 LLM 调用"""
        self.stats.total_llm_calls += 1
        self.stats.total_llm_time += call_time
        self.stats.llm_call_times.append(call_time)
        if not success:
            if is_timeout:
                self.stats.llm_timeout_count += 1
            else:
                self.stats.llm_error_count += 1
            self.stats.total_errors += 1
    
    def record_graph_insert(self, insert_time: float, success: bool = True):
        """记录图谱插入操作"""
        self.stats.total_graph_inserts += 1
        self.stats.total_graph_insert_time += insert_time
        self.stats.insert_times.append(insert_time)
        if not success:
            self.stats.total_errors += 1
            self.stats.skipped_records += 1
    
    def record_clustering(self, clustering_time: float):
        """记录聚类操作"""
        self.stats.clustering_count += 1
        self.stats.total_clustering_time += clustering_time
        self.stats.clustering_times.append(clustering_time)
    
    def record_question_cache(self, hit: bool):
        """记录 Question 缓存"""
        if hit:
            self.stats.question_cache_hits += 1
        else:
            self.stats.question_cache_misses += 1
    
    def record_pre_extract(self, pre_extract_time: float):
        """记录预提取时间"""
        self.stats.pre_extract_time += pre_extract_time
    
    def record_batch(self, batch_time: float):
        """记录批处理时间"""
        self.stats.batch_count += 1
        self.stats.batch_processing_time += batch_time
    
    def update_graph_size(self, node_count: int, edge_count: int):
        """更新图谱规模"""
        self.stats.final_node_count = node_count
        self.stats.final_edge_count = edge_count
    
    def update_record_counts(self, total_records: int, unique_questions: int):
        """更新记录数量"""
        self.stats.total_records = total_records
        self.stats.unique_questions = unique_questions
    
    def sample_cpu(self):
        """采样 CPU 使用率"""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
        except:
            pass
    
    def finalize(self):
        """完成统计，计算所有派生指标"""
        end_time = time.time()
        self.stats.total_build_time = end_time - self.start_time
        self.stats.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算内存使用
        try:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.stats.peak_memory_mb = max(current_memory, self.start_memory)
        except:
            pass
        
        # 计算平均 CPU
        if self.cpu_samples:
            self.stats.avg_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples)
        
        # 计算派生指标
        self.stats.calculate_derived_metrics()
    
    def save(self, output_dir: str, filename: str = "build_stats.json"):
        """
        保存统计信息到文件
        
        Args:
            output_dir: 输出目录
            filename: 文件名（默认 build_stats.json）
        """
        self.finalize()
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # 转换为字典
        stats_dict = self.stats.to_dict()
        
        # 添加一些额外的元数据
        stats_dict['metadata'] = {
            'python_version': sys.version,
            'platform': sys.platform,
            'collected_at': datetime.now().isoformat(),
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n[统计] 构建统计信息已保存到: {output_path}")
        
        # 同时保存一个人类可读的摘要
        self.save_summary(output_dir, filename.replace('.json', '_summary.txt'))
    
    def save_summary(self, output_dir: str, filename: str = "build_stats_summary.txt"):
        """保存人类可读的统计摘要"""
        self.finalize()
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("图谱构建统计摘要\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"版本: {self.stats.version}\n")
            f.write(f"模型: {self.stats.model_name}\n")
            f.write(f"开始时间: {self.stats.start_time}\n")
            f.write(f"结束时间: {self.stats.end_time}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("数据规模\n")
            f.write("-" * 80 + "\n")
            f.write(f"总记录数: {self.stats.total_records}\n")
            f.write(f"唯一 question 数: {self.stats.unique_questions}\n")
            f.write(f"最终节点数: {self.stats.final_node_count}\n")
            f.write(f"最终边数: {self.stats.final_edge_count}\n")
            f.write(f"平均每条记录产生的节点数: {self.stats.avg_nodes_per_record:.2f}\n")
            f.write(f"平均每条记录产生的边数: {self.stats.avg_edges_per_record:.2f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("时间统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总构建时间: {self.stats.total_build_time:.2f} 秒 ({self.stats.total_build_time/60:.2f} 分钟)\n")
            f.write(f"平均每条记录处理时间: {self.stats.avg_time_per_record:.2f} 秒\n")
            f.write(f"平均每个 question 处理时间: {self.stats.avg_time_per_question:.2f} 秒\n")
            f.write(f"预提取时间: {self.stats.pre_extract_time:.2f} 秒\n")
            f.write(f"批处理时间: {self.stats.batch_processing_time:.2f} 秒\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("LLM 调用统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总 LLM 调用次数: {self.stats.total_llm_calls}\n")
            f.write(f"总 LLM 调用时间: {self.stats.total_llm_time:.2f} 秒\n")
            f.write(f"平均每次 LLM 调用时间: {self.stats.avg_llm_time_per_call:.2f} 秒\n")
            f.write(f"LLM 超时次数: {self.stats.llm_timeout_count}\n")
            f.write(f"LLM 错误次数: {self.stats.llm_error_count}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("图谱操作统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总插入次数: {self.stats.total_graph_inserts}\n")
            f.write(f"总插入时间: {self.stats.total_graph_insert_time:.2f} 秒\n")
            f.write(f"平均每次插入时间: {self.stats.avg_graph_insert_time:.2f} 秒\n")
            f.write(f"聚类次数: {self.stats.clustering_count}\n")
            f.write(f"总聚类时间: {self.stats.total_clustering_time:.2f} 秒\n")
            f.write(f"平均每次聚类时间: {self.stats.avg_clustering_time:.2f} 秒\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("吞吐量统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"记录吞吐量: {self.stats.throughput_records_per_sec:.2f} 条/秒\n")
            if self.stats.throughput_tokens_per_sec > 0:
                f.write(f"Token 吞吐量: {self.stats.throughput_tokens_per_sec:.2f} token/秒\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("缓存统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question 缓存命中次数: {self.stats.question_cache_hits}\n")
            f.write(f"Question 缓存未命中次数: {self.stats.question_cache_misses}\n")
            f.write(f"Question 缓存命中率: {self.stats.question_cache_hit_rate*100:.2f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("资源使用统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"峰值内存使用: {self.stats.peak_memory_mb:.2f} MB\n")
            f.write(f"平均 CPU 使用率: {self.stats.avg_cpu_percent:.2f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("错误统计\n")
            f.write("-" * 80 + "\n")
            f.write(f"总错误数: {self.stats.total_errors}\n")
            f.write(f"跳过的记录数: {self.stats.skipped_records}\n")
            f.write(f"成功率: {self.stats.success_rate*100:.2f}%\n\n")
        
        print(f"[统计] 统计摘要已保存到: {output_path}")


def compare_stats(stats_files: List[str], output_file: str = "comparison.json"):
    """
    对比多个版本的统计信息
    
    Args:
        stats_files: 统计文件路径列表
        output_file: 输出对比文件路径
    """
    all_stats = []
    for stats_file in stats_files:
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                all_stats.append(stats)
    
    if not all_stats:
        print("没有找到有效的统计文件")
        return
    
    # 创建对比表
    comparison = {
        'versions': [s.get('version', 'unknown') for s in all_stats],
        'metrics': {}
    }
    
    # 对比关键指标
    key_metrics = [
        'total_build_time', 'avg_time_per_record', 'throughput_records_per_sec',
        'total_llm_calls', 'avg_llm_time_per_call', 'total_graph_inserts',
        'clustering_count', 'final_node_count', 'final_edge_count',
        'question_cache_hit_rate', 'success_rate'
    ]
    
    for metric in key_metrics:
        comparison['metrics'][metric] = [s.get(metric, 0) for s in all_stats]
    
    # 保存对比结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"对比结果已保存到: {output_file}")


