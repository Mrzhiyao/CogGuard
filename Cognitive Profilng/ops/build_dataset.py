import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

BASE_DIR = Path('./datasets/ops/chaosmesh_results_new')
CODE_DIR = Path('./datasets/ops')

# 准确的硬件配置（来自用户提供）
HARDWARE_CONFIG = {
    "node_2070s": {
        "node_id": "node_2070s",
        "type": "Consumer-grade Desktop",
        "cpu": {
            "cores": 10,
            "threads": 20,
            "clock_speed": "5.3GHz"
        },
        "gpu": {
            "model": "RTX 2070 Super",
            "vram_mib": 8192,
            "compute_tops_int8": 52
        },
        "memory_gib": 31,
        "bandwidth_mbps": 1000
    },
    "node_3090": {
        "node_id": "node_3090",
        "type": "Workstation",
        "cpu": {
            "cores": 10,
            "threads": 20,
            "clock_speed": "2.2GHz"
        },
        "gpu": {
            "model": "RTX 3090",
            "vram_mib": 24576,
            "compute_tops_int8": 284
        },
        "memory_gib": 31,
        "bandwidth_mbps": 1000
    },
    "node_4090d": {
        "node_id": "node_4090d",
        "type": "Consumer-grade Desktop",
        "cpu": {
            "cores": 24,
            "threads": 32,
            "clock_speed": "3.2GHz"
        },
        "gpu": {
            "model": "RTX 4090 D",
            "vram_mib": 24564,
            "compute_tops_int8": 588
        },
        "memory_gib": 62,
        "bandwidth_mbps": 2500
    }
}

# 主机名到节点ID的映射
HOST_TO_NODE = {
    '2070s-1': 'node_2070s',
    '3090-1': 'node_3090',
    '3090-2': 'node_3090',
    '4090d-1': 'node_4090d'
}

# 读取代码文件
def read_code_file(filename: str) -> str:
    """读取代码文件内容"""
    file_path = CODE_DIR / filename
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# 无法读取代码文件 {filename}: {e}"

T2T_READ_CODE = '''def read_task_file(self) -> Tuple[Optional[str], float]:
    """读取任务文件 - 20秒超时，失败返回None"""
    config = self.file_config
    cmd = f'ssh -o ConnectTimeout=5 {config["user"]}@{config["host"]} "cat \\"{config["file_path"]}\\""'
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True, 
        timeout=20
    )
    
    if result.returncode == 0 and result.stdout and result.stdout.strip():
        content = result.stdout.strip()
        return content, read_time
    return None, read_time'''

T2T_EXECUTE_CODE = read_code_file('test_t2t.py')

I2T_READ_CODE = '''def read_task_file(self) -> Tuple[Optional[str], float]:
    """读取任务文件 - 下载JPG到本地，20秒超时，失败返回None"""
    config = self.file_config
    cmd = f'scp -o ConnectTimeout=5 {config["user"]}@{config["host"]}:"{config["file_path"]}" {self.local_path}'
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True, 
        timeout=20
    )
    
    if result.returncode == 0 and os.path.exists(self.local_path):
        return self.local_path, read_time
    return None, read_time'''

I2T_EXECUTE_CODE = read_code_file('test_i2t.py')


def parse_fault_commands(commands_detail: str) -> List[Dict]:
    faults = []
    for cmd in commands_detail.split('|'):
        cmd = cmd.strip()
        if not cmd:
            continue
        
        if ':' in cmd:
            fault_type_prefix = cmd.split(':', 1)[0].strip()
        else:
            fault_type_prefix = cmd
        
        fault_info = {
            'type': '',
            'params': {}
        }
        
        if 'network' in cmd:
            if 'loss' in cmd:
                fault_info['type'] = 'network_loss'
                match = re.search(r'--percent\s+(\d+)', cmd)
                if match:
                    fault_info['params']['percent'] = int(match.group(1))
            elif 'corrupt' in cmd:
                fault_info['type'] = 'network_corrupt'
                match = re.search(r'--percent\s+(\d+)', cmd)
                if match:
                    fault_info['params']['percent'] = int(match.group(1))
            elif 'duplicate' in cmd:
                fault_info['type'] = 'network_duplicate'
                match = re.search(r'--percent\s+(\d+)', cmd)
                if match:
                    fault_info['params']['percent'] = int(match.group(1))
            elif 'delay' in cmd:
                fault_info['type'] = 'network_delay'
                match = re.search(r'-l\s+(\d+)ms', cmd)
                if match:
                    fault_info['params']['delay_ms'] = int(match.group(1))
            elif 'bandwidth' in cmd:
                fault_info['type'] = 'network_bandwidth'
                match = re.search(r'--rate\s+(\d+)mbps', cmd)
                if match:
                    fault_info['params']['rate_mbps'] = int(match.group(1))
        elif 'file' in cmd:
            if 'rename' in cmd:
                fault_info['type'] = 'file_rename'
            elif 'replace' in cmd:
                fault_info['type'] = 'file_replace'
            elif 'modify' in cmd:
                fault_info['type'] = 'file_modify'
        elif 'disk' in cmd:
            if 'read' in cmd:
                fault_info['type'] = 'disk_read'
            elif 'write' in cmd:
                fault_info['type'] = 'disk_write'
        elif 'stress' in cmd:
            if 'mem' in cmd:
                fault_info['type'] = 'memory_stress'
                match = re.search(r'--size\s+(\d+)GB', cmd)
                if match:
                    fault_info['params']['size_gb'] = int(match.group(1))
            elif 'cpu' in cmd:
                fault_info['type'] = 'cpu_stress'
        
        match = re.search(r'-d\s+(\S+)', cmd)
        if match:
            fault_info['params']['device'] = match.group(1)
        
        match = re.search(r'-i\s+(\S+)', cmd)
        if match:
            fault_info['params']['target_ip'] = match.group(1)
        
        faults.append(fault_info)
    
    return faults


def build_hardware_profile(host: str) -> str:
    node_id = HOST_TO_NODE.get(host)
    if not node_id or node_id not in HARDWARE_CONFIG:
        return "Hardware configuration not available"
    
    hw = HARDWARE_CONFIG[node_id]
    parts = []
    
    parts.append("## Hardware Profile Characteristic")
    parts.append(f"### Node ID: {hw['node_id']}")
    parts.append("")
    parts.append("### Type:")
    parts.append(f"- {hw.get('type', 'Unknown')}")
    parts.append("")
    parts.append("### CPU:")
    parts.append(f"- Cores: {hw['cpu']['cores']}")
    parts.append(f"- Threads: {hw['cpu']['threads']}")
    parts.append(f"- Clock Speed: {hw['cpu']['clock_speed']}")
    parts.append("")
    parts.append("### GPU:")
    parts.append(f"- Model: {hw['gpu']['model']}")
    parts.append(f"- VRAM: {hw['gpu']['vram_mib']} MiB")
    parts.append(f"- Compute (INT8): {hw['gpu']['compute_tops_int8']} TOPS")
    parts.append("")
    parts.append("### Memory:")
    parts.append(f"- Size: {hw['memory_gib']} GiB")
    parts.append("")
    parts.append("### Network:")
    parts.append(f"- Bandwidth: {hw['bandwidth_mbps']} Mbps")
    
    return "\n".join(parts)


def build_runtime_profile(faults: List[Dict], task_row: Dict) -> str:
    """构建运行时配置（故障组合）"""
    parts = []
    
    parts.append("## Runtime Profile Characteristic")
    parts.append("### Fault Injection Pattern:")
    parts.append("")
    
    for i, fault in enumerate(faults, 1):
        fault_desc = f"{i}. {fault['type']}"
        if fault['params']:
            params_str = ", ".join([f"{k}={v}" for k, v in fault['params'].items()])
            fault_desc += f" ({params_str})"
        parts.append(fault_desc)
    
    parts.append("")
    parts.append("### Fault Combination Analysis:")
    parts.append(f"- Total Faults: {len(faults)}")
    fault_types = [f['type'] for f in faults]
    parts.append(f"- Fault Types: {', '.join(fault_types)}")
    parts.append(f"- Injection Pattern: {'Simultaneous' if len(faults) > 1 else 'Single'}")
    
    return "\n".join(parts)


def build_node_knowledge_graph(host: str, all_results: List[Dict], task_type: str) -> str:
    """构建节点知识图（只包含硬件配置）"""
    hardware_profile = build_hardware_profile(host)
    return hardware_profile


def build_task_description(task_row: Dict, task_type: str) -> str:
    """构建任务描述（从t2t.py/i2t.py提取）"""
    if task_type == 't2t':
        parts = [
            "Task: Text Summarization (文本总结任务)",
            "",
            "Step 1: Get Input Data (获取输入数据)",
            "- Source: Read paper file from NFS server",
            "- File Path: /nfs/ai/ai-model/chaos_mesh/NIPS2017attentionisallyouneedPaper.txt",
            "- Method: SSH command to read file content",
            "",
            "Code for reading data:",
            "```python",
            T2T_READ_CODE,
            "```",
            "",
            "Step 2: Execute Inference Task (执行推理任务)",
            "- Task Description: 请总结以下内容",
            "- Input: Paper text content read from NFS server",
            "- Model: qwen15-7b",
            "",
            "Code for inference:",
            "```python",
            T2T_EXECUTE_CODE,
            "```"
        ]
    else:  
        parts = [
            "Task: Image Description (图像描述任务)",
            "",
            "Step 1: Get Input Data (获取输入数据)",
            "- Source: Download image file from NFS server",
            "- File Path: /nfs/ai/ai-model/chaos_mesh/i2t.jpg",
            "- Method: SCP command to download to local",
            "- Local Path: ./chaos-mesh/i2t.jpg",
            "",
            "Code for downloading data:",
            "```python",
            I2T_READ_CODE,
            "```",
            "",
            "Step 2: Execute Inference Task (执行推理任务)",
            "- Task Description: Summarize the content of the figure",
            "- Input: Image URL (http://***:12365/data_chaos_mesh/i2t.jpg)",
            "- Model: qwen2-vl-2b",
            "",
            "Code for inference:",
            "```python",
            I2T_EXECUTE_CODE,
            "```"
        ]
    
    return "\n".join(parts)


def extract_log_summary(log_file: Path, scenario_id: int) -> str:
    """从日志文件中提取特定scenario的执行摘要（从实际日志文件）"""
    summary_lines = []
    in_scenario = False
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if f'开始实验场景 {scenario_id}' in line:
                    in_scenario = True
                    summary_lines.append(f"开始场景 {scenario_id}")
                
                if in_scenario:
                    if '故障注入成功' in line:
                        summary_lines.append(f"  - {line.strip()}")
                    elif '文件读取' in line and '成功' in line:
                        summary_lines.append(f"  - {line.strip()}")
                    elif '文件下载' in line and '成功' in line:
                        summary_lines.append(f"  - {line.strip()}")
                    elif '推理任务完成' in line:
                        summary_lines.append(f"  - {line.strip()}")
                    elif '实验成功完成' in line or '实验失败' in line:
                        summary_lines.append(f"  - {line.strip()}")
                        break
                    elif 'ERROR' in line or '失败' in line:
                        summary_lines.append(f"  - {line.strip()}")
    except Exception as e:
        return f"日志读取错误: {e}"
    
    return "\n".join(summary_lines) if summary_lines else "无日志记录"


def calculate_score(result_row: Dict) -> float:
    error_msg = result_row.get('error_msg', '').strip()
    success = result_row.get('success', 'False') == 'True'
    
    if error_msg:
        base_score = 0.0
        if '文件读取失败' in error_msg or '文件下载失败' in error_msg:
            base_score = 15.0
        elif '超时' in error_msg:
            base_score = 40.0
        elif '连接' in error_msg or 'Connection' in error_msg:
            base_score = 30.0
        else:
            # 其他错误，给5-20分
            base_score = 10.0
        
        try:
            total_time = float(result_row.get('total_time', 0))
            if total_time > 20:
                # 尝试时间较长，说明系统有一定响应，给加分
                base_score += min((total_time - 20) / 100 * 10, 10)
        except:
            pass
        
        return max(0.0, min(100.0, base_score))
    
    if success:
        base_score = 100.0
        
        try:
            total_time = float(result_row.get('total_time', 0))
            file_read_time = float(result_row.get('file_read_time', 0))
            task_time = float(result_row.get('task_time', 0))
            

            if total_time > 100:
                time_penalty = 40 + min((total_time - 100) / 100 * 20, 20)
                base_score -= time_penalty
            elif total_time > 50:
                time_penalty = 30 + (total_time - 50) / 50 * 10
                base_score -= time_penalty
            elif total_time > 30:
                time_penalty = 20 + (total_time - 30) / 20 * 10
                base_score -= time_penalty
            elif total_time > 20:
                time_penalty = (total_time - 20) / 10 * 5
                base_score -= time_penalty
            
            # 文件读取时间过长
            if file_read_time > 10:
                file_penalty = min((file_read_time - 10) / 10 * 5, 5)
                base_score -= file_penalty
            
            # 任务执行时间过长
            if task_time > 10:
                task_penalty = min((task_time - 10) / 10 * 5, 5)
                base_score -= task_penalty
        except:
            pass
        
        # 故障恢复惩罚
        try:
            fault_count = int(result_row.get('fault_count', 0))
            recovery_count = int(result_row.get('recovery_count', 0))
            if fault_count > 0 and recovery_count < fault_count:
                # 每个未恢复的故障扣15分
                base_score -= (fault_count - recovery_count) * 15
        except:
            pass
        
        return max(0.0, min(100.0, base_score))
    
    return 0.0


def build_dataset_for_host(host: str, task_type: str):
    host_dir = BASE_DIR / host
    
    if task_type == 't2t':
        task_file = host_dir / 'tasks' / 'Qwen2_5-3B-Instruct.csv'
        if (host_dir / 'results_t2t_only.csv').exists():
            result_file = host_dir / 'results_t2t_only.csv'
        elif (host_dir / 'results_t2t_merged.csv').exists():
            result_file = host_dir / 'results_t2t_merged.csv'
        else:
            result_file = host_dir / 'results_t2t.csv'
        
        if (host_dir / 'logs' / 'chaos_executor_t2t_only.log').exists():
            log_file = host_dir / 'logs' / 'chaos_executor_t2t_only.log'
        elif (host_dir / 'logs' / 'chaos_executor_t2t_merged.log').exists():
            log_file = host_dir / 'logs' / 'chaos_executor_t2t_merged.log'
        else:
            log_file = host_dir / 'logs' / 'chaos_executor_t2t.log'
    else: 
        task_file = host_dir / 'tasks' / 'Qwen2-VL-2B-Instruct.csv'
        result_file = host_dir / 'results_i2t.csv'
        log_file = host_dir / 'logs' / 'chaos_executor_i2t.log'
    
    if not task_file.exists() or not result_file.exists():
        print(f"⚠️  文件不存在: {host}/{task_type}")
        return []
    
    tasks = []
    seen_scenario_ids = set()  # 用于去重
    with open(task_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario_id = int(row['scenario_id'])
            if scenario_id not in seen_scenario_ids:
                seen_scenario_ids.add(scenario_id)
                tasks.append(row)
    
    all_results = []
    results = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_results.append(row)
            scenario_id = int(row['scenario_id'])
            results[scenario_id] = row
    
    node_knowledge_graph = build_node_knowledge_graph(host, all_results, task_type)
    
    training_data = []
    
    for task in tasks:
        scenario_id = int(task['scenario_id'])
        
        # 跳过没有结果的任务
        if scenario_id not in results:
            continue
        
        result = results[scenario_id]
        
        # 解析故障信息（从tasks CSV的commands_detail）
        faults = parse_fault_commands(task['commands_detail'])
        
        # 构建运行时配置（故障组合）
        runtime_profile = build_runtime_profile(faults, task)
        
        # 合并知识图
        combined_graph = f"{node_knowledge_graph}\n\n{runtime_profile}"
        
        # 构建任务描述（从t2t.py/i2t.py提取）
        task_description = build_task_description(task, task_type)
        
        # 提取日志摘要（从实际日志文件）
        log_summary = ""
        if log_file.exists():
            log_summary = extract_log_summary(log_file, scenario_id)
        
        # 计算得分（基于results CSV的total_time, success, error_msg）
        score = calculate_score(result)
        
        # 构建prompt（只包含任务特征，不包含知识图和执行日志）
        prompt = f"""You are a CogGuard system expert. Based on the task characteristics, hardware profile, and runtime profile (fault injection pattern), predict the task execution performance score.

Task Characteristics:
{task_description}

Predict the performance score (0-100) for this task execution."""
        
        # 分类标签
        if score >= 80:
            tgt = "High"
        elif score >= 60:
            tgt = "Medium"
        elif score >= 40:
            tgt = "Low"
        else:
            tgt = "Very Low"
        
        # 所有数据都来自实际文件
        sample = {
            "src": prompt,
            "e": combined_graph,
            "tgt": tgt,
            "label": score,
            "scenario_id": scenario_id,
            "host": host,
            "task_type": task_type,
            "fault_types": task['fault_types'],  # 来自tasks CSV
            "total_faults": int(task['total_faults']),  # 来自tasks CSV
            "success": result.get('success', 'False'),  # 来自results CSV
            "total_time": float(result.get('total_time', 0)),  # 来自results CSV
            "file_read_time": float(result.get('file_read_time', 0)),  # 来自results CSV
            "task_time": float(result.get('task_time', 0)),  # 来自results CSV
            "fault_count": int(result.get('fault_count', 0)),  # 来自results CSV
            "recovery_count": int(result.get('recovery_count', 0)),  # 来自results CSV
            "error_msg": result.get('error_msg', ''),  # 来自results CSV
        }
        
        training_data.append(sample)
    
    return training_data


def main():

    hosts = ['2070s-1', '3090-2', '4090d-1']
    all_data = []
    
    for host in hosts:

        t2t_data = build_dataset_for_host(host, 't2t')
        if t2t_data:
            all_data.extend(t2t_data)
            print(f"    ✅ 生成 {len(t2t_data)} 条T2T数据")
        
        i2t_data = build_dataset_for_host(host, 'i2t')
        if i2t_data:
            all_data.extend(i2t_data)
            print(f"    ✅ 生成 {len(i2t_data)} 条I2T数据")
    
    if all_data:
        output_file = BASE_DIR.parent / 'CogGuard_dataset_corrected.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n✅ 已生成数据集: {output_file}")
        print(f"   总样本数: {len(all_data)}")
        
        # 统计信息
        scores = [s['label'] for s in all_data]
        print(f"   得分范围: {min(scores):.2f} - {max(scores):.2f}")
        print(f"   平均得分: {sum(scores)/len(scores):.2f}")
        
        success_count = sum(1 for s in all_data if s['success'] == 'True')
        print(f"   成功任务: {success_count}/{len(all_data)} ({success_count/len(all_data)*100:.1f}%)")
        
        print("  ✅ src: 来自任务描述（t2t.py/i2t.py）和代码文件")
        print("  ✅ e: 来自硬件配置（用户提供）+ 故障模式（tasks CSV）+ 历史统计（results CSV）")
        print("  ✅ tgt: 根据label计算")
        print("  ✅ label: 根据results CSV的total_time, success, error_msg计算")
        print("  ✅ 其他字段: 全部来自results CSV和tasks CSV")


if __name__ == '__main__':
    main()

