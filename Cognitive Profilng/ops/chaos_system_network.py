import itertools
import json
import subprocess
import time
import logging
import os
import csv
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict

# 目标IP地址列表
TARGET_IPS = [
    '***',
    '***', 
    '***',
    '***'
]

# 模型配置
MODEL_CONFIGS = {
    'Qwen2.5-3B-Instruct': {
        'path': '***',
        'config_file': 'NIPS2017attentionisallyouneedPaper.txt',
        'type': 'language_model'
    },
    'Qwen2.5-7B-Instruct': {
        'path': '***',
        'config_file': 'NIPS2017attentionisallyouneedPaper.txt',
        'type': 'language_model'
    },
    'Meta-Llama-3.1-8B-Instruct': {
        'path': '***',
        'config_file': 'NIPS2017attentionisallyouneedPaper.txt',
        'type': 'language_model'
    },
    'glm-4-9b-chat': {
        'path': '***',
        'config_file': 'NIPS2017attentionisallyouneedPaper.txt',
        'type': 'language_model'
    },
    'Qwen2-VL-2B-Instruct': {
        'path': '***',
        'config_file': '6G7M_20150417_063859_191.jpg',
        'type': 'vision_language_model'
    },
    'Qwen2-VL-7B-Instruct': {
        'path': '***',
        'config_file': '6G7M_20150417_063859_191.jpg',
        'type': 'vision_language_model'
    },
    'stable-diffusion': {
        'path': '***',
        'config_file': '6G7M_20150421_121002_835.jpg',
        'type': 'diffusion_model'
    },
    'CogVideo-1.0': {
        'path': '***',
        'config_file': 't2i.txt',
        'type': 'video_generation_model'
    }
}

# 优化的故障类型配置 - 网络故障合并，移除端口占用
FAULT_TYPES_CONFIG = {
    'cpu_stress': {
        'combinations': [(0, 0), (10, 50), (20, 80)],
        'description': 'CPU压力测试 (workers, load%)',
        'cmd_template': 'chaosd attack stress cpu --workers {} --load {}',
        'recover_cmd': 'chaosd recover {uid}',
        'target_specific': False
    },
    'memory_stress': {
        'combinations': [0, 25],
        'description': '内存压力测试 (GB)', 
        'cmd_template': 'chaosd attack stress mem --size {}GB',
        'recover_cmd': 'chaosd recover {uid}',
        'target_specific': False
    },
    'disk_read': {
        'combinations': [0, 200],
        'description': '磁盘读压力 (GB)',
        'cmd_template': 'chaosd attack disk add-payload read -s {}G -n 7 -p {model_config_path}',
        'recover_cmd': 'chaosd recover {uid}',
        'target_specific': False,
        'model_dependent': True
    },
    'disk_write': {
        'combinations': [0, 20],
        'description': '磁盘写压力 (GB)',
        'cmd_template': 'chaosd attack disk add-payload write -s {}G -n 8',
        'recover_cmd': 'chaosd recover {uid}',
        'target_specific': False
    },
    'file_fault': {
        'combinations': ['none', 'model_missing', 'content_error'],
        'description': '文件故障类型',
        'cmd_templates': {
            'none': '',
            'model_missing': 'chaosd attack file rename --source-file {model_config_path} --dest-file {new_path}',
            'content_error': 'chaosd attack file replace --origin-string "attention" --dest-string "none" --file-name {model_config_path}'
        },
        'recover_cmd': 'chaosd recover {uid}',
        'target_specific': False,
        'model_dependent': True
    },
    # 🔧 合并的网络故障类型 - 每次只能选择一种网络故障
    'network_fault': {
        'combinations': [
            'none',           # 无网络故障
            'packet_loss_70', # 网络丢包 50%
            'packet_error_70',# 网络错误 50%
            'packet_duplicate_70', # 网络重复 50%
            'delay_6000ms',   # 网络延迟 6000ms
            'bandwidth_5mbps' # 带宽限制 5mbps
        ],
        'description': '网络故障类型 (统一管理，避免冲突)',
        'cmd_templates': {
            'none': '',
            'packet_loss_70': 'chaosd attack network loss -d enp2s0 -i storage-host.local --percent 70',
            'packet_error_70': 'chaosd attack network corrupt -d enp2s0 -i storage-host.local --percent 70',
            'packet_duplicate_70': 'chaosd attack network duplicate -d enp2s0 -i storage-host.local --percent 70',
            'delay_6000ms': 'chaosd attack network delay -d enp2s0 -i storage-host.local -l 6000ms',
            'bandwidth_5mbps': 'chaosd attack network bandwidth --buffer 10000 --device enp2s0 --limit 10000 --rate 5mbps'
        },
        'recover_cmd': 'chaosd recover {uid}',
        'target_specific': False,
        'nfs_target': True
    }
}

class EnhancedChaosFaultGenerator:
    '''增强的故障组合生成器，支持多IP和多模型'''
    def __init__(self, fault_types_config: Dict, target_ips: List[str], model_configs: Dict):
        self.fault_types = fault_types_config
        self.fault_names = list(fault_types_config.keys())
        self.target_ips = target_ips
        self.model_configs = model_configs

    def get_model_config_path(self, model_name: str) -> str:
        '''获取模型配置文件完整路径'''
        model_config = self.model_configs[model_name]
        return f"{model_config['path']}/{model_config['config_file']}"

    def calculate_total_combinations(self) -> Tuple[int, Dict]:
        '''计算总组合数和详细信息'''
        total = 1
        details = {}
        for fault_type, config in self.fault_types.items():
            count = len(config['combinations'])
            total *= count
            details[fault_type] = {
                'count': count,
                'combinations': config['combinations'],
                'description': config['description']
            }
        return total, details

    def generate_fault_command(self, fault_type: str, fault_value: Any, 
                              target_ip: str = None, model_name: str = None) -> Optional[str]:
        '''根据故障类型、值、目标IP和模型生成命令'''
        config = self.fault_types[fault_type]

        model_config_path = None
        if config.get('model_dependent') and model_name:
            model_config_path = self.get_model_config_path(model_name)

        if fault_type == 'network_fault':
            if fault_value == 'none':
                return None
            cmd_template = config['cmd_templates'][fault_value]
            return cmd_template

        elif fault_type == 'file_fault':
            if fault_value == 'none':
                return None
            cmd_template = config['cmd_templates'][fault_value]
            if model_config_path and '{model_config_path}' in cmd_template:
                dir_path = os.path.dirname(model_config_path)
                filename = os.path.basename(model_config_path)

                name, ext = os.path.splitext(filename)

                new_filename = f"{name}3{ext}"

                new_path = f"{dir_path}/{new_filename}"

                cmd_template = cmd_template.replace('{model_config_path}', model_config_path)
                cmd_template = cmd_template.replace('{new_path}', new_path)
            return cmd_template

        # 处理CPU压力
        elif fault_type == 'cpu_stress':
            workers, load = fault_value
            if workers == 0 or load == 0:
                return None
            return config['cmd_template'].format(workers, load)

        # 处理磁盘读取
        elif fault_type == 'disk_read':
            if fault_value == 0:
                return None
            cmd_template = config['cmd_template']
            if model_config_path and '{model_config_path}' in cmd_template:
                cmd_template = cmd_template.replace('{model_config_path}', model_config_path)
            return cmd_template.format(fault_value)

        # 处理其他故障类型
        else:
            if fault_value == 0:
                return None
            return config['cmd_template'].format(fault_value)

    def generate_all_scenarios(self, limit: Optional[int] = None) -> List[Dict]:
        '''生成所有测试场景（包括IP和模型组合）'''
        fault_combinations = [self.fault_types[name]['combinations'] for name in self.fault_names]

        all_scenarios = []
        scenario_id = 1

        for target_ip in self.target_ips:
            for model_name in self.model_configs.keys():
                # 使用itertools.product生成所有故障组合
                for combo in itertools.product(*fault_combinations):
                    if limit and scenario_id > limit:
                        return all_scenarios

                    fault_state = {}
                    commands = []

                    for i, fault_name in enumerate(self.fault_names):
                        fault_value = combo[i]
                        fault_state[fault_name] = fault_value

                        cmd = self.generate_fault_command(fault_name, fault_value, target_ip, model_name)
                        if cmd:
                            if fault_name == 'network_fault':
                                description = f"网络故障: {fault_value}"
                            else:
                                description = self.fault_types[fault_name]['description']

                            commands.append({
                                'type': fault_name,
                                'value': fault_value,
                                'command': cmd,
                                'description': description
                            })

                    if commands:
                        all_scenarios.append({
                            'scenario_id': scenario_id,
                            'target_ip': target_ip,
                            'model_name': model_name,
                            'model_type': self.model_configs[model_name]['type'],
                            'fault_state': fault_state,
                            'active_faults': commands,
                            'total_faults': len(commands),
                            'fault_types_summary': [cmd['type'] for cmd in commands]
                        })
                        scenario_id += 1

        return all_scenarios

class EnhancedChaosExperimentManager:
    def __init__(self, log_dir: str = "/tmp/chaos_logs"):
        self.log_dir = log_dir
        self.setup_logging()
        self.generator = EnhancedChaosFaultGenerator(
            FAULT_TYPES_CONFIG, TARGET_IPS, MODEL_CONFIGS
        )

    def setup_logging(self):
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/chaos_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def export_all_scenarios_to_csv(self, output_file: str = "chaos_scenarios_network_optimized.csv") -> str:
        '''导出所有测试场景到CSV文件'''
        scenarios = self.generator.generate_all_scenarios()

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'scenario_id', 'execution_target', 'nfs_target', 'model_name', 'model_type', 
                'total_faults', 'fault_types', 'commands_detail'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for scenario in scenarios:
                fault_types = '; '.join(scenario['fault_types_summary'])

                commands_detail = []
                for cmd in scenario['active_faults']:
                    commands_detail.append(f"{cmd['type']}: {cmd['command']}")
                commands_str = ' | '.join(commands_detail)

                writer.writerow({
                    'scenario_id': scenario['scenario_id'],
                    'execution_target': scenario['target_ip'],
                    'nfs_target': 'storage-host.local',
                    'model_name': scenario['model_name'],
                    'model_type': scenario['model_type'],
                    'total_faults': scenario['total_faults'],
                    'fault_types': fault_types,
                    'commands_detail': commands_str
                })

        self.logger.info(f"已导出 {len(scenarios)} 个测试场景到 {output_file}")
        return output_file

def main():
    manager = EnhancedChaosExperimentManager()

    total, details = manager.generator.calculate_total_combinations()

    network_config = details['network_fault']

    # 计算总场景数
    total_scenarios = total * len(TARGET_IPS) * len(MODEL_CONFIGS)

    # 导出CSV
    csv_file = manager.export_all_scenarios_to_csv()
    print(f"\n🎯 已生成CSV: {csv_file}")

    return manager

if __name__ == "__main__":
    manager = main()
