import csv
import os
from collections import defaultdict
from typing import Dict, List
from datetime import datetime


class EnhancedChaosScenariosCSVSplitter:

    def __init__(self):
        self.target_ips = [
            '***',  
            '***', 
            '***',
            '***'
        ]

        self.model_names = [
            'Qwen2.5-3B-Instruct',
            'Qwen2.5-7B-Instruct',
            'Meta-Llama-3.1-8B-Instruct',
            'glm-4-9b-chat',
            'Qwen2-VL-2B-Instruct',
            'Qwen2-VL-7B-Instruct',
            'stable-diffusion',
            'CogVideo-1.0'
        ]

    def diagnose_csv(self, input_csv_file: str) -> Dict:
        diagnosis = {
            'total_rows': 0,
            'unique_ips': set(),
            'unique_models': set(),
            'ip_field_name': None,
            'model_field_name': None,
            'all_fields': [],
            'matched_rows': 0,
            'unmatched_rows': 0,
            'unmatched_samples': [],
            'ip_mismatches': defaultdict(int),
            'model_mismatches': defaultdict(int)
        }

        try:
            with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                diagnosis['all_fields'] = reader.fieldnames

                # 自动检测IP字段名
                ip_candidates = ['execution_target', 'target_ip', 'ip', 'host_ip', 'server_ip']
                for field in ip_candidates:
                    if field in reader.fieldnames:
                        diagnosis['ip_field_name'] = field
                        break

                # 自动检测模型字段名
                model_candidates = ['model_name', 'model', 'model_id']
                for field in model_candidates:
                    if field in reader.fieldnames:
                        diagnosis['model_field_name'] = field
                        break

                for row in reader:
                    diagnosis['total_rows'] += 1

                    # 获取IP
                    target_ip = ''
                    if diagnosis['ip_field_name']:
                        target_ip = row.get(diagnosis['ip_field_name'], '').strip()
                    
                    # 获取模型名
                    model_name = ''
                    if diagnosis['model_field_name']:
                        model_name = row.get(diagnosis['model_field_name'], '').strip()

                    diagnosis['unique_ips'].add(target_ip)
                    diagnosis['unique_models'].add(model_name)

                    # 检查是否匹配
                    ip_matched = target_ip in self.target_ips
                    model_matched = model_name in self.model_names

                    if ip_matched and model_matched and model_name:
                        diagnosis['matched_rows'] += 1
                    else:
                        diagnosis['unmatched_rows'] += 1
                        
                        # 记录不匹配的原因
                        if not ip_matched and target_ip:
                            diagnosis['ip_mismatches'][target_ip] += 1
                        if not model_matched and model_name:
                            diagnosis['model_mismatches'][model_name] += 1

                        # 保存前10个不匹配的样本
                        if len(diagnosis['unmatched_samples']) < 10:
                            diagnosis['unmatched_samples'].append({
                                'row_num': diagnosis['total_rows'],
                                'ip': target_ip,
                                'model': model_name,
                                'ip_matched': ip_matched,
                                'model_matched': model_matched
                            })

        except Exception as e:
            diagnosis['error'] = str(e)

        return diagnosis

    def print_diagnosis(self, diagnosis: Dict):


        if 'error' in diagnosis:
            print(f" 错误: {diagnosis['error']}")
            return

        for ip in sorted(diagnosis['unique_ips']):
            status = "✓" if ip in self.target_ips else "✗"

        for model in sorted(diagnosis['unique_models']):
            status = "✓" if model in self.model_names else "✗"

        if diagnosis['ip_mismatches']:
            for ip, count in sorted(diagnosis['ip_mismatches'].items(), key=lambda x: -x)[:10]:
                print(f"  • {ip}: {count} 次")

        if diagnosis['model_mismatches']:
            for model, count in sorted(diagnosis['model_mismatches'].items(), key=lambda x: -x)[:10]:
                print(f"  • {model}: {count} 次")

        if diagnosis['unmatched_samples']:
            print(f"\n🔍 未匹配样本 (前10个):")
            for sample in diagnosis['unmatched_samples']:
                print(f"  行 {sample['row_num']}: IP={sample['ip']} (匹配:{sample['ip_matched']}), "
                      f"模型={sample['model']} (匹配:{sample['model_matched']})")

    def split_csv_by_ip_and_model(self, input_csv_file: str, 
                                  output_base_dir: str = "./csv_hierarchical_1026",
                                  strict_mode: bool = True) -> Dict:

        os.makedirs(output_base_dir, exist_ok=True)

        hierarchical_data = defaultdict(lambda: defaultdict(list))
        total_records = 0
        fieldnames = None
        
        discovered_ips = set()
        discovered_models = set()

        try:
            with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                fieldnames = reader.fieldnames

                # 自动检测字段名
                ip_field = None
                model_field = None
                
                for candidate in ['execution_target', 'target_ip', 'ip', 'host_ip']:
                    if candidate in fieldnames:
                        ip_field = candidate
                        break
                
                for candidate in ['model_name', 'model', 'model_id']:
                    if candidate in fieldnames:
                        model_field = candidate
                        break

                if not ip_field or not model_field:
                    return {"error": f"未找到IP字段或模型字段。可用字段: {fieldnames}"}


                for row in reader:
                    target_ip = row.get(ip_field, '').strip()
                    model_name = row.get(model_field, '').strip()

                    if not strict_mode:
                        discovered_ips.add(target_ip)
                        discovered_models.add(model_name)

                    should_include = False
                    if strict_mode:
                        should_include = (target_ip in self.target_ips and 
                                        model_name in self.model_names and 
                                        model_name)
                    else:
                        should_include = target_ip and model_name

                    if should_include:
                        hierarchical_data[target_ip][model_name].append(row)
                        total_records += 1

        except FileNotFoundError:
            return {"error": f"文件 {input_csv_file} 不存在"}
        except Exception as e:
            return {"error": f"读取CSV文件时出错: {str(e)}"}

        ips_to_process = self.target_ips if strict_mode else sorted(discovered_ips)
        models_to_process = self.model_names if strict_mode else sorted(discovered_models)

        results = {
            'created_structure': {},
            'statistics': {},
            'total_records': total_records,
            'output_base_directory': output_base_dir,
            'created_files': [],
            'mode': 'strict' if strict_mode else 'flexible'
        }

        for ip in ips_to_process:
            ip_data = hierarchical_data.get(ip, {})

            if ip_data:
                safe_ip = ip.replace('.', '_').replace(':', '_')
                ip_folder = f"{output_base_dir}/{safe_ip}"
                os.makedirs(ip_folder, exist_ok=True)

                results['created_structure'][ip] = {
                    'folder_path': ip_folder,
                    'model_files': {},
                    'total_scenarios': 0
                }

                results['statistics'][ip] = {}

                for model_name in models_to_process:
                    model_records = ip_data.get(model_name, [])

                    if model_records:
                        safe_model_name = self.get_safe_model_name(model_name)
                        model_csv_file = f"{ip_folder}/{safe_model_name}.csv"

                        with open(model_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(model_records)

                        record_count = len(model_records)
                        results['created_structure'][ip]['model_files'][model_name] = {
                            'file_path': model_csv_file,
                            'record_count': record_count
                        }

                        results['created_structure'][ip]['total_scenarios'] += record_count
                        results['statistics'][ip][model_name] = record_count
                        results['created_files'].append({
                            'ip': ip,
                            'model': model_name,
                            'file_path': model_csv_file,
                            'record_count': record_count
                        })

                        print(f"  📄 {safe_model_name}.csv ({record_count} 条记录)")

        return results

    def get_safe_model_name(self, model_name: str) -> str:
        safe_name = model_name.replace('.', '_').replace('/', '_').replace('\\', '_')
        safe_name = safe_name.replace(' ', '_').replace(':', '_')
        return safe_name


def main():

    splitter = EnhancedChaosScenariosCSVSplitter()

    input_file = "chaos_scenarios_network.csv"
    
    if not os.path.exists(input_file):
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            for i, f in enumerate(csv_files, 1):
                print(f"  {i}. {f}")
        return

    diagnosis = splitter.diagnose_csv(input_file)
    splitter.print_diagnosis(diagnosis)


    choice = input("\n请选择 (1/2/3) [默认=1]: ").strip() or "1"

    if choice == "3":
        print("退出程序")
        return

    strict_mode = (choice == "1")
    mode_name = "严格模式" if strict_mode else "灵活模式"

    results = splitter.split_csv_by_ip_and_model(
        input_file, 
        strict_mode=strict_mode
    )

    if 'error' in results:
        return

    print(f"\n拆分完成!")
    print(f"输出目录: {results['output_base_directory']}")
    print(f"总记录数: {results['total_records']:,}")
    print(f"创建文件数: {len(results['created_files'])}")

    summary_file = splitter.create_hierarchical_summary(results)
    script_results = splitter.create_deployment_scripts(results)

    return results


if __name__ == "__main__":
    results = main()
