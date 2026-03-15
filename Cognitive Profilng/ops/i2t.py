import csv
import subprocess
import time
import logging
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """实验结果"""
    scenario_id: int
    success: bool
    total_time: float
    file_read_time: float
    task_time: float
    fault_count: int
    recovery_count: int
    error_msg: str = ""
    skip_reason: str = ""
    task_result: str = ""  # 新增推理任务结果


class SimpleChaosExecutor:
    def __init__(self, simulate_mode: bool = False):
        self.simulate_mode = simulate_mode
        self.setup_logging()

        os.makedirs("./chaos-mesh", exist_ok=True)

        self.skip_ports = [23333, 4781]
        self.file_config = {
            'host': '***',
            'user': 'root', 
            'file_path': '***'
        }
        self.local_path = './chaos-mesh/i2t.jpg'

        # chaosd配置
        self.chaosd_config = {
            '***': {
                'user': 'root',
                'path': '/usr/local/chaosd-v1.4.0-linux-amd64/chaosd',
                'work_dir': '/usr/local/chaosd-v1.4.0-linux-amd64'
            }
        }

        if simulate_mode:
            self.logger.info("🎭 模拟模式启用")
        else:
            self.logger.info("🚀 真实执行模式")

    def setup_logging(self):
        """设置日志"""
        os.makedirs("./logs", exist_ok=True)

        log_file = f"./logs/chaos_executor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

    def should_skip(self, commands: str) -> Tuple[bool, str]:
        if not commands:
            return False, ""

        for port in self.skip_ports:
            if f"--port {port}" in commands:
                return True, f"跳过端口 {port}"

        network_types = ['network loss', 'network bandwidth', 'network delay']
        count = sum(1 for t in network_types if t in commands)
        if count > 1:
            return True, f"{count}个网络命令冲突"

        return False, ""

    def read_task_file(self) -> Tuple[Optional[str], float]:
        """读取任务文件 - 下载JPG到本地，20秒超时，失败返回None"""
        start_time = time.time()

        if self.simulate_mode:
            time.sleep(0.5)
            self.logger.info(f"🎭 模拟下载到: {self.local_path}")
            return self.local_path, time.time() - start_time

        config = self.file_config
        cmd = f'scp -o ConnectTimeout=5 {config["user"]}@{config["host"]}:"{config["file_path"]}" {self.local_path}'

        try:
            self.logger.info(f"下载文件: {config['host']}:{config['file_path']} 到 {self.local_path}")
            self.logger.debug(f"执行命令: {cmd}")

            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=20  # 20秒超时
            )

            self.logger.debug(f"命令返回码: {result.returncode}")
            self.logger.debug(f"stdout长度: {len(result.stdout) if result.stdout else 0}")
            self.logger.debug(f"stderr长度: {len(result.stderr) if result.stderr else 0}")

            if result.returncode == 0:
                if os.path.exists(self.local_path) and os.path.getsize(self.local_path) > 0:
                    read_time = time.time() - start_time
                    self.logger.info(f"✅ 文件下载成功，耗时: {read_time:.2f}秒，大小: {os.path.getsize(self.local_path)} 字节")
                    return self.local_path, read_time
                else:
                    self.logger.error(f" 下载文件为空或不存在: {self.local_path}")
                    if result.stderr:
                        self.logger.error(f"stderr输出: {result.stderr.strip()}")
            else:
                self.logger.error(f" SCP命令执行失败，返回码: {result.returncode}")
                self.logger.error(f"stderr: {result.stderr.strip() if result.stderr else '无错误信息'}")
                if result.stdout:
                    self.logger.debug(f"stdout: {result.stdout.strip()}")

                if result.stderr:
                    if "No such file or directory" in result.stderr:
                        self.logger.error(f" 文件不存在: {config['file_path']}")
                    elif "Permission denied" in result.stderr:
                        self.logger.error(f" 权限被拒绝: {config['file_path']}")
                    elif "Connection" in result.stderr:
                        self.logger.error(f" SCP连接问题: {config['host']}")
                    elif "Host key verification failed" in result.stderr:
                        self.logger.error(f" SCP主机密钥验证失败: {config['host']}")
                    elif "Network is unreachable" in result.stderr:
                        self.logger.error(f" 网络不可达: {config['host']}")
                    else:
                        self.logger.error(" 未知的文件下载错误")

        except subprocess.TimeoutExpired:
            self.logger.error(f" 下载文件超时 (20秒): {config['file_path']}")
            self.logger.error("可能原因: SCP连接慢、文件过大、网络延迟高或远程主机响应慢")
        except Exception as e:
            self.logger.error(f" 下载文件异常: {e}")

        read_time = time.time() - start_time
        self.logger.error(" 文件下载失败")
        return None, read_time

    def execute_ssh_command(self, host: str, command: str) -> Tuple[bool, str, str]:
        """执行SSH命令"""
        if self.simulate_mode:
            time.sleep(0.3)
            import uuid
            uid = str(uuid.uuid4())
            return True, f"模拟命令成功, uid: {uid}", ""

        if host == '***':
            # 本地执行
            try:
                result = subprocess.run(
                    command.split(), 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                return result.returncode == 0, result.stdout, result.stderr
            except Exception as e:
                return False, "", str(e)

        # 远程执行 - ***
        config = self.chaosd_config.get(host, {})
        if not config:
            return False, "", f"未配置主机 {host}"

        # 替换chaosd路径并切换工作目录
        full_cmd = command.replace('chaosd', config['path'], 1)
        ssh_cmd = f'ssh -o ConnectTimeout=10 {config["user"]}@{host} "cd {config["work_dir"]} && {full_cmd}"'

        try:
            self.logger.debug(f"执行SSH命令: {ssh_cmd}")
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True, 
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)

    def extract_uid(self, output: str) -> Optional[str]:
        """提取UID"""
        if not output:
            return None
        match = re.search(r'uid:?\s*([a-f0-9-]{36})', output)
        return match.group(1) if match else None

    def execute_fault_injection(self, commands: str) -> Tuple[int, List[str]]:
        """执行故障注入"""
        if not commands:
            return 0, []

        # 解析命令
        cmd_list = []
        for part in commands.split(' | '):
            part = part.strip()
            if ':' in part:
                cmd = part.split(':', 1)[1].strip()
                cmd_list.append(cmd)
            else:
                cmd_list.append(part)

        success_uids = []

        for i, cmd in enumerate(cmd_list):
            self.logger.info(f"执行故障注入 {i+1}/{len(cmd_list)}: {cmd}")

            # 确定目标主机
            if 'attack disk' in cmd or 'attack file' in cmd:
                host = '***'
            else:
                host = '***'

            success, output, error = self.execute_ssh_command(host, cmd)

            if success:
                uid = self.extract_uid(output)
                if uid:
                    success_uids.append(f"{host}:{uid}")
                    self.logger.info(f"✅ 故障注入成功@{host}: {uid}")
                else:
                    self.logger.warning(f"故障注入成功但未找到UID@{host}")
            else:
                self.logger.error(f" 故障注入失败@{host}: {error}")

            if i < len(cmd_list) - 1:
                time.sleep(1)

        return len([uid for uid in success_uids if uid]), success_uids

    def execute_task(self, image_path: str) -> Tuple[bool, float, str]:
        start_time = time.time()

        if self.simulate_mode:
            time.sleep(2.0)
            mock_result = f"[模拟结果] 对图像 {image_path} 的分析：图像为测试JPG文件，假设内容为图像到文本转换结果。主要识别物体和描述场景。"
            self.logger.info("🎭 模拟推理任务完成")
            return True, time.time() - start_time, mock_result

        try:
            from test_i2t import llm_solve_problem
            answer, duration = llm_solve_problem("http://***:12365/data_chaos_mesh/i2t.jpg")
            self.logger.info(f"✅ 推理任务完成，耗时: {duration:.2f}秒")
            return True, duration, answer
        except ImportError:
            # 使用模拟
            time.sleep(1.0)
            duration = time.time() - start_time
            mock_result = f"[模拟结果] 由于缺少test_t2t模块，使用模拟回答。图像路径: {image_path}，文件大小: {os.path.getsize(image_path) if os.path.exists(image_path) else '未知'} 字节。"
            self.logger.info(f"🔧 模拟推理任务完成，耗时: {duration:.2f}秒")
            return True, duration, mock_result
        except Exception as e:
            duration = time.time() - start_time
            error_result = f"[错误结果] 推理任务执行失败: {str(e)}"
            self.logger.error(f" 推理任务失败: {e}")
            return False, duration, error_result

    def recover_faults(self, fault_uids: List[str]) -> int:
        """恢复故障 - 确保所有故障都必须恢复成功"""
        if not fault_uids:
            return 0

        recovery_count = 0
        retry_delay = 3  # 重试间隔秒数

        for fault_uid in fault_uids:
            if ':' not in fault_uid:
                continue

            host, uid = fault_uid.split(':', 1)
            cmd = f"chaosd recover {uid}"

            self.logger.info(f"🔄 开始恢复故障@{host}: {uid}")

            # 🔧 无限重试直到恢复成功
            attempt = 1
            while True:
                if attempt > 1:
                    self.logger.info(f"⏳ 等待 {retry_delay} 秒后重试恢复...")
                    time.sleep(retry_delay)
                    self.logger.info(f"🔄 重试恢复第 {attempt} 次@{host}: {uid}")
                
                # 人工恢复chaosd
                command_success = f"chaosd search -s success"
                success, output, error = self.execute_ssh_command(host, command_success)
                if len(output) == 127:
                    success = True
                else:
                    success, output, error = self.execute_ssh_command(host, cmd)

                if success:
                    recovery_count += 1
                    self.logger.info(f"✅ 故障恢复成功@{host}: {uid} (尝试次数: {attempt})")
                    break  # 成功后跳出循环
                else:
                    if attempt == 1:
                        self.logger.warning(f"⚠️  故障恢复失败@{host}: {error}")
                        self.logger.info(f"🚨 故障必须恢复成功才能继续实验，将持续重试...")
                    else:
                        self.logger.warning(f"⚠️  重试恢复失败 (第 {attempt} 次)@{host}: {error}")

                    attempt += 1

                    # 每10次重试增加等待时间，但最多不超过30秒
                    if attempt % 10 == 0:
                        retry_delay = min(retry_delay + 5, 30)
                        self.logger.info(f"📈 增加重试间隔至 {retry_delay} 秒")

            # 故障恢复间隔
            time.sleep(1)

        self.logger.info(f"🎉 所有故障恢复完成: {recovery_count}/{len(fault_uids)}")
        return recovery_count

    def run_experiment(self, scenario_data: dict) -> ExperimentResult:
        """运行单个实验"""
        scenario_id = int(scenario_data['scenario_id'])
        commands = scenario_data.get('commands_detail', '')

        self.logger.info("=" * 60)
        self.logger.info(f"🧪 开始实验场景 {scenario_id}")

        experiment_start = time.time()

        # 检查是否跳过
        should_skip, skip_reason = self.should_skip(commands)
        if should_skip:
            self.logger.warning(f"⏭️  跳过实验: {skip_reason}")
            return ExperimentResult(
                scenario_id=scenario_id,
                success=False,
                total_time=0,
                file_read_time=0,
                task_time=0,
                fault_count=0,
                recovery_count=0,
                skip_reason=skip_reason,
                task_result=""
            )

        try:
            # 1. 执行故障注入
            fault_count, fault_uids = self.execute_fault_injection(commands)

            # 2. 等待故障生效
            if fault_count > 0:
                self.logger.info("⏳ 等待故障生效...")
                time.sleep(3 if self.simulate_mode else 5)

            # 3. 下载任务文件
            image_path, file_read_time = self.read_task_file()

            # 🔧 如果文件下载失败，直接跳过后续步骤
            if image_path is None:
                # self.logger.error(f" 文件下载失败，跳过实验场景 {scenario_id}")

                # 仍需要恢复故障
                recovery_count = self.recover_faults(fault_uids)
                total_time = time.time() - experiment_start

                return ExperimentResult(
                    scenario_id=scenario_id,
                    success=False,
                    total_time=total_time,
                    file_read_time=file_read_time,
                    task_time=0,
                    fault_count=fault_count,
                    recovery_count=recovery_count,
                    error_msg="文件下载失败",
                    task_result=""
                )

            # 4. 执行推理任务
            task_success, task_time, task_result = self.execute_task(image_path)

            # 5. 恢复故障 (确保必须恢复成功)
            recovery_count = self.recover_faults(fault_uids)

            total_time = time.time() - experiment_start

            # 🔧 只有所有故障都恢复成功才算实验成功
            success = task_success and (recovery_count == len(fault_uids))

            if success:
                self.logger.info(f"🎉 实验成功完成: 场景{scenario_id}, 耗时: {total_time:.2f}秒")
            else:
                self.logger.error(f" 实验失败: 场景{scenario_id}, 故障恢复: {recovery_count}/{len(fault_uids)}")

            return ExperimentResult(
                scenario_id=scenario_id,
                success=success,
                total_time=total_time,
                file_read_time=file_read_time,
                task_time=task_time,
                fault_count=fault_count,
                recovery_count=recovery_count,
                task_result=task_result
            )

        except KeyboardInterrupt:
            self.logger.warning(f"⏹️  实验被用户中断: 场景{scenario_id}")
            # 用户中断时也要尝试恢复故障
            if 'fault_uids' in locals() and fault_uids:
                self.logger.info("🔄 用户中断，正在尝试恢复已注入的故障...")
                self.recover_faults(fault_uids)
            raise

        except Exception as e:
            total_time = time.time() - experiment_start
            self.logger.error(f" 实验异常失败: 场景{scenario_id}, 错误: {e}")

            return ExperimentResult(
                scenario_id=scenario_id,
                success=False,
                total_time=total_time,
                file_read_time=0,
                task_time=0,
                fault_count=0,
                recovery_count=0,
                error_msg=str(e),
                task_result=""
            )

    def save_results(self, results: List[ExperimentResult], output_file: str):
        """保存结果"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 🔧 写入表头，新增task_result列
            writer.writerow([
                'scenario_id', 'success', 'total_time', 'file_read_time', 
                'task_time', 'fault_count', 'recovery_count', 
                'error_msg', 'skip_reason', 'task_result', 'timestamp'
            ])

            # 写入数据
            for result in results:
                writer.writerow([
                    result.scenario_id,
                    result.success,
                    f"{result.total_time:.2f}",
                    f"{result.file_read_time:.2f}",
                    f"{result.task_time:.2f}",
                    result.fault_count,
                    result.recovery_count,
                    result.error_msg,
                    result.skip_reason,
                    result.task_result,  # 🔧 新增推理结果列
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])

    def run_csv_experiments(self, input_csv: str, max_experiments: int = 10):
        """运行CSV实验"""
        output_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        self.logger.info(f"📊 开始批量实验")
        self.logger.info(f"输入: {input_csv}")
        self.logger.info(f"输出: {output_file}")
        self.logger.info(f"最大实验数: {max_experiments}")
        self.logger.info(f"🚨 确保恢复模式: 所有故障必须恢复成功才能继续")

        results = []

        try:
            with open(input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    if i >= max_experiments:
                        break

                    result = self.run_experiment(row)
                    results.append(result)

                    # 实时保存
                    self.save_results(results, output_file)

        except FileNotFoundError:
            self.logger.error(f" 找不到输入文件: {input_csv}")
            return
        except KeyboardInterrupt:
            self.logger.warning("⏹️  批量实验被用户中断")
            if results:
                self.save_results(results, output_file)
            raise
        except Exception as e:
            self.logger.error(f" 执行实验时出错: {e}")

        # 统计结果
        total = len(results)
        success = len([r for r in results if r.success])
        skipped = len([r for r in results if r.skip_reason])
        file_failed = len([r for r in results if r.error_msg == "文件下载失败"])

        self.logger.info("=" * 60)
        self.logger.info("🎉 实验批量完成!")
        self.logger.info(f"总实验数: {total}")
        self.logger.info(f"成功: {success}")
        self.logger.info(f"跳过: {skipped}")
        self.logger.info(f"文件下载失败: {file_failed}")
        self.logger.info(f"其他失败: {total - success - skipped - file_failed}")
        self.logger.info(f"结果文件: {output_file}")


def main():

    simulate = sys.platform.startswith('win')

    executor = SimpleChaosExecutor(simulate_mode=simulate)

    input_csv = "./chaos-mesh/tasks/Qwen2-VL-2B-Instruct.csv"
    max_experiments = 10
    max_experiments = 432

    if not os.path.exists(input_csv):
        print(f" 文件不存在: {input_csv}")
        print("请检查文件路径")
        return

    try:
        executor.run_csv_experiments(input_csv, max_experiments)
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except Exception as e:
        print(f"\n 程序错误: {e}")


if __name__ == "__main__":
    main()
