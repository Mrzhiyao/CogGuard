import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig,Result
from ray.train import RunConfig, CheckpointConfig
from torch.utils.data import Dataset
from ray.train.huggingface.transformers import RayTrainReportCallback
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any, List, Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
import sys
import os
import torch
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
import time
import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from datetime import datetime
from collections import defaultdict
import random
import json
import re
os.environ["NCCL_SOCKET_IFNAME"] = "enp4s0f2"
if "SWANLAB_API_KEY" not in os.environ:
    os.environ.setdefault("SWANLAB_MODE", "disabled")

t = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())


swanlab_callback = SwanLabCallback(
    project="LoRA-Qwen2.5-3B-Instruct-experiment", 
    experiment_name="Qwen2.5-3B-Instruct-LoRA-experiment-cross-steps-kv"
)

MARK_L = '<MARK_L>'
MARK_R = '<MARK_R>'
MARK_EG = '<MARK_EG>'
MARK_KG = '<MARK_KG>'
NUM_MARKER_TOKENS = 4


class AccumulationCycleTracker:
    def __init__(self, gradient_accumulation_steps: int):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_accumulation_step = 0
        self.cycle_id = 0
    
    def step(self) -> bool:
        self.current_accumulation_step += 1
        if self.current_accumulation_step >= self.gradient_accumulation_steps:
            self.current_accumulation_step = 0
            self.cycle_id += 1
            return True
        return False
    
    def is_new_cycle(self) -> bool:
        return self.current_accumulation_step == 0
    
    def get_current_step(self) -> int:
        return self.current_accumulation_step
    
    def get_cycle_id(self) -> int:
        return self.cycle_id
    
    def is_last_step_in_cycle(self) -> bool:
        return self.current_accumulation_step == self.gradient_accumulation_steps - 1


@dataclass
class CustomArguments:
    context_type: str = field(
        default='marking_incontext',
        metadata={
            "help": (
                "vanilla_incontext refers to the context without marking tokens"
                "marking_incontext refers to the context with marking tokens"
            )
        },
    )
    tune_params: str = field(
        default='added_params',
        metadata={
            "help": (
                "all_params refers to tuning all parameters of model"
                "added_params refers to tuning LoRA modules and added tokens embedding"
            )
        }
    )
    adapter_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to adapter directory for evaluation"}
    )
    enable_profiler: bool = field(
        default=False,
        metadata={"help": "启用性能分析"}
    )
    enable_data_augmentation: bool = field(
        default=True,
        metadata={"help": "Enable data augmentation with historical knowledge graphs"}
    )
    max_samples_per_problem: int = field(
        default=10,
        metadata={"help": "Maximum number of samples per problem after augmentation"}
    )
    num_historical_samples: int = field(
        default=3,
        metadata={"help": "Number of historical knowledge graphs to sample per original sample"}
    )
    enable_contrastive_loss: bool = field(
        default=True,
        metadata={"help": "Enable contrastive loss for same problem with different knowledge graphs"}
    )
    contrastive_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for contrastive loss"}
    )
    def __post_init__(self):
        if self.context_type not in ['vanilla_incontext', 'marking_incontext']:
            raise ValueError('--context_type error.')
        if self.tune_params not in ['all_params', 'added_params']:
            raise ValueError('--tune_params error.')


class ValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, test_file, validation_result_file, base_model_path, best_model_save_dir=None, ray_checkpoint_base=None):
        self.tokenizer = tokenizer
        self.test_file = test_file
        self.validation_result_file = validation_result_file
        self.base_model_path = base_model_path
        self.best_model_save_dir = best_model_save_dir
        self.ray_checkpoint_base = ray_checkpoint_base
        self.best_accuracy = 0.0
        self.best_checkpoint = None
        self.best_checkpoint_local = None
        
        if self.best_model_save_dir:
            os.makedirs(self.best_model_save_dir, exist_ok=True)
            print(f"[Validation] Best model will be saved to: {self.best_model_save_dir}")
        
        if not os.path.exists(validation_result_file):
            with open(validation_result_file, 'w', encoding='utf-8') as f:
                f.write("step\tepoch\ttrain_loss\tval_accuracy\ttimestamp\tcheckpoint_path\tbest_model_local_path\n")
    
    def on_save(self, args, state, control, **kwargs):

        if hasattr(state, 'process_index') and state.process_index != 0:
            print(f"[Validation] Skipping validation on non-main process (rank {state.process_index})")
            return
        
        skip_validation_steps = [1, 2, 3, 4, 5]
        if state.global_step in skip_validation_steps:
            print(f"[Validation] Skipping validation at step {state.global_step} (early training steps)")
            return
        
        if state.global_step > 0:
            import threading
            def delayed_validation():
                import time
                import torch
                print(f"[Validation] Validation thread started, waiting 5 seconds...")
                time.sleep(5)
                try:
                    if torch.cuda.is_available():
                        print(f"[Validation] Clearing GPU cache before validation...")
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    print(f"[Validation] Starting validation at step {state.global_step}...")
                    self._run_validation(args, state)
                    print(f"[Validation] Validation completed at step {state.global_step}")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[Validation] ERROR: Validation failed at step {state.global_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            validation_thread = threading.Thread(target=delayed_validation, daemon=True)
            validation_thread.start()
            print(f"[Validation] Started validation thread for step {state.global_step}, will run after 5 seconds...")
    
    def _run_validation(self, args, state):
        import torch
        from datetime import datetime
        import time
        
        validation_dir = os.path.dirname(os.path.abspath(__file__))
        if validation_dir not in sys.path:
            sys.path.insert(0, validation_dir)
        
        try:
            from validation import run_inference_on_dataset
            print(f"[Validation] Successfully imported validation module")
        except ImportError as e:
            print(f"[Validation] Error importing validation module: {e}")
            return
        
        try:
            checkpoint_path = None
            
            if state.global_step > 0:
                if self.ray_checkpoint_base and os.path.exists(self.ray_checkpoint_base):
                    print(f"[Validation] Searching in Ray checkpoint path: {self.ray_checkpoint_base}")
                    ray_checkpoint_dirs = []
                    if os.path.exists(self.ray_checkpoint_base):
                        for item in os.listdir(self.ray_checkpoint_base):
                            item_path = os.path.join(self.ray_checkpoint_base, item)
                            if os.path.isdir(item_path) and item.startswith("checkpoint_"):
                                ray_checkpoint_dirs.append((item_path, os.path.getmtime(item_path)))
                    
                    if ray_checkpoint_dirs:
                        ray_checkpoint_dirs.sort(key=lambda x: x[1], reverse=True)
                        latest_ray_checkpoint = ray_checkpoint_dirs[0][0]
                        print(f"[Validation] Found latest Ray checkpoint: {latest_ray_checkpoint}")
                        
                        for root, dirs, files in os.walk(latest_ray_checkpoint):
                            if "adapter_config.json" in files:
                                checkpoint_path = root
                                print(f"[Validation] Found HuggingFace checkpoint in Ray checkpoint: {checkpoint_path}")
                                break
                
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    checkpoint_dir = args.output_dir
                    hf_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{state.global_step}")
                    
                    if os.path.exists(hf_checkpoint_path):
                        checkpoint_path = hf_checkpoint_path
                        print(f"[Validation] Using HuggingFace checkpoint: {checkpoint_path}")
                    elif os.path.exists(checkpoint_dir):
                        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                        if checkpoints:
                            try:
                                checkpoints.sort(key=lambda x: int(x.split("-")[1]) if len(x.split("-")) > 1 and x.split("-")[1].isdigit() else 0)
                                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                                print(f"[Validation] Using latest HuggingFace checkpoint: {checkpoint_path}")
                            except:
                                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                                print(f"[Validation] Using HuggingFace checkpoint: {checkpoint_path}")
                
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    print(f"[Validation] ERROR: No checkpoint found at step {state.global_step}")
                    print(f"[Validation] Checked HuggingFace path: {args.output_dir}")
                    if os.path.exists(args.output_dir):
                        print(f"[Validation] HuggingFace output_dir contents: {os.listdir(args.output_dir)}")
                    if self.ray_checkpoint_base:
                        print(f"[Validation] Checked Ray checkpoint path: {self.ray_checkpoint_base}")
                        if os.path.exists(self.ray_checkpoint_base):
                            print(f"[Validation] Ray checkpoint_base contents: {os.listdir(self.ray_checkpoint_base)}")
                    print(f"[Validation] Skipping validation - checkpoint not found")
                    return
                
                print(f"[Validation] Using checkpoint path: {checkpoint_path}")
                
                adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
                
                if not os.path.exists(adapter_config_path) and not os.path.exists(trainer_state_path):
                    for root, dirs, files in os.walk(checkpoint_path):
                        if "adapter_config.json" in files or "trainer_state.json" in files:
                            checkpoint_path = root
                            adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
                            break
                
                if not os.path.exists(adapter_config_path):
                    print(f"[Validation] ERROR: Checkpoint at {checkpoint_path} is incomplete (missing adapter_config.json)")
                    if os.path.exists(checkpoint_path):
                        contents = os.listdir(checkpoint_path)
                        print(f"[Validation] Checkpoint contents: {contents}")
                        if any("rng_state" in f for f in contents):
                            print(f"[Validation] This appears to be a Ray checkpoint, not a HuggingFace Trainer checkpoint")
                            print(f"[Validation] HuggingFace checkpoints should be in: {args.output_dir}")
                            print(f"[Validation] Searching for HuggingFace checkpoint in subdirectories...")
                            for root, dirs, files in os.walk(checkpoint_path):
                                if "adapter_config.json" in files:
                                    checkpoint_path = root
                                    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                                    print(f"[Validation] Found adapter_config.json in subdirectory: {checkpoint_path}")
                                    break
                            if not os.path.exists(adapter_config_path):
                                print(f"[Validation] Still cannot find adapter_config.json, skipping validation")
                                return
                    else:
                        print(f"[Validation] Checkpoint path does not exist: {checkpoint_path}")
                        print(f"[Validation] Skipping validation for this checkpoint")
                        return
                
                print(f"[Validation] Checkpoint is complete, adapter_config.json found at: {adapter_config_path}")
                print(f"[Validation] Full checkpoint path: {checkpoint_path}")
                if os.path.exists(checkpoint_path):
                    contents = os.listdir(checkpoint_path)
                    print(f"[Validation] Checkpoint directory contents: {contents}")
                
                print(f"\n[Validation] Running validation on checkpoint: {checkpoint_path}")
                try:
                    accuracy = run_inference_on_dataset(
                        lora_path=checkpoint_path,
                        test_file=self.test_file,
                        tokenizer=self.tokenizer,
                        verbose=False  # 不打印详细过程
                    )
                except RuntimeError as e:
                    if "size of tensor" in str(e) or "must match" in str(e):
                        print(f"[Validation] ERROR: Dimension mismatch during inference")
                        print(f"[Validation] This may be due to input length mismatch between training and validation")
                        print(f"[Validation] Error details: {e}")
                        print(f"[Validation] Skipping validation for this checkpoint")
                        return
                    else:
                        raise
                
                train_loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0
                
                print(f"[Validation] Step {state.global_step}: Train Loss={train_loss:.4f}, Val Accuracy={accuracy:.4f} ({accuracy:.2%})")
                
                best_model_local_path = self.best_checkpoint_local if self.best_checkpoint_local else ""
                if accuracy > self.best_accuracy:
                    old_best_accuracy = self.best_accuracy
                    self.best_accuracy = accuracy
                    self.best_checkpoint = checkpoint_path
                    print(f"[Validation] New best accuracy: {accuracy:.4f} ({accuracy:.2%}) at checkpoint {checkpoint_path}")
                    
                    if self.best_model_save_dir:
                        try:
                            import shutil
                            best_model_dir_name = f"best_model_step_{state.global_step}_acc_{accuracy:.4f}"
                            best_model_local_path = os.path.join(self.best_model_save_dir, best_model_dir_name)
                            
                            if self.best_checkpoint_local and os.path.exists(self.best_checkpoint_local):
                                print(f"[Validation] Removing old best model: {self.best_checkpoint_local}")
                                shutil.rmtree(self.best_checkpoint_local, ignore_errors=True)
                            
                            print(f"[Validation] Copying best model to local: {best_model_local_path}")
                            if os.path.exists(checkpoint_path):
                                shutil.copytree(checkpoint_path, best_model_local_path, dirs_exist_ok=True)
                                self.best_checkpoint_local = best_model_local_path
                                print(f"[Validation] Best model saved to: {best_model_local_path}")
                            else:
                                print(f"[Validation] Warning: Checkpoint path {checkpoint_path} does not exist, cannot copy")
                                best_model_local_path = ""
                        except Exception as e:
                            print(f"[Validation] Error copying best model: {e}")
                            import traceback
                            traceback.print_exc()
                            best_model_local_path = ""
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result_line = f"{state.global_step}\t{state.epoch:.2f}\t{train_loss:.4f}\t{accuracy:.4f}\t{timestamp}\t{checkpoint_path}\t{best_model_local_path}\n"
                
                with open(self.validation_result_file, 'a', encoding='utf-8') as f:
                    f.write(result_line)
                
                print(f"[Validation] Validation result saved to: {self.validation_result_file}")
                
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[Validation] Error during validation: {e}")
            import traceback
            traceback.print_exc()
    
    def on_save_end(self, args, state, control, **kwargs):
        import torch
        from datetime import datetime
        
        print(f"\n[Validation] ========== on_save_end triggered ==========")
        print(f"[Validation] Step: {state.global_step}, Epoch: {state.epoch:.2f}")
        print(f"[Validation] Output dir: {args.output_dir}")
        
        if hasattr(state, 'process_index') and state.process_index != 0:
            print(f"[Validation] Skipping validation on non-main process (rank {state.process_index})")
            return
        
        validation_dir = os.path.dirname(os.path.abspath(__file__))
        if validation_dir not in sys.path:
            sys.path.insert(0, validation_dir)
        
        try:
            from validation import run_inference_on_dataset
            print(f"[Validation] Successfully imported validation module")
        except ImportError as e:
            print(f"[Validation] Error importing validation module: {e}")
            print(f"[Validation] Current sys.path: {sys.path[:5]}")  # 只打印前5个路径
            return
        
        try:
            checkpoint_path = None
            
            if state.global_step > 0:
                if self.ray_checkpoint_base and os.path.exists(self.ray_checkpoint_base):
                    print(f"[Validation] Searching in Ray checkpoint path: {self.ray_checkpoint_base}")
                    ray_checkpoint_dirs = []
                    if os.path.exists(self.ray_checkpoint_base):
                        for item in os.listdir(self.ray_checkpoint_base):
                            item_path = os.path.join(self.ray_checkpoint_base, item)
                            if os.path.isdir(item_path) and item.startswith("checkpoint_"):
                                ray_checkpoint_dirs.append((item_path, os.path.getmtime(item_path)))
                    
                    if ray_checkpoint_dirs:
                        ray_checkpoint_dirs.sort(key=lambda x: x[1], reverse=True)
                        latest_ray_checkpoint = ray_checkpoint_dirs[0][0]
                        print(f"[Validation] Found latest Ray checkpoint: {latest_ray_checkpoint}")
                        
                        for root, dirs, files in os.walk(latest_ray_checkpoint):
                            if "adapter_config.json" in files:
                                checkpoint_path = root
                                print(f"[Validation] Found HuggingFace checkpoint in Ray checkpoint: {checkpoint_path}")
                                break
                
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    checkpoint_dir = args.output_dir
                    hf_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{state.global_step}")
                    
                    if os.path.exists(hf_checkpoint_path):
                        checkpoint_path = hf_checkpoint_path
                        print(f"[Validation] Using HuggingFace checkpoint: {checkpoint_path}")
                    elif os.path.exists(checkpoint_dir):
                        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                        if checkpoints:
                            try:
                                checkpoints.sort(key=lambda x: int(x.split("-")[1]) if len(x.split("-")) > 1 and x.split("-")[1].isdigit() else 0)
                                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                                print(f"[Validation] Using latest HuggingFace checkpoint: {checkpoint_path}")
                            except:
                                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                                print(f"[Validation] Using HuggingFace checkpoint: {checkpoint_path}")
                
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    print(f"[Validation] ERROR: No checkpoint found at step {state.global_step}")
                    print(f"[Validation] Checked HuggingFace path: {args.output_dir}")
                    if os.path.exists(args.output_dir):
                        print(f"[Validation] HuggingFace output_dir contents: {os.listdir(args.output_dir)}")
                    if self.ray_checkpoint_base:
                        print(f"[Validation] Checked Ray checkpoint path: {self.ray_checkpoint_base}")
                        if os.path.exists(self.ray_checkpoint_base):
                            print(f"[Validation] Ray checkpoint_base contents: {os.listdir(self.ray_checkpoint_base)}")
                    print(f"[Validation] Skipping validation - checkpoint not found")
                    return
                
                print(f"[Validation] Using checkpoint path: {checkpoint_path}")
                
                import time
                print(f"[Validation] Waiting 3 seconds for checkpoint to be fully saved...")
                time.sleep(3)
                print(f"[Validation] Checkpoint should be saved now, checking...")
                
                adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
                
                if not os.path.exists(adapter_config_path) and not os.path.exists(trainer_state_path):
                    for root, dirs, files in os.walk(checkpoint_path):
                        if "adapter_config.json" in files or "trainer_state.json" in files:
                            checkpoint_path = root
                            adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
                            break
                
                if not os.path.exists(adapter_config_path):
                    print(f"[Validation] ERROR: Checkpoint at {checkpoint_path} is incomplete (missing adapter_config.json)")
                    if os.path.exists(checkpoint_path):
                        contents = os.listdir(checkpoint_path)
                        print(f"[Validation] Checkpoint contents: {contents}")
                        if any("rng_state" in f for f in contents):
                            print(f"[Validation] This appears to be a Ray checkpoint, not a HuggingFace Trainer checkpoint")
                            print(f"[Validation] HuggingFace checkpoints should be in: {args.output_dir}")
                            print(f"[Validation] Searching for HuggingFace checkpoint in subdirectories...")
                            for root, dirs, files in os.walk(checkpoint_path):
                                if "adapter_config.json" in files:
                                    checkpoint_path = root
                                    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                                    print(f"[Validation] Found adapter_config.json in subdirectory: {checkpoint_path}")
                                    break
                            if not os.path.exists(adapter_config_path):
                                print(f"[Validation] Still cannot find adapter_config.json, skipping validation")
                                return
                    else:
                        print(f"[Validation] Checkpoint path does not exist: {checkpoint_path}")
                        print(f"[Validation] Skipping validation for this checkpoint")
                        return
                
                print(f"[Validation] Checkpoint is complete, adapter_config.json found at: {adapter_config_path}")
                print(f"[Validation] Full checkpoint path: {checkpoint_path}")
                if os.path.exists(checkpoint_path):
                    contents = os.listdir(checkpoint_path)
                    print(f"[Validation] Checkpoint directory contents: {contents}")
                
                print(f"\n[Validation] Running validation on checkpoint: {checkpoint_path}")
                try:
                    accuracy = run_inference_on_dataset(
                        lora_path=checkpoint_path,
                        test_file=self.test_file,
                        tokenizer=self.tokenizer,
                        verbose=False  # 不打印详细过程
                    )
                except RuntimeError as e:
                    if "size of tensor" in str(e) or "must match" in str(e):
                        print(f"[Validation] ERROR: Dimension mismatch during inference")
                        print(f"[Validation] This may be due to input length mismatch between training and validation")
                        print(f"[Validation] Error details: {e}")
                        print(f"[Validation] Skipping validation for this checkpoint")
                        return
                    else:
                        raise
                
                train_loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0
                
                print(f"[Validation] Step {state.global_step}: Train Loss={train_loss:.4f}, Val Accuracy={accuracy:.4f} ({accuracy:.2%})")
                
                best_model_local_path = self.best_checkpoint_local if self.best_checkpoint_local else ""
                if accuracy > self.best_accuracy:
                    old_best_accuracy = self.best_accuracy
                    self.best_accuracy = accuracy
                    self.best_checkpoint = checkpoint_path
                    print(f"[Validation] New best accuracy: {accuracy:.4f} ({accuracy:.2%}) at checkpoint {checkpoint_path}")
                    
                    if self.best_model_save_dir:
                        try:
                            import shutil
                            best_model_dir_name = f"best_model_step_{state.global_step}_acc_{accuracy:.4f}"
                            best_model_local_path = os.path.join(self.best_model_save_dir, best_model_dir_name)
                            
                            if self.best_checkpoint_local and os.path.exists(self.best_checkpoint_local):
                                print(f"[Validation] Removing old best model: {self.best_checkpoint_local}")
                                shutil.rmtree(self.best_checkpoint_local, ignore_errors=True)
                            
                            print(f"[Validation] Copying best model to local: {best_model_local_path}")
                            if os.path.exists(checkpoint_path):
                                shutil.copytree(checkpoint_path, best_model_local_path, dirs_exist_ok=True)
                                self.best_checkpoint_local = best_model_local_path
                                print(f"[Validation] Best model saved to: {best_model_local_path}")
                            else:
                                print(f"[Validation] Warning: Checkpoint path {checkpoint_path} does not exist, cannot copy")
                                best_model_local_path = ""
                        except Exception as e:
                            print(f"[Validation] Error copying best model: {e}")
                            import traceback
                            traceback.print_exc()
                            best_model_local_path = ""
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result_line = f"{state.global_step}\t{state.epoch:.2f}\t{train_loss:.4f}\t{accuracy:.4f}\t{timestamp}\t{checkpoint_path}\t{best_model_local_path}\n"
                
                with open(self.validation_result_file, 'a', encoding='utf-8') as f:
                    f.write(result_line)
                
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[Validation] Error during validation: {e}")
            import traceback
            traceback.print_exc()


class CustomTrainer(Trainer):
    def __init__(self, custom_args=None, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_args = custom_args
        self.tokenizer = tokenizer
        
        if custom_args and custom_args.enable_contrastive_loss:
            self.cycle_tracker = AccumulationCycleTracker(
                gradient_accumulation_steps=self.args.gradient_accumulation_steps
            )
        else:
            self.cycle_tracker = None
        
        if tokenizer is not None:
            self.marker_token_ids = {
                'MARK_L': tokenizer.convert_tokens_to_ids(MARK_L),
                'MARK_R': tokenizer.convert_tokens_to_ids(MARK_R),
                'MARK_EG': tokenizer.convert_tokens_to_ids(MARK_EG),
                'MARK_KG': tokenizer.convert_tokens_to_ids(MARK_KG),
            }
        else:
            self.marker_token_ids = None
        
        self.last_loss_components = {}
        
        self.cycle_contrastive_data = {
            'problem_names': [],
            'logits_list': [],
            'sample_indices': [],
        }
        
        self.current_batch_problem_names = None

        self.profiler_enabled = custom_args.enable_profiler if custom_args else False
        self.profiler = None
        if self.profiler_enabled:
            try:
                import torch.profiler
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                )
                self.profiler.start()
            except Exception as e:
                self.profiler = None
                self.profiler_enabled = False

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        import time
        step_start_time = time.time()
        
        current_step = getattr(self.state, 'global_step', 0) if hasattr(self, 'state') else 0
        if current_step <= 10:
            print(f"[DEBUG training_step] Step {current_step}: 开始，时间: {time.strftime('%H:%M:%S', time.localtime())}")
        
        model.train()
        
        if hasattr(inputs, 'get') or isinstance(inputs, dict):
            self.current_batch_problem_names = inputs.get('problem_name', None) if hasattr(inputs, 'get') else (inputs['problem_name'] if 'problem_name' in inputs else None)
        else:
            self.current_batch_problem_names = None
        
        if current_step <= 10:
            print(f"[DEBUG training_step] Step {current_step}: 准备inputs，耗时: {time.time() - step_start_time:.2f}s")
        
        inputs = self._prepare_inputs(inputs)
        
        if self.current_batch_problem_names is not None:
            if not isinstance(inputs, dict):
                inputs = dict(inputs)
            inputs['problem_name'] = self.current_batch_problem_names

        if current_step <= 10:
            print(f"[DEBUG training_step] Step {current_step}: 开始compute_loss，耗时: {time.time() - step_start_time:.2f}s")
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if current_step <= 10:
            print(f"[DEBUG training_step] Step {current_step}: compute_loss完成，耗时: {time.time() - step_start_time:.2f}s")

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if current_step <= 10:
            print(f"[DEBUG training_step] Step {current_step}: 开始backward，耗时: {time.time() - step_start_time:.2f}s")
        
        self.accelerator.backward(loss)
        
        if current_step <= 10:
            print(f"[DEBUG training_step] Step {current_step}: backward完成，耗时: {time.time() - step_start_time:.2f}s")

        if self.custom_args.context_type == 'marking_incontext':
            actual_model = model.module if hasattr(model, 'module') else model
            if self.custom_args.tune_params == "added_params":
                try:
                    emb_weight = actual_model.get_input_embeddings().modules_to_save.default.weight
                except AttributeError:
                    emb_weight = actual_model.get_input_embeddings().weight
            elif self.custom_args.tune_params == "all_params":
                emb_weight = actual_model.get_input_embeddings().weight
            else:
                raise ValueError("No implementation.")
            
            if emb_weight.requires_grad:
                if emb_weight.grad is not None:
                    emb_weight.grad[:-NUM_MARKER_TOKENS, :] = 0

        if self.cycle_tracker:
            if current_step <= 10:
                print(f"[DEBUG training_step] Step {current_step}: 检查cycle_tracker，耗时: {time.time() - step_start_time:.2f}s")
            is_new_cycle = self.cycle_tracker.step()
            if is_new_cycle:
                if current_step <= 10:
                    print(f"[DEBUG training_step] Step {current_step}: 新cycle开始，清空对比学习数据，耗时: {time.time() - step_start_time:.2f}s")
                self.cycle_contrastive_data = {
                    'problem_names': [],
                    'logits_list': [],
                    'sample_indices': [],
                }

        if current_step <= 10:
            print(f"[DEBUG training_step] Step {current_step}: training_step完成，总耗时: {time.time() - step_start_time:.2f}s")
            if hasattr(self, 'state') and self.state is not None:
                accumulation_step = getattr(self.state, 'global_step', 0) % self.args.gradient_accumulation_steps
                if accumulation_step == self.args.gradient_accumulation_steps - 1:
                    print(f"[DEBUG training_step] Step {current_step}: 这是accumulation cycle的最后一步，下一步将调用optimizer.step()")

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer=None, optimizer_idx: int = None, optimizer_closure=None, on_tp_phase: bool = False, **kwargs):
        import time
        current_step = 0
        if hasattr(self, 'state') and self.state is not None:
            current_step = getattr(self.state, 'global_step', 0)
        
        if current_step <= 10:
            print(f"[DEBUG optimizer_step] Step {current_step}: optimizer_step开始，时间: {time.strftime('%H:%M:%S', time.localtime())}")
        
        result = super().optimizer_step(
            epoch=epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            optimizer_closure=optimizer_closure,
            on_tp_phase=on_tp_phase,
            **kwargs
        )
        
        if current_step <= 10:
            print(f"[DEBUG optimizer_step] Step {current_step}: optimizer_step完成，时间: {time.strftime('%H:%M:%S', time.localtime())}")
        
        return result

    def compute_loss(self, model, inputs, return_outputs=False):
        import time
        loss_start_time = time.time()
        
        current_step = 0
        if hasattr(self, 'state') and self.state is not None:
            current_step = getattr(self.state, 'global_step', 0)
        
        if self.profiler:
            self.profiler.step()

        if not isinstance(inputs, dict):
            if hasattr(inputs, 'keys'):
                inputs = dict(inputs)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        batch_size = input_ids.shape[0]
        
        if current_step <= 10:
            print(f"[DEBUG compute_loss] Step {current_step}: 准备model_inputs，耗时: {time.time() - loss_start_time:.2f}s")
        
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        outputs = model(**model_inputs)
        loss = outputs.loss
        logits = outputs.logits
        
        contrastive_loss = None
        if (self.custom_args and 
            self.custom_args.enable_contrastive_loss and
            self.cycle_tracker is not None):
            
            try:
                problem_names = None
                if self.current_batch_problem_names is not None:
                    problem_names = self.current_batch_problem_names
                elif isinstance(inputs, dict) and 'problem_name' in inputs:
                    problem_names = inputs['problem_name']
                elif hasattr(inputs, 'get'):
                    problem_names = inputs.get('problem_name', None)
                
                if problem_names and isinstance(problem_names, (list, torch.Tensor)):
                    if isinstance(problem_names, torch.Tensor):
                        problem_names = problem_names.cpu().tolist()
                    
                    if len(problem_names) == batch_size:
                        score_repr = logits.detach()
                        
                        for i, pname in enumerate(problem_names):
                            if pname and str(pname) != '':
                                self.cycle_contrastive_data['problem_names'].append(str(pname))
                                self.cycle_contrastive_data['logits_list'].append(score_repr[i].clone())
                                self.cycle_contrastive_data['sample_indices'].append(
                                    len(self.cycle_contrastive_data['problem_names']) - 1
                                )
                
                cycle_current_step = self.cycle_tracker.get_current_step()
                is_last_step_in_cycle = (cycle_current_step == self.args.gradient_accumulation_steps - 1)
                
                if is_last_step_in_cycle and len(self.cycle_contrastive_data['problem_names']) > 0:
                    problem_groups = defaultdict(list)
                    for idx, pname in enumerate(self.cycle_contrastive_data['problem_names']):
                        if pname and str(pname) != '':
                            problem_groups[pname].append(idx)
                    
                    if len(problem_groups) > 0:
                        contrastive_losses = []
                        
                        for pname, indices in problem_groups.items():
                            if len(indices) > 1:
                                group_logits = torch.stack([
                                    self.cycle_contrastive_data['logits_list'][idx] 
                                    for idx in indices
                                ])
                                
                                for i in range(group_logits.shape[0]):
                                    for j in range(i+1, group_logits.shape[0]):
                                        sim = torch.cosine_similarity(
                                            group_logits[i:i+1], 
                                            group_logits[j:j+1],
                                            dim=1
                                        )
                                        contrastive_losses.append(sim ** 2)
                        
                        if contrastive_losses:
                            contrastive_loss = torch.stack(contrastive_losses).mean()
                            contrastive_loss = contrastive_loss.to(logits.device).requires_grad_(True)
                            
                            if self.args.logging_steps > 0:
                                step = getattr(self.state, 'global_step', 0)
                                if step % self.args.logging_steps == 0:
                                    print(f"[对比学习成功] Step {step}, Cycle {self.cycle_tracker.get_cycle_id()}: "
                                          f"收集了 {len(self.cycle_contrastive_data['problem_names'])} 个样本, "
                                          f"{len(problem_groups)} 个题目组, "
                                          f"计算了 {len(contrastive_losses)} 个对比loss对, "
                                          f"平均对比loss: {contrastive_loss.item():.6f}")
            except Exception as e:
                if self.args.logging_steps > 0:
                    step = getattr(self.state, 'global_step', 0)
                    if step % self.args.logging_steps == 0:
                        print(f"Warning: Contrastive loss calculation failed: {e}")
                        import traceback
                        traceback.print_exc()
        
        base_loss = loss.clone().detach() if isinstance(loss, torch.Tensor) else loss
        contrastive_loss_value = None
        if contrastive_loss is not None and self.custom_args.contrastive_loss_weight > 0:
            contrastive_loss_value = contrastive_loss.detach() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
            loss = loss + self.custom_args.contrastive_loss_weight * contrastive_loss
        
        if current_step <= 10:
            print(f"[DEBUG compute_loss] Step {current_step}: 准备记录loss组成部分，耗时: {time.time() - loss_start_time:.2f}s")
        
        self.last_loss_components = {
            'base_loss': float(base_loss.item()) if isinstance(base_loss, torch.Tensor) else float(base_loss),
            'contrastive_loss': float(contrastive_loss_value.item()) if contrastive_loss_value is not None and isinstance(contrastive_loss_value, torch.Tensor) else (float(contrastive_loss_value) if contrastive_loss_value is not None else 0.0),
            'contrastive_loss_weight': self.custom_args.contrastive_loss_weight if self.custom_args else 0.0,
            'total_loss': float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss),
        }
        if contrastive_loss_value is not None:
            self.last_loss_components['contrastive_loss_scaled'] = self.last_loss_components['contrastive_loss'] * self.last_loss_components['contrastive_loss_weight']
        else:
            self.last_loss_components['contrastive_loss_scaled'] = 0.0
        
        if current_step <= 10:
            print(f"[DEBUG compute_loss] Step {current_step}: compute_loss完成，总耗时: {time.time() - loss_start_time:.2f}s")
        
        return (loss, outputs) if return_outputs else loss


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        return (loss, None, None)
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        import time
        log_start_time = time.time()
        
        current_step = 0
        if hasattr(self, 'state') and self.state is not None:
            current_step = getattr(self.state, 'global_step', 0)
        
        if current_step <= 10:
            print(f"[DEBUG log] Step {current_step}: log方法开始，时间: {time.strftime('%H:%M:%S', time.localtime())}")
        
        is_main_process = True
        try:
            if hasattr(self.args, 'local_rank'):
                is_main_process = (self.args.local_rank == -1 or self.args.local_rank == 0)
            elif hasattr(self.args, 'process_index'):
                is_main_process = (self.args.process_index == 0)
            elif torch.distributed.is_initialized():
                is_main_process = (torch.distributed.get_rank() == 0)
        except Exception:
            is_main_process = True
        
        if is_main_process:
            if hasattr(self, 'state') and self.state.global_step % 100 == 0:
                print(f"[调试] Step {self.state.global_step}: log方法被调用，logs keys: {list(logs.keys())}")
                print(f"[调试] report_to: {self.args.report_to}")
            
            if self.last_loss_components:
                logs['loss/base_loss'] = self.last_loss_components.get('base_loss', 0.0)
                logs['loss/contrastive_loss'] = self.last_loss_components.get('contrastive_loss', 0.0)
                logs['loss/contrastive_loss_weight'] = self.last_loss_components.get('contrastive_loss_weight', 0.0)
                logs['loss/contrastive_loss_scaled'] = self.last_loss_components.get('contrastive_loss_scaled', 0.0)
                logs['loss/total_loss'] = self.last_loss_components.get('total_loss', 0.0)
                total = self.last_loss_components.get('total_loss', 1.0)
                if total > 0:
                    logs['loss/contrastive_loss_ratio'] = self.last_loss_components.get('contrastive_loss_scaled', 0.0) / total
            
            if current_step <= 10:
                print(f"[DEBUG log] Step {current_step}: 准备调用super().log()，耗时: {time.time() - log_start_time:.2f}s")
            
            super().log(logs, start_time)
            
            if current_step <= 10:
                print(f"[DEBUG log] Step {current_step}: super().log()完成，总耗时: {time.time() - log_start_time:.2f}s")
        else:
            if current_step <= 10:
                print(f"[DEBUG log] Step {current_step}: 非主进程，跳过日志记录")
            pass


def train_func(config):
    import os
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "enp4s0f2"
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if "SWANLAB_API_KEY" not in os.environ:
        os.environ.setdefault("SWANLAB_MODE", "disabled")
    
    tokenizer = AutoTokenizer.from_pretrained(f'./Qwen2.5-3B-Instruct')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    new_tokens = [MARK_L, MARK_R, MARK_EG, MARK_KG]
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)

    enhanced_data_file = config.get("enhanced_data_file", './datasets/train_set/split_data_3b_ray_20_ray_3nodes.jsonl')

    print(f"加载数据集: {enhanced_data_file}")
    train_dataset = load_dataset('json', data_files=enhanced_data_file)['train']
    
    available_columns = train_dataset.column_names
    columns_to_select = ['input_ids', 'labels']
    if 'attention_mask' in available_columns:
        columns_to_select.append('attention_mask')
    for col in ['problem_name', 'user_id', 'timestamp']:
        if col in available_columns:
            columns_to_select.append(col)
    
    train_dataset = train_dataset.select_columns(columns_to_select)
    dataset_size = len(train_dataset)
    
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"[DEBUG] Worker rank: {rank}/{world_size-1}, 数据集大小: {dataset_size}")
        else:
            print(f"[DEBUG] 非DDP模式，数据集大小: {dataset_size}")
    except:
        print(f"[DEBUG] 无法获取worker信息，数据集大小: {dataset_size}")
    
    print(f"最终数据集大小: {dataset_size}")
    
    global_batch_size = (
        config["per_device_train_batch_size"]
        * config["num_workers"]
        * config["gradient_accumulation_steps"]
    )
    steps_per_epoch = math.ceil(dataset_size / global_batch_size)
    max_steps = steps_per_epoch * config["num_train_epochs"]
    
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 改为序列分类任务
        target_modules=config["target_modules"],
        inference_mode=config["inference_mode"],

        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],

        modules_to_save=["embed_tokens", "score"],  # 添加score（Qwen/Llama系列使用score作为分类头名称），确保分类头被训练
        bias="none",
        fan_in_fan_out=False,
    )
    args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        
        logging_steps=config["logging_steps"],
        num_train_epochs=config["num_train_epochs"],
        save_steps=steps_per_epoch * 4, 
        learning_rate=config["learning_rate"],
        save_on_each_node=config["save_on_each_node"],  
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=config["report_to"],
        max_steps=max_steps,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        deepspeed=None,
        fsdp=[],
        bf16=True,
        tf32=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f'./Qwen2.5-3B-Instruct',
        num_labels=5,
        problem_type="single_label_classification",
        device_map=None,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model.config.pad_token_id to {model.config.pad_token_id}")

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def custom_collate_fn(features):
        non_tensor_fields = {}
        if len(features) > 0:
            for field in ['problem_name', 'user_id', 'timestamp']:
                if field in features[0]:
                    non_tensor_fields[field] = [f.get(field, '') for f in features]
        
        tensor_features = []
        for f in features:
            tensor_feature = {k: v for k, v in f.items() if k not in non_tensor_fields}
            tensor_features.append(tensor_feature)
        
        standard_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        batch = standard_collator(tensor_features)
        
        for field, values in non_tensor_fields.items():
            batch[field] = values
        
        return batch
    
    validation_callback = None
    if config.get("enable_validation", True):
        validation_result_file = config.get("validation_result_file", "./datasets/validation_results.txt")
        test_file = config.get("test_file", "./datasets/validation_processed_20_qwen25_3b.jsonl")
        best_model_save_dir = config.get("best_model_save_dir", "./datasets/best_models")
        ray_checkpoint_base = config.get("checkpoint_path", None)
        if ray_checkpoint_base:
            run_name = "Qwen2.5-3B-Instruct-LoRA-ray-train-cross-steps-kv"
            ray_checkpoint_base = os.path.join(ray_checkpoint_base, run_name)
        validation_callback = ValidationCallback(
            tokenizer=tokenizer,
            test_file=test_file,
            validation_result_file=validation_result_file,
            base_model_path=f'./Qwen2.5-3B-Instruct',
            best_model_save_dir=best_model_save_dir,
            ray_checkpoint_base=ray_checkpoint_base
        )
        print(f"[Validation] Validation enabled. Results will be saved to: {validation_result_file}")
        if best_model_save_dir:
            print(f"[Validation] Best models will be saved to: {best_model_save_dir}")
        if ray_checkpoint_base:
            print(f"[Validation] Ray checkpoint search path: {ray_checkpoint_base}")
    
    # 构建callbacks列表
    callbacks = [swanlab_callback]
    callbacks = []
    if validation_callback:
        callbacks.append(validation_callback)
    
    trainer = CustomTrainer(
        custom_args=CustomArguments(
            context_type='marking_incontext',
            tune_params='added_params',
            adapter_dir=None,
            enable_contrastive_loss=config.get("enable_contrastive_loss", True),
            contrastive_loss_weight=config.get("contrastive_loss_weight", 0.05),
        ),
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        callbacks=callbacks,
    )
    trainer.add_callback(RayTrainReportCallback())
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    trainer.train()


def main():
    ray.init()
    
    config = {
        "use_gpu": True,
        "num_workers": 3,
        "checkpoint_path": f"/path/to/lora_checkpoints/{t}",
        "task_type": TaskType.SEQ_CLS,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "inference_mode": False,
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "output_dir": f"./result/output/Qwen2.5-3B-Instruct_lora_cross_steps_kv/{t}",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 6,
        "logging_steps": 10,
        "num_train_epochs": 48,
        "num_for_save" : 1,
        "learning_rate": 3e-5,
        "save_on_each_node": False,
        "gradient_checkpointing": True,
        "report_to": ["swanlab"],
        "enhanced_data_file": './datasets/train_set/split_data_3b_ray_20_ray_3nodes.jsonl',
        "enable_contrastive_loss": False,
        "contrastive_loss_weight": 0.1,
        "enable_validation": False,
        "test_file": './datasets/validation_processed_20_qwen25_3b.jsonl',
        "validation_result_file": './datasets/validation_results_20_qwen25_3b_our_3nodes.txt',
        "best_model_save_dir": './datasets/best_models',
        "enable_profiler": False,
    }

    scaling_config = ScalingConfig(
        num_workers=config["num_workers"], 
        use_gpu=config["use_gpu"],
    )

    ray_trainer = TorchTrainer(
        train_func,
        train_loop_config=config, 
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="Qwen2.5-3B-Instruct-LoRA-ray-train-cross-steps-kv",
            storage_path = config["checkpoint_path"],
            checkpoint_config=CheckpointConfig(
                num_to_keep=15  # 保留最新 checkpoint
            )
        )
    )
    result: Result = ray_trainer.fit()        
    

if __name__ == "__main__":
    main()

