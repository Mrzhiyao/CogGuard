# CogGuard Demo & Offline Evaluation

## Part 1: Interactive Demo (Frontend)

The demo is a Flask web app that loads **one** base model + dynamically switches LoRA adapters for EDU/OPS scenarios.

Large checkpoint files are distributed separately on [Google Drive](https://drive.google.com/drive/folders/1H-IYYixwArmH64ekR974wCBVUmnQoXkf?usp=sharing). The repository may keep only placeholder directories for `checkpoints/`.

### Demo Configuration

| Component | Path | Notes |
|-----------|------|-------|
| Base Model | `base_model/Qwen2.5-3B-Instruct/` | Qwen2.5-3B-Instruct |
| EDU LoRA | `checkpoints/edu/Qwen2.5-3B-Instruct/stu20_node3/` | Ray distributed, 3 nodes |
| OPS LoRA | `checkpoints/ops/Qwen2.5-3B-Instruct/` | Chaos engineering |
| EDU Test Data | `datas/test_raw_40_new.jsonl` | Demo raw text data, 611 records |
| OPS Test Data | `datas/test_raw_ops.jsonl` | Demo raw text data, 215 records |
| SCORE_START_TOKEN | 151666 | Qwen2.5-3B tokenizer |

### Run

```bash
cd demo/
python app.py --port 7860 --preload ops
```

Options: `--port` (default 7860), `--preload` (e.g. `ops`, `edu`, `ops edu`)

### Requirements

```bash
pip install torch transformers peft safetensors flask flask-cors
```

GPU memory: ~8 GB (3B model bfloat16 + LoRA)

---

## Part 2: Offline Evaluation (All Models)

All 7 base models with their EDU LoRA checkpoints and pre-tokenized test data for batch inference.

### Base Models

| Base Model | Parameters | SCORE_START_TOKEN |
|------------|-----------|-------------------|
| Qwen2.5-3B-Instruct | 3B | 151666 |
| Qwen3-0.6B | 0.6B | 151670 |
| Qwen3-1.7B | 1.7B | 151670 |
| Qwen3-8B | 8B | 151670 |
| Gemma-2-2B-it | 2B | 256001 |
| Gemma-2-9B-it | 9B | 256001 |
| Llama-3.1-8B | 8B | 128257 |

### LoRA Checkpoints — Education (EDU)

| Base Model | Checkpoint | Size |
|------------|-----------|------|
| Qwen2.5-3B-Instruct | `stu40` | 838M |
| Qwen2.5-3B-Instruct | `stu20_node2` (Ray, 2 nodes) | 838M |
| Qwen2.5-3B-Instruct | `stu20_node3` (Ray, 3 nodes) | 838M |
| Qwen2.5-3B-Instruct | `stu20_node4` (Ray, 4 nodes) | 838M |
| Qwen3-0.6B | `stu40` | 389M |
| Qwen3-1.7B | `stu40` | 742M |
| Qwen3-8B | `stu40` | 1.5G |
| Gemma-2-2B-it | `stu40` | 3.5G |
| Gemma-2-9B-it | `stu40` | 5.6G |
| Llama-3.1-8B | `stu40` | 3.3G |

### LoRA Checkpoints — AIOps (OPS)

| Base Model | Size |
|------------|------|
| Qwen2.5-3B-Instruct | 838M |

OPS scenario currently only has Qwen2.5-3B-Instruct.

### Pre-tokenized Test Data (`checkpoints/test_data/`)

Each file is tokenized with the corresponding base model's tokenizer. Used by `ssh/inference_*.py` for offline batch evaluation.

| Test File | Base Model |
|-----------|-----------|
| `validation_processed_40_new_qwen25_3b.jsonl` | Qwen2.5-3B-Instruct |
| `validation_processed_40_new_qwen3_06b.jsonl` | Qwen3-0.6B |
| `validation_processed_40_new_qwen3_17b.jsonl` | Qwen3-1.7B |
| `validation_processed_40_new_qwen3_8b.jsonl` | Qwen3-8B |
| `validation_processed_40_new_gemma2_2b.jsonl` | Gemma-2-2B-it |
| `validation_processed_40_new_gemma2_9b.jsonl` | Gemma-2-9B-it |
| `validation_processed_40_new_llama3_1_8b.jsonl` | Llama-3.1-8B |

---

## Directory Structure

```
demo/
├── app.py
├── templates/index.html
├── base_model/
│   ├── Qwen2.5-3B-Instruct/           
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   ├── Gemma-2-2B-it/
│   ├── Qwen3-8B/
│   ├── Llama-3.1-8B/
│   └── Gemma-2-9B-it/
├── checkpoints/
│   ├── edu/
│   │   ├── Qwen2.5-3B-Instruct/
│   │   │   ├── stu40/
│   │   │   ├── stu20_node2/
│   │   │   ├── stu20_node3/           ← demo default
│   │   │   └── stu20_node4/
│   │   ├── Qwen3-0.6B/stu40/
│   │   ├── Qwen3-1.7B/stu40/
│   │   ├── Gemma-2-2B-it/stu40/
│   │   ├── Qwen3-8B/stu40/
│   │   ├── Llama-3.1-8B/stu40/
│   │   └── Gemma-2-9B-it/stu40/
│   ├── ops/
│   │   └── Qwen2.5-3B-Instruct/
│   └── test_data/                      # Pre-tokenized evaluation data
├── ssh/
│   ├── inference_*.py                  # Offline inference scripts
│   └── outputs/                        # Exported extreme-error samples
├── datas/                              # Demo raw test data
└── ../edu_data/ and ../ops_data/       # Additional example data in repository root
```
