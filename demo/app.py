import os
import json
import glob
import re
import torch
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file as load_safetensors
from peft import PeftModel

app = Flask(__name__)
CORS(app)

CLASS_TO_SCORE_RANGE = {0: "0-20", 1: "20-40", 2: "40-60", 3: "60-80", 4: "80-100"}
CLASS_TO_VALUE = {0: 10, 1: 30, 2: 50, 3: 70, 4: 90}
SCORE_START_TOKEN = 151666

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DEMO_DIR, "models")


def resolve_path(*parts, env_var=None):
    if env_var and os.environ.get(env_var):
        return os.path.abspath(os.environ[env_var])
    return os.path.abspath(os.path.join(*parts))


BASE_MODEL_PATH = resolve_path(MODEL_DIR, "Qwen2.5-3B-Instruct", env_var="COGGUARD_BASE_MODEL_PATH")
MARK_TOKENS = ["<MARK_L>", "<MARK_R>", "<MARK_EG>", "<MARK_KG>"]

DATA_DIR = os.path.join(DEMO_DIR, "datas")
EDU_DATA_FILE = os.path.join(DATA_DIR, "test_raw_40_new.jsonl")
OPS_DATA_FILE = os.path.join(DATA_DIR, "test_raw_ops.jsonl")

LORA_REGISTRY = {
    "ops": {
        "name": "AIOps - Chaos Engineering",
        "path": resolve_path(MODEL_DIR, "lora", "ops", "checkpoint", env_var="COGGUARD_OPS_LORA_PATH"),
        "scenario": "ops",
        "description": "Predicts task execution score under chaos fault injection",
    },
    "edu": {
        "name": "Education - Student Score",
        "path": resolve_path(MODEL_DIR, "lora", "edu", "checkpoint", env_var="COGGUARD_EDU_LORA_PATH"),
        "scenario": "edu",
        "description": "Predicts student performance score based on knowledge profile",
    },
}

base_model = None
tokenizer = None
active_adapter = None
loaded_adapters = set()

edu_data = []
ops_data = []


def load_datasets():
    global edu_data, ops_data
    if os.path.exists(EDU_DATA_FILE):
        with open(EDU_DATA_FILE, "r", encoding="utf-8") as f:
            edu_data = [json.loads(line) for line in f]
        print(f"[DATA] Loaded {len(edu_data)} EDU samples")
    if os.path.exists(OPS_DATA_FILE):
        with open(OPS_DATA_FILE, "r", encoding="utf-8") as f:
            ops_data = [json.loads(line) for line in f]
        print(f"[DATA] Loaded {len(ops_data)} OPS samples")


def parse_edu_e_field(e_text):
    """Split the combined 'e' field into entity_graph and knowledge_graph portions."""
    eg_part = ""
    kg_part = ""
    marker = "[Knowledge Point Graph"
    idx = e_text.find(marker)
    if idx != -1:
        eg_part = e_text[:idx].strip()
        kg_part = e_text[idx:].strip()
        eg_header = "[Entity Graph - Directly Extracted from Problems]:"
        eg_part = eg_part.replace("Student Knowledge Summary (Combined):\n\n", "").replace(eg_header, "").strip()
        kg_header_match = re.match(r"\[Knowledge Point Graph[^\]]*\]:", kg_part)
        if kg_header_match:
            kg_part = kg_part[kg_header_match.end():].strip()
    else:
        eg_part = e_text
    return eg_part, kg_part


def parse_ops_e_field(e_text):
    """Split the OPS 'e' field into hardware_profile and fault_pattern."""
    hw_part = ""
    fault_part = ""
    marker = "## Runtime Profile Characteristic"
    idx = e_text.find(marker)
    if idx != -1:
        hw_part = e_text[:idx].strip()
        fault_part = e_text[idx:].strip()
        hw_part = hw_part.replace("## Hardware Profile Characteristic", "").strip()
        fault_part = fault_part.replace("## Runtime Profile Characteristic", "").strip()
    else:
        hw_part = e_text
    return hw_part, fault_part


def get_checkpoint_vocab_size(lora_path):
    sf_files = glob.glob(os.path.join(lora_path, "adapter_model.safetensors"))
    if sf_files:
        sd = load_safetensors(sf_files[0])
        for k, v in sd.items():
            if "embed_tokens" in k and "weight" in k:
                return v.shape[0]
    return None


def load_base_model():
    global base_model, tokenizer
    if base_model is not None:
        return
    print(f"[BASE] Loading tokenizer from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.add_tokens(MARK_TOKENS, special_tokens=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[BASE] Loading base model (SequenceClassification, num_labels=5)...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True, num_labels=5,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    print(f"[BASE] Base model loaded. Vocab size: {base_model.config.vocab_size}")


def load_adapter(adapter_id):
    global base_model, active_adapter
    if adapter_id not in LORA_REGISTRY:
        raise ValueError(f"Unknown adapter: {adapter_id}")
    load_base_model()
    lora_info = LORA_REGISTRY[adapter_id]
    lora_path = lora_info["path"]
    if adapter_id in loaded_adapters:
        if active_adapter != adapter_id:
            print(f"[ADAPTER] Switching to: {adapter_id}")
            base_model.set_adapter(adapter_id)
            active_adapter = adapter_id
        return
    ckpt_vocab_size = get_checkpoint_vocab_size(lora_path)
    current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    if ckpt_vocab_size and ckpt_vocab_size != current_vocab_size:
        print(f"[ADAPTER] Resizing embeddings: {current_vocab_size} -> {ckpt_vocab_size}")
        base_model.resize_token_embeddings(ckpt_vocab_size)
    print(f"[ADAPTER] Loading LoRA '{adapter_id}' from {lora_path}...")
    if len(loaded_adapters) == 0:
        base_model = PeftModel.from_pretrained(base_model, model_id=lora_path, adapter_name=adapter_id)
    else:
        base_model.load_adapter(lora_path, adapter_name=adapter_id)
    base_model.set_adapter(adapter_id)
    base_model.eval()
    loaded_adapters.add(adapter_id)
    active_adapter = adapter_id
    print(f"[ADAPTER] '{adapter_id}' loaded. Total: {loaded_adapters}")


def run_inference(adapter_id, src_text, eg_text, kg_text):
    load_adapter(adapter_id)
    mark_l_id = tokenizer.convert_tokens_to_ids("<MARK_L>")
    mark_r_id = tokenizer.convert_tokens_to_ids("<MARK_R>")
    mark_eg_id = tokenizer.convert_tokens_to_ids("<MARK_EG>")
    mark_kg_id = tokenizer.convert_tokens_to_ids("<MARK_KG>")
    src_ids = tokenizer.encode(src_text, add_special_tokens=False)
    eg_ids = tokenizer.encode(eg_text, add_special_tokens=False)
    kg_ids = tokenizer.encode(kg_text, add_special_tokens=False)
    input_ids = src_ids + [mark_l_id] + [mark_eg_id] + eg_ids + [mark_kg_id] + kg_ids + [mark_r_id] + [SCORE_START_TOKEN]
    input_tensor = torch.tensor([input_ids]).to(base_model.device)
    attention_mask = torch.ones_like(input_tensor).to(base_model.device)
    with torch.no_grad():
        outputs = base_model(input_ids=input_tensor, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()[0]
    return {
        "predicted_class": int(predicted_class),
        "predicted_range": CLASS_TO_SCORE_RANGE[predicted_class],
        "predicted_value": CLASS_TO_VALUE[predicted_class],
        "probabilities": {CLASS_TO_SCORE_RANGE[i]: round(float(probs[i]) * 100, 2) for i in range(5)},
        "input_length": len(input_ids),
        "adapter": adapter_id,
    }


# ==================== API Routes ====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/adapters")
def list_adapters():
    result = {}
    for aid, info in LORA_REGISTRY.items():
        result[aid] = {
            "name": info["name"], "scenario": info["scenario"],
            "description": info["description"],
            "loaded": aid in loaded_adapters, "active": aid == active_adapter,
        }
    return jsonify({"adapters": result, "active": active_adapter})


@app.route("/api/adapters/load", methods=["POST"])
def load_adapter_api():
    try:
        data = request.json
        adapter_id = data.get("adapter_id")
        if not adapter_id:
            return jsonify({"success": False, "error": "adapter_id is required"}), 400
        load_adapter(adapter_id)
        return jsonify({"success": True, "active": active_adapter, "loaded_adapters": list(loaded_adapters)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== Data Browse APIs ====================

@app.route("/api/data/edu/students")
def edu_students():
    students = defaultdict(lambda: {"count": 0, "problems": []})
    for i, d in enumerate(edu_data):
        uid = str(d["user_id"])
        students[uid]["count"] += 1
        if len(students[uid]["problems"]) < 3:
            students[uid]["problems"].append(d.get("problem_name", ""))
    result = [{"user_id": uid, "count": info["count"], "sample_problems": info["problems"]}
              for uid, info in sorted(students.items(), key=lambda x: -x[1]["count"])]
    return jsonify({"students": result, "total": len(edu_data)})


@app.route("/api/data/edu/samples")
def edu_samples():
    user_id = request.args.get("user_id", "")
    samples = []
    for i, d in enumerate(edu_data):
        if user_id and str(d["user_id"]) != user_id:
            continue
        samples.append({
            "idx": i, "user_id": str(d["user_id"]),
            "problem_name": d.get("problem_name", ""),
            "timestamp": d.get("timestamp", ""),
            "label": d.get("label", 0),
        })
    return jsonify({"samples": samples})


@app.route("/api/data/edu/detail/<int:idx>")
def edu_detail(idx):
    if idx < 0 or idx >= len(edu_data):
        return jsonify({"success": False, "error": "Invalid index"}), 404
    d = edu_data[idx]
    eg, kg = parse_edu_e_field(d.get("e", ""))
    return jsonify({
        "success": True,
        "src": d.get("src", ""),
        "entity_graph": eg,
        "knowledge_graph": kg,
        "user_id": str(d.get("user_id", "")),
        "problem_name": d.get("problem_name", ""),
        "timestamp": d.get("timestamp", ""),
        "label": d.get("label", 0),
    })


@app.route("/api/data/ops/hosts")
def ops_hosts():
    hosts = defaultdict(lambda: {"count": 0, "task_types": set(), "fault_samples": []})
    for i, d in enumerate(ops_data):
        h = d.get("host", "unknown")
        hosts[h]["count"] += 1
        hosts[h]["task_types"].add(d.get("task_type", ""))
        if len(hosts[h]["fault_samples"]) < 3:
            hosts[h]["fault_samples"].append(d.get("fault_types", ""))
    result = [{"host": h, "count": info["count"], "task_types": sorted(info["task_types"]),
               "sample_faults": info["fault_samples"]}
              for h, info in sorted(hosts.items())]
    return jsonify({"hosts": result, "total": len(ops_data)})


@app.route("/api/data/ops/samples")
def ops_samples():
    host = request.args.get("host", "")
    samples = []
    for i, d in enumerate(ops_data):
        if host and d.get("host", "") != host:
            continue
        samples.append({
            "idx": i, "scenario_id": d.get("scenario_id", ""),
            "host": d.get("host", ""), "task_type": d.get("task_type", ""),
            "fault_types": d.get("fault_types", ""),
            "total_faults": d.get("total_faults", 0),
            "success": d.get("success", ""),
            "label": d.get("label", 0),
        })
    return jsonify({"samples": samples})


@app.route("/api/data/ops/detail/<int:idx>")
def ops_detail(idx):
    if idx < 0 or idx >= len(ops_data):
        return jsonify({"success": False, "error": "Invalid index"}), 404
    d = ops_data[idx]
    hw, fault = parse_ops_e_field(d.get("e", ""))
    return jsonify({
        "success": True,
        "src": d.get("src", ""),
        "hardware_profile": hw,
        "fault_pattern": fault,
        "scenario_id": d.get("scenario_id", ""),
        "host": d.get("host", ""),
        "task_type": d.get("task_type", ""),
        "fault_types": d.get("fault_types", ""),
        "label": d.get("label", 0),
        "success": d.get("success", ""),
    })


# ==================== Predict APIs ====================

@app.route("/api/predict/raw", methods=["POST"])
def predict_raw():
    try:
        data = request.json
        adapter_id = data.get("adapter_id", active_adapter)
        if not adapter_id:
            return jsonify({"success": False, "error": "No adapter specified"}), 400
        result = run_inference(adapter_id, data.get("src_text", ""), data.get("eg_text", ""), data.get("kg_text", ""))
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict/ops", methods=["POST"])
def predict_ops():
    try:
        data = request.json
        src = data.get("src_text", "")
        eg = data.get("eg_text", "")
        kg = data.get("kg_text", "")
        result = run_inference("ops", src, eg, kg)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict/edu", methods=["POST"])
def predict_edu():
    try:
        data = request.json
        src = data.get("src_text", "")
        eg = data.get("eg_text", "")
        kg = data.get("kg_text", "")
        result = run_inference("edu", src, eg, kg)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok", "base_model_loaded": base_model is not None,
        "loaded_adapters": list(loaded_adapters), "active_adapter": active_adapter,
        "data": {"edu_samples": len(edu_data), "ops_samples": len(ops_data)},
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--preload", nargs="*", choices=list(LORA_REGISTRY.keys()), default=[])
    args = parser.parse_args()

    load_datasets()

    if args.preload:
        load_base_model()
        for s in args.preload:
            load_adapter(s)

    app.run(host="0.0.0.0", port=args.port, debug=False)
