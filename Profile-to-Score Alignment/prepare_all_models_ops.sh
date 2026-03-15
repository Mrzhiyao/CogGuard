FORCE_OVERWRITE=false
if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
    FORCE_OVERWRITE=true
    echo "使用 --force 模式：自动覆盖已存在的文件"
    echo ""
fi

# 模型列表（模型路径 -> 模型名称）
declare -A MODELS=(
    ["./Qwen3-0.6B"]="qwen3_06b"
    ["./Qwen3-1.7B"]="qwen3_17b"
    ["./gemma-2-2b-it"]="gemma2_2b"
    ["./Qwen2.5-3B-Instruct"]="qwen25_3b"
    ["./Qwen3-8B"]="qwen3_8b"
    ["./Meta-Llama-3.1-8B-Instruct"]="llama3_1_8b"
    ["./gemma-2-9b-it"]="gemma2_9b"
)


INPUT_FILE="./ops_dataset.jsonl"

echo "=========================================="
echo "为多个模型构建 OPS 场景数据集"
echo "=========================================="
echo ""

# 步骤 1: 检查输入文件是否存在
echo "步骤 1: 检查 OPS 数据集文件..."
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: OPS 数据集文件不存在: $INPUT_FILE"
    exit 1
fi

FILE_SIZE=$(du -h "$INPUT_FILE" | cut -f1)
LINE_COUNT=$(wc -l < "$INPUT_FILE")
echo "✓ 找到 OPS 数据集文件: $INPUT_FILE"
echo "  大小: $FILE_SIZE"
echo "  记录数: $LINE_COUNT 条"
echo ""

# 步骤 2: 为每个模型生成 tokenized 数据
echo "步骤 2: 为每个模型生成 tokenized 数据..."
echo "=========================================="
echo ""

MODEL_COUNT=${#MODELS[@]}
CURRENT_MODEL=0

for MODEL_DIR in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_DIR]}"
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    
    OUTPUT_FILE="./datasets/ops_processed_${MODEL_NAME}.jsonl"
    
    echo "[$CURRENT_MODEL/$MODEL_COUNT] 处理模型: $MODEL_NAME"
    echo "  模型路径: $MODEL_DIR"
    echo "  输入文件: $INPUT_FILE"
    echo "  输出文件: $OUTPUT_FILE"
    
    # 检查模型目录是否存在
    if [ ! -d "$MODEL_DIR" ]; then
        echo "   警告: 模型目录不存在，跳过: $MODEL_DIR"
        echo ""
        continue
    fi
    
    # 检查是否已存在（可选：如果已存在则跳过）
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        if [ "$FORCE_OVERWRITE" = true ]; then
            echo "   文件已存在 ($FILE_SIZE)，将覆盖（--force 模式）"
        else
            echo "   文件已存在 ($FILE_SIZE)，是否覆盖？(y/n，默认n)"
            read -t 10 -n 1 answer || answer="n"
            echo ""
            if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
                echo "  跳过: $OUTPUT_FILE"
                echo ""
                continue
            fi
        fi
        echo "  将覆盖现有文件..."
    fi
    
    python prepare_ops_dataset_for_training.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --model_dir "$MODEL_DIR" \
        --discretize
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 成功: $OUTPUT_FILE"
        
        # 显示文件信息
        if [ -f "$OUTPUT_FILE" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
            echo "    大小: $FILE_SIZE"
            echo "    记录数: $LINE_COUNT 条"
        fi
    else
        echo "  ✗ 失败: $OUTPUT_FILE"
    fi
    
    echo ""
done


for MODEL_DIR in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_DIR]}"
    OUTPUT_FILE="./datasets/ops_processed_${MODEL_NAME}.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  ✓ $OUTPUT_FILE"
    else
        echo "  ✗ $OUTPUT_FILE (未生成)"
    fi
done
echo ""

