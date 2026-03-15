BASE_DIR="./our/our_without_cluster"

# 检查是否使用 --force 参数
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

# 共享的合并文件（所有模型共用）
MERGED_FILE="./datasets/all_merged_40_new.jsonl"

echo "=========================================="
echo "为多个模型构建完整数据集"
echo "=========================================="
echo ""

# 步骤 1: 检查并生成 all.jsonl 文件（如果还没有）
echo "步骤 1: 检查 all.jsonl 文件..."
ALL_JSONL_COUNT=$(find "$BASE_DIR" -name "all.jsonl" | wc -l)
if [ "$ALL_JSONL_COUNT" -eq 0 ]; then
    echo "未找到 all.jsonl 文件，开始合并 train.jsonl 和 test.jsonl..."
    python merge_train_test.py --base_dir "$BASE_DIR"
    if [ $? -ne 0 ]; then
        echo "错误: 合并 train 和 test 文件失败"
        exit 1
    fi
    echo "✓ 合并完成"
else
    echo "✓ 已找到 $ALL_JSONL_COUNT 个 all.jsonl 文件"
fi
echo ""

# 步骤 2: 生成 all_merged_40_new.jsonl（如果还没有）
# 优先使用 all_merged_40_new.jsonl（去重后的新文件）
if [ -f "$MERGED_FILE" ]; then
    echo "步骤 2: all_merged_40_new.jsonl 已存在，使用: $MERGED_FILE"
else
    echo "步骤 2: 生成 all_merged_40_new.jsonl（补全知识图）..."
    python merge_and_enrich_data.py \
        --base_dir "$BASE_DIR" \
        --output_file "$MERGED_FILE" \
        --data_type all
    
    if [ $? -ne 0 ]; then
        echo "错误: 合并和补全数据失败"
        exit 1
    fi
    echo "✓ 生成完成: $MERGED_FILE"
fi
echo ""

# 步骤 3: 为每个模型生成 tokenized 数据
echo "步骤 3: 为每个模型生成 tokenized 数据..."
echo "=========================================="
echo ""

MODEL_COUNT=${#MODELS[@]}
CURRENT_MODEL=0

for MODEL_DIR in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_DIR]}"
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    
    OUTPUT_FILE="./all_processed_40_new_${MODEL_NAME}.jsonl"
    
    echo "[$CURRENT_MODEL/$MODEL_COUNT] 处理模型: $MODEL_NAME"
    echo "  模型路径: $MODEL_DIR"
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
    
    # 生成 tokenized 数据
    echo "  开始 tokenize..."
    python prepare_dataset_for_training.py \
        --input_file "$MERGED_FILE" \
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

echo "=========================================="
echo "处理完成！"
echo "=========================================="
echo ""
echo "生成的文件："
echo "  合并文件（所有模型共用）: $MERGED_FILE"
for MODEL_DIR in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_DIR]}"
    OUTPUT_FILE="./all_processed_40_new_${MODEL_NAME}.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  ✓ $OUTPUT_FILE"
    else
        echo "  ✗ $OUTPUT_FILE (未生成)"
    fi
done
echo ""

