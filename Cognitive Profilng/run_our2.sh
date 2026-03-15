#!/bin/bash

# 基础配置
EDU_DATA_DIR="/edu_data"
SCRIPT_DIR="/double_graph"
PYTHON_SCRIPT="prepare_our_data2.py"
OUTPUT_DIR="/our/"

# 切换到脚本目录
cd "$SCRIPT_DIR" || exit 1

echo "=========================================="
echo "补全数据集"
echo "=========================================="
echo ""

echo "查找之前已构建的学生..."
TRAIN_STUDENTS_FILE="/tmp/train_students_$(date +%s).txt"
find "$OUTPUT_DIR" -name "train.jsonl" -type f | \
    xargs -I {} sh -c 'd=$(dirname {}); d=$(dirname $d); echo $(basename $d) | sed "s/student_//;s/_practice.*//"' | \
    sort -u > "$TRAIN_STUDENTS_FILE"

TRAIN_STUDENT_COUNT=$(wc -l < "$TRAIN_STUDENTS_FILE")
echo "找到 $TRAIN_STUDENT_COUNT 个已完成的学生"
echo ""

if [ "$TRAIN_STUDENT_COUNT" -eq 0 ]; then
    echo "错误: 未找到之前已完成的学生，请先运行 run_our1.sh 构建大部分学生的知识图"
    exit 1
fi

# 遍历 edu_data 目录下的所有 CSV 文件，但只处理已构建知识图的学生
# 注意：使用进程替换来避免子shell问题
PROCESSED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

while IFS= read -r csv_file; do
    # 提取用户ID
    csv_basename=$(basename "$csv_file" .csv)
    user_id=$(echo "$csv_basename" | sed 's/_practice$//')
    
    # 检查该用户是否在训练集中
    if ! grep -q "^${user_id}$" "$TRAIN_STUDENTS_FILE"; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        if [ $((SKIPPED_COUNT % 10)) -eq 0 ]; then
            echo "跳过: 用户 $user_id 未构建过 (已跳过 $SKIPPED_COUNT 个)"
        fi
        continue
    fi
    
    # 检查该用户是否已有知识图
    user_kg_dir="$OUTPUT_DIR/knowledge_point_graph_${user_id}_practice/model_General"
    if [ ! -d "$user_kg_dir" ]; then
        echo "跳过: 用户 $user_id 的知识图目录不存在（需要先构建知识图）"
        continue
    fi
    
    # 【新增】检查测试集知识图是否已经构建完成
    test_file="$OUTPUT_DIR/student_${user_id}_practice/model_General/test.jsonl"
    kg_progress_file="$user_kg_dir/knowledge_point_progress.json"
    
    if [ -f "$test_file" ] && [ -f "$kg_progress_file" ]; then
        # 使用Python检查测试集时间戳是否在知识图范围内
        check_result=$(python3 << PYEOF
import json
import sys
from datetime import datetime

try:
    # 读取测试集最新时间戳
    with open("$test_file") as f:
        test_timestamps = [json.loads(line).get('timestamp', '') for line in f if line.strip()]
    test_timestamps = [ts for ts in test_timestamps if ts]
    
    if not test_timestamps:
        print("NEEDS_BUILD")
        sys.exit(0)
    
    test_timestamps.sort()
    latest_test_time = datetime.strptime(test_timestamps[-1], "%Y-%m-%d %H:%M:%S")
    
    # 读取知识图最新时间戳
    with open("$kg_progress_file") as f:
        kg_data = json.load(f)
    summaries = kg_data.get('knowledge_summaries', {})
    kg_timestamps = [key.split('_record_')[0] for key in summaries.keys() if '_record_' in key]
    kg_timestamps.sort()
    
    if not kg_timestamps:
        print("NEEDS_BUILD")
        sys.exit(0)
    
    latest_kg_time = datetime.strptime(kg_timestamps[-1], "%Y-%m-%d %H:%M:%S")
    
    # 如果测试集最新时间戳在知识图范围内，说明已经构建完成
    if latest_test_time <= latest_kg_time:
        print("ALREADY_DONE")
    else:
        print("NEEDS_BUILD")
except Exception as e:
    print("NEEDS_BUILD")
PYEOF
)
        
        if [ "$check_result" = "ALREADY_DONE" ]; then
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            echo "跳过: 用户 $user_id 的测试集知识图已构建完成 ($SKIPPED_COUNT/$TRAIN_STUDENT_COUNT 已跳过)"
            continue
        fi
    fi
    
    echo "=========================================="
    echo "处理 CSV 文件: $csv_file"
    echo "用户ID: $user_id (已有)"
    echo "=========================================="
    
    python "$PYTHON_SCRIPT" \
        --csv_path "$csv_file" \
        --knowledge_points_file /edu_data/knowledge_points.md \
        --output_dir /our/our_with_cluster \
        --global_entity_graph_dir /our/entity_graph_global \
        --llm_model General \
        --llm_api_key " " \
        --llm_base_url  \
        --pg_user  \
        --pg_password  \
        --pg_database  \
        --temporal_mode \
        --llm_timeout 240 \
        --snapshot_interval 1 \
        --insert_timeout 1800
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
        echo "✓ 成功处理: $csv_file (已处理 $PROCESSED_COUNT 个)"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        echo "✗ 处理失败: $csv_file (失败 $FAILED_COUNT 个)"
        echo "继续处理下一个文件..."
    fi
    
    echo ""
done < <(find "$EDU_DATA_DIR" -type f -name "*_practice.csv")

echo "=========================================="
echo "处理完成！"
echo "统计信息:"
echo "  - 成功处理: $PROCESSED_COUNT 个学生"
echo "  - 跳过（已完成）: $SKIPPED_COUNT 个学生"
echo "  - 处理失败: $FAILED_COUNT 个学生"
echo "  - 已构建知识图的学生总数: $TRAIN_STUDENT_COUNT 个"
echo "=========================================="

# 清理临时文件
rm -f "$TRAIN_STUDENTS_FILE"



