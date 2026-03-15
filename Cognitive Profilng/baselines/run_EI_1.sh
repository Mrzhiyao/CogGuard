#!/bin/bash

# 基础配置
EDU_DATA_DIR="/edu_data"
SCRIPT_DIR="/baselines/double_graph"
PYTHON_SCRIPT="prepare_EI_data1.py"
EXISTING_STUDENTS_DIR="$SCRIPT_DIR/EI"

# 切换到脚本目录
cd "$SCRIPT_DIR" || exit 1

# 获取已存在的学生ID列表（从 student_XXXXX_practice 目录中提取）
echo "正在扫描已存在的学生目录..."
EXISTING_STUDENTS_FILE=$(mktemp)
for student_dir in "$EXISTING_STUDENTS_DIR"/student_*_practice; do
    if [ -d "$student_dir" ]; then
        student_basename=$(basename "$student_dir")
        # 提取学生ID（student_XXXXX_practice -> XXXXX）
        student_id=$(echo "$student_basename" | sed 's/^student_//' | sed 's/_practice$//')
        echo "$student_id" >> "$EXISTING_STUDENTS_FILE"
    fi
done

EXISTING_COUNT=$(wc -l < "$EXISTING_STUDENTS_FILE" | tr -d ' ')
echo "找到 $EXISTING_COUNT 个已存在的学生目录"

# 遍历 edu_data 目录下的所有 CSV 文件（包括子目录）
find "$EDU_DATA_DIR" -type f -name "*.csv" | while read -r csv_file; do
    # 提取 CSV 文件的基础名称（去掉路径和扩展名）
    csv_basename=$(basename "$csv_file" .csv)
    
    # 检查是否是已存在的学生（格式：XXXXX_practice 或 其他格式）
    # 尝试提取学生ID
    student_id=""
    if [[ "$csv_basename" =~ ^([0-9]+)_practice$ ]]; then
        student_id="${BASH_REMATCH[1]}"
    elif [[ "$csv_basename" =~ ^([0-9]+)$ ]]; then
        student_id="${BASH_REMATCH[1]}"
    else
        # 尝试从文件名中提取数字部分作为学生ID
        student_id=$(echo "$csv_basename" | grep -oE '[0-9]+' | head -1)
    fi
    
    # 检查该学生是否在已存在列表中
    if [ -z "$student_id" ] || ! grep -q "^${student_id}$" "$EXISTING_STUDENTS_FILE"; then
        echo "跳过: $csv_file (学生ID: ${student_id:-未知}, 不在已存在列表中)"
        continue
    fi
    
    echo "=========================================="
    echo "处理 CSV 文件: $csv_file"
    echo "学生ID: $student_id"
    echo "=========================================="
    
    python3 "$PYTHON_SCRIPT" \
        --csv_path "$csv_file" \
        --output_dir EI/EI_without_cluster \
        --llm_model General \
        --llm_api_key " " \
        --llm_base_url "" \
        --pg_user  \
        --pg_password  \
        --pg_database  \
        --temporal_mode \
        --llm_timeout 240 \
        --snapshot_interval 1 \
        --insert_timeout 1800
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✓ 成功处理: $csv_file"
    else
        echo "✗ 处理失败: $csv_file"
        echo "继续处理下一个文件..."
    fi
    
    echo ""
done

echo "=========================================="
echo "所有 CSV 文件处理完成！"
echo "=========================================="

# 清理临时文件
rm -f "$EXISTING_STUDENTS_FILE"

