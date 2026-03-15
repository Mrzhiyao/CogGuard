#!/bin/bash

# 基础配置
EDU_DATA_DIR="/edu_data"
SCRIPT_DIR="/double_graph"
PYTHON_SCRIPT="prepare_our_data1.py"


# 切换到脚本目录
cd "$SCRIPT_DIR" || exit 1

# 遍历 edu_data 目录下的所有 CSV 文件（包括子目录）
find "$EDU_DATA_DIR" -type f -name "*.csv" | while read -r csv_file; do
    echo "=========================================="
    echo "处理 CSV 文件: $csv_file"
    echo "=========================================="
    
    # 提取 CSV 文件的相对路径（用于输出目录命名）
    csv_rel_path="${csv_file#$EDU_DATA_DIR/}"
    csv_dir=$(dirname "$csv_rel_path")
    csv_basename=$(basename "$csv_file" .csv)
    
    python "$PYTHON_SCRIPT" \
        --csv_path "$csv_file" \
        --knowledge_points_file /edu_data/knowledge_points.md \
        --output_dir our/our_without_cluster \
        --global_entity_graph_dir our/entity_graph_global \
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






