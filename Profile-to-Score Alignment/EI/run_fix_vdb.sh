
# 图谱的向量数据库文件

python fix_vdb.py \
    --input_file ./datasets/students_20_new/training\ set/train_processed_20_new_no_embedding.jsonl \
    --validation_file ./datasets/students_20_new/validation\ set/validation_processed_20_new_no_embedding.jsonl \
    --test_file ./datasets/students_20_new/test\ set/test_processed_20_new_no_embedding.jsonl \
    --embedding_api_key "" \
    --embedding_base_url  \
    --embedding_model conan \
    --embedding_dim 1792

