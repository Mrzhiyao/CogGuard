import sys
import asyncio
import json
from pathlib import Path
import argparse

# 添加 student_sim: https://github.com/McCartney01/student_sim
STUDENT_SIM_PATH = "./student_sim-main"
if STUDENT_SIM_PATH not in sys.path:
    sys.path.insert(0, STUDENT_SIM_PATH)

try:
    from src.nano_graphrag import GraphRAG
    from src.nano_graphrag._llm import openai_embedding
    from src.nano_graphrag._utils import wrap_embedding_func_with_attrs
    from src.utils import generate
    from src.mapping import MODEL_FUNC_MAP
    import networkx as nx
except ImportError as e:
    print(f"错误: 无法导入 student_sim 模块: {e}")
    sys.exit(1)

# 20个学生的ID
# STUDENT_IDS_20 = [
#     "134015446", "135363688", "156242714", "190904875", "303035807",
#     "451443795", "138766644", "172255240", "174144880", "190778770",
#     "216337872", "235541946", "271819945", "291946738", "292852703",
#     "314863369", "339292326", "380644542", "405568650", "427624027"
# ]

# STUDENT_IDS_30 = [
#     "134015446", "135363688", "156242714", "190904875", "303035807",
#     "451443795", "138766644", "172255240", "174144880", "190778770",
#     "216337872", "235541946", "271819945", "291946738", "292852703",
#     "314863369", "339292326", "380644542", "405568650", "427624027"
# ]

STUDENT_IDS_40 = [
    "57706049", "64904597", "69501469", "86094393", "86151237", "134015446", 
    "135363688", "138766644", "156242714", "172255240", "174144880", "190778770", 
    "190904875", "216337872", "235541946", "271819945", "291946738", "292852703", 
    "303035807", "314863369", "339292326", "380644542", "405568650", "427624027", 
    "451443795", "463554746", "466662879", "604128734", "626294389", "662511357", 
    "679796740", "756999517", "767155168", "771346787", "783928287", "800795936", 
    "822141656", "831427858", "843247902", "915205165"
]


def create_custom_embedding_func(api_key: str, base_url: str, model: str = "conan", embedding_dim: int = 1792):
    import httpx
    import numpy as np
    
    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
    async def custom_embedding(texts: list[str]) -> np.ndarray:
        """自定义 embedding 函数"""
        if not texts:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        
        try:
            payload = {"model": model, "input": texts}
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
            
            data = result.get("data", [])
            if not data:
                return np.zeros((0, embedding_dim), dtype=np.float32)
            
            vectors = [item["embedding"] for item in data]
            return np.asarray(vectors, dtype=np.float32)
        except Exception as e:
            print(f"Embedding 调用失败: {e}")
            raise
    
    return custom_embedding


def extract_entities_from_graph(graph_file: Path) -> list[dict]:
    """从图结构中提取所有实体"""
    if not graph_file.exists():
        return []
    
    try:
        g = nx.read_graphml(str(graph_file))
        entities = []
        
        for node_id, node_data in g.nodes(data=True):
            entity_name = node_id.strip('"')  # 移除引号
            entity_type = node_data.get('entity_type', '').strip('"')
            description = node_data.get('description', '').strip('"')
            
            if entity_name:
                entities.append({
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": description
                })
        
        return entities
    except Exception as e:
        print(f"  错误: 读取图文件失败: {e}")
        return []


def filter_train_entities(entities: list[dict], train_data: list[dict], 
                         test_problem_names: set, validation_problem_names: set) -> list[dict]:
    """过滤实体，只保留来自训练数据的实体"""
    if not test_problem_names and not validation_problem_names:
        return entities
    
    train_problem_names = set()
    for item in train_data:
        problem_name = item.get('problem_name', '')
        if problem_name:
            train_problem_names.add(problem_name)
    
    filtered_entities = []
    for entity in entities:
        # 检查实体的描述中是否包含测试集/验证集的问题名称
        description = entity.get('description', '').lower()
        entity_name = entity.get('entity_name', '').lower()
        content = (entity_name + ' ' + description).lower()
        
        # 如果包含测试集或验证集的问题名称，跳过
        is_from_test_or_val = False
        for problem_name in test_problem_names:
            if problem_name and problem_name.lower() in content:
                is_from_test_or_val = True
                break
        if not is_from_test_or_val:
            for problem_name in validation_problem_names:
                if problem_name and problem_name.lower() in content:
                    is_from_test_or_val = True
                    break
        
        if not is_from_test_or_val:
            filtered_entities.append(entity)
    
    return filtered_entities


async def fix_vdb_for_student(
    student_id: str,
    kg_dir: Path,
    embedding_func,
    model_func,
    train_data: list[dict] = None,
    test_problem_names: set = None,
    validation_problem_names: set = None,
    filter_train_only: bool = True,
    dry_run: bool = False
):
    """为单个学生修复向量数据库"""
    print(f"\n处理学生: {student_id}")
    print(f"  知识图谱目录: {kg_dir}")
    
    if not kg_dir.exists():
        print(f"知识图谱目录不存在，跳过")
        return False
    
    # 检查向量数据库文件是否已存在
    vdb_file = kg_dir / "vdb_entities.json"
    if vdb_file.exists() and not dry_run:
        print(f"向量数据库文件已存在，跳过（如需重新生成，请先删除该文件）")
        return True
    
    # 读取图结构
    graph_file = kg_dir / "graph_chunk_entity_relation.graphml"
    entities = extract_entities_from_graph(graph_file)
    
    if not entities:
        print(f"未找到实体，跳过")
        return False
    
    print(f"从图结构中提取了 {len(entities)} 个实体")
    
    # 过滤实体（只保留训练数据）
    if filter_train_only and train_data:
        original_count = len(entities)
        entities = filter_train_entities(
            entities, 
            train_data, 
            test_problem_names or set(), 
            validation_problem_names or set()
        )
        filtered_count = original_count - len(entities)
        print(f"过滤后保留 {len(entities)} 个训练数据实体（过滤掉 {filtered_count} 个验证集/测试集实体）")
    elif filter_train_only:
        print(f"警告: 启用了过滤但未提供训练数据，将包含所有实体（可能包含验证集/测试集数据）")
    
    if dry_run:
        print(f"[DRY RUN] 将生成 {len(entities)} 个实体的embeddings并保存到向量数据库")
        return True
    
    # 初始化 GraphRAG（只用于保存向量数据库）
    try:
        graph = GraphRAG(
            working_dir=str(kg_dir),
            best_model_func=model_func,
            cheap_model_func=model_func,
            embedding_func=embedding_func
        )
        
        from src.nano_graphrag._op import compute_mdhash_id
        
        data_for_vdb = {
            compute_mdhash_id(entity["entity_name"], prefix="ent-"): {
                "content": entity["entity_name"] + " " + entity["description"],
                "entity_name": entity["entity_name"],
            }
            for entity in entities
        }
        
        print(f"  正在生成embeddings并插入向量数据库...")
        print(f"  （这可能需要一些时间，取决于实体数量和embedding API速度）")
        
        # 插入向量数据库
        await graph.entities_vdb.upsert(data_for_vdb)
        
        # 保存向量数据库
        await graph.entities_vdb.index_done_callback()
        
        print(f"  ✓ 向量数据库已保存: {vdb_file}")
        return True
        
    except Exception as e:
        print(f"  ✗ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description='修复知识图谱的向量数据库文件')
    parser.add_argument('--kg_base_dir', type=str,
                        default='EI/EI_without_cluster',
                        help='知识图谱基础目录')
    parser.add_argument('--student_ids', type=str, nargs='+', default=None,
                        help='要修复的学生ID列表（默认：所有20个学生）')
    parser.add_argument('--input_file', type=str, default=None,
                        help='训练数据文件（用于过滤验证集/测试集实体）。推荐使用 training set/train_processed_20_new_no_embedding.jsonl')
    parser.add_argument('--validation_file', type=str, default=None,
                        help='验证集文件（用于提取问题名称）')
    parser.add_argument('--test_file', type=str, default=None,
                        help='测试集文件（用于提取问题名称）')
    parser.add_argument('--no_filter', action='store_true',
                        help='不过滤，包含所有实体（会引入数据泄露风险，不推荐）')
    parser.add_argument('--embedding_api_key', type=str, default=None,
                        help='Embedding API key')
    parser.add_argument('--embedding_base_url', type=str, default=None,
                        help='Embedding API base URL')
    parser.add_argument('--embedding_model', type=str, default='conan',
                        help='Embedding 模型名称（默认 conan）')
    parser.add_argument('--embedding_dim', type=int, default=1792,
                        help='Embedding 维度（默认 1792）')
    parser.add_argument('--dry_run', action='store_true',
                        help='只检查，不实际修复')
    
    args = parser.parse_args()
    
    # 处理过滤选项（默认启用过滤）
    filter_train_only = not args.no_filter
    
    # 确定要处理的学生
    student_ids = args.student_ids if args.student_ids else STUDENT_IDS_20
    
    # 创建embedding函数
    if args.embedding_api_key and args.embedding_base_url:
        embedding_func = create_custom_embedding_func(
            api_key=args.embedding_api_key,
            base_url=args.embedding_base_url,
            model=args.embedding_model,
            embedding_dim=args.embedding_dim
        )
        print(f"使用自定义 Embedding API: {args.embedding_base_url}")
    else:
        embedding_func = openai_embedding
        print("使用默认 OpenAI Embedding API（需要设置 OPENAI_API_KEY 环境变量）")
    
    # 加载训练数据和问题名称（用于过滤）
    train_data_dict = {}
    test_problem_names = set()
    validation_problem_names = set()
    
    if filter_train_only:
        if args.input_file:
            import json
            print(f"加载训练数据: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                all_data = [json.loads(line) for line in f if line.strip()]
            # 按学生组织数据
            for item in all_data:
                student_id = item.get('user_id')
                if student_id in student_ids:
                    if student_id not in train_data_dict:
                        train_data_dict[student_id] = []
                    train_data_dict[student_id].append(item)
            print(f"加载了 {len(all_data)} 条训练数据")
        else:
            print(f"启用了过滤但未提供 --input_file，将无法过滤验证集/测试集实体")
        
        if args.validation_file:
            import json
            print(f"加载验证集问题名称: {args.validation_file}")
            with open(args.validation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        problem_name = item.get('problem_name', '')
                        if problem_name:
                            validation_problem_names.add(problem_name)
            print(f"  提取了 {len(validation_problem_names)} 个验证集问题名称")
        
        if args.test_file:
            import json
            print(f"加载测试集问题名称: {args.test_file}")
            with open(args.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        problem_name = item.get('problem_name', '')
                        if problem_name:
                            test_problem_names.add(problem_name)
            print(f"  提取了 {len(test_problem_names)} 个测试集问题名称")
    
    # 创建模型函数（用于GraphRAG初始化，实际不会调用）
    model_func = MODEL_FUNC_MAP.get("3.5", generate)
    
    # 处理每个学生
    kg_base_dir = Path(args.kg_base_dir)
    success_count = 0
    total_entities = 0
    
    for student_id in student_ids:
        kg_dir = kg_base_dir / f"graph_{student_id}_practice" / "model_General"
        
        # 获取该学生的训练数据
        student_train_data = train_data_dict.get(student_id, [])
        
        success = await fix_vdb_for_student(
            student_id=student_id,
            kg_dir=kg_dir,
            embedding_func=embedding_func,
            model_func=model_func,
            train_data=student_train_data,
            test_problem_names=test_problem_names,
            validation_problem_names=validation_problem_names,
            filter_train_only=filter_train_only,
            dry_run=args.dry_run
        )
        
        # 统计实体数量（用于估算时间）
        if success or args.dry_run:
            graph_file = kg_dir / "graph_chunk_entity_relation.graphml"
            entities = extract_entities_from_graph(graph_file)
            if filter_train_only and student_train_data:
                entities = filter_train_entities(
                    entities, 
                    student_train_data, 
                    test_problem_names, 
                    validation_problem_names
                )
            total_entities += len(entities)
        
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"修复完成！")
    print(f"成功: {success_count}/{len(student_ids)}")
    print(f"总实体数: {total_entities}")
    
    if not args.dry_run:
        estimated_time = total_entities * 0.1 / 60  # 分钟
        print(f"估算耗时: 约 {estimated_time:.1f} 分钟（取决于embedding API速度）")
    else:
        print(f"[DRY RUN] 未实际执行修复")


if __name__ == '__main__':
    asyncio.run(main())

