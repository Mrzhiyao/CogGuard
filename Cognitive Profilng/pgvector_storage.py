import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import importlib
import numpy as np

if TYPE_CHECKING:
    import psycopg  # type: ignore[import]  # pragma: no cover
    from psycopg import sql  # type: ignore[import]  # pragma: no cover
    from psycopg.types.json import Json  # type: ignore[import]  # pragma: no cover
else:
    try:
        psycopg = importlib.import_module("psycopg")
        sql = importlib.import_module("psycopg.sql")
        Json = importlib.import_module("psycopg.types.json").Json
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "未找到 psycopg 库"
        ) from exc

from nano_graphrag._utils import logger
from nano_graphrag.base import BaseVectorStorage


def _filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _vector_literal(embedding: np.ndarray) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding.tolist()) + "]"


@dataclass
class PGVectorStorage(BaseVectorStorage):
    table_name: Optional[str] = None
    connection_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        params = dict(self.global_config.get("vector_db_storage_cls_kwargs", {}))
        params.update(self.connection_kwargs)

        self._table_name = params.get("table_name") or f"{self.namespace}_embeddings"
        self._embedding_dim = params.get("embedding_dim", self.embedding_func.embedding_dim)
        self._batch_size = self.global_config.get("embedding_batch_num", 32)

        dsn = params.get("dsn")
        connect_args = params.get("connection") or {}
        if dsn:
            self._connect_kwargs = {"conninfo": dsn}
        else:
            connect_params = {
                "host": params.get("host"),
                "port": params.get("port", 5432),
                "dbname": params.get("dbname"),
                "user": params.get("user"),
                "password": params.get("password"),
            }
            connect_params.update(connect_args)
            self._connect_kwargs = _filter_none_values(connect_params)

        if not self._connect_kwargs:
            raise ValueError("PGVectorStorage 缺少数据库连接配置")

        self._ensure_table()

    def _connect(self):
        return psycopg.connect(**self._connect_kwargs)

    def _ensure_table(self):
        with self._connect() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                create_sql = sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        embedding vector({dim}),
                        entity_name TEXT,
                        content TEXT,
                        meta JSONB
                    );
                    """
                ).format(
                    table=sql.Identifier(self._table_name),
                    dim=sql.Literal(int(self._embedding_dim)),
                )
                cur.execute(create_sql)

    async def upsert(self, data: Dict[str, Dict]):
        if not data:
            logger.warning("PGVectorStorage 收到空数据，跳过 upsert")
            return []

        list_data = [
            {
                "__id__": key,
                **{field: value for field, value in value_dict.items() if field in self.meta_fields},
            }
            for key, value_dict in data.items()
        ]

        contents = [value["content"] for value in data.values()]
        batches = [
            contents[i : i + self._batch_size]
            for i in range(0, len(contents), self._batch_size)
        ]

        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        await asyncio.get_event_loop().run_in_executor(
            None,
            self._upsert_sync,
            list_data,
            embeddings,
            list(data.items()),
        )

    def _upsert_sync(
        self,
        list_data: list[Dict[str, Any]],
        embeddings: np.ndarray,
        raw_items: list[tuple[str, Dict[str, Any]]],
    ):
        insert_sql = sql.SQL(
            """
            INSERT INTO {table} (id, embedding, entity_name, content, meta)
            VALUES (%(id)s, %(embedding)s::vector, %(entity_name)s, %(content)s, %(meta)s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                entity_name = EXCLUDED.entity_name,
                content = EXCLUDED.content,
                meta = EXCLUDED.meta;
            """
        ).format(table=sql.Identifier(self._table_name))

        with self._connect() as conn:
            with conn.cursor() as cur:
                for idx, item in enumerate(list_data):
                    raw_key, raw_value = raw_items[idx]
                    embedding_literal = _vector_literal(embeddings[idx])
                    meta = {field: raw_value.get(field) for field in self.meta_fields}
                    cur.execute(
                        insert_sql,
                        {
                            "id": item["__id__"],
                            "embedding": embedding_literal,
                            "entity_name": item.get("entity_name"),
                            "content": raw_value.get("content", ""),
                            "meta": Json(meta or {}),
                        },
                    )
            conn.commit()

    async def query(self, query: str, top_k: int):
        if top_k <= 0:
            return []

        embedding = await self.embedding_func([query])
        vector_literal = _vector_literal(embedding[0])

        return await asyncio.get_event_loop().run_in_executor(
            None, self._query_sync, vector_literal, top_k
        )

    def _query_sync(self, vector_literal: str, top_k: int):
        query_sql = sql.SQL(
            """
            SELECT id,
                   entity_name,
                   content,
                   meta,
                   embedding <#> %(embedding)s::vector AS distance
            FROM {table}
            ORDER BY distance ASC
            LIMIT %(limit)s;
            """
        ).format(table=sql.Identifier(self._table_name))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query_sql,
                    {
                        "embedding": vector_literal,
                        "limit": top_k,
                    },
                )
                rows = cur.fetchall()

        results = []
        for row in rows:
            id_value, entity_name, content, meta, distance = row
            record = {
                "id": id_value,
                "entity_name": entity_name,
                "content": content,
                "meta": meta,
                "distance": float(distance) if distance is not None else None,
            }
            results.append(record)
        return results

