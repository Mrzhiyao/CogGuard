from typing import List, Optional

import httpx
import numpy as np

from nano_graphrag._utils import wrap_embedding_func_with_attrs

def _resolve_base_url(base_url: str) -> str:
    if not base_url:
        raise ValueError("Embedding service base_url 不能为空")
    return base_url.rstrip("/")


def _build_headers(api_key: Optional[str]) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def create_conan_embedding_func(
    base_url: str,
    api_key: Optional[str],
    model_name: str = "conan",
    embedding_dim: int = 1792,
    timeout: float = 120.0,
):

    base_url = _resolve_base_url(base_url)
    endpoint = f"{base_url}/v1/embeddings"
    headers = _build_headers(api_key)

    async def _embedding(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        payload = {"model": model_name, "input": texts}
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data", [])
        if not data:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        vectors = [item["embedding"] for item in data]
        return np.asarray(vectors, dtype=np.float32)

    return wrap_embedding_func_with_attrs(
        embedding_dim=embedding_dim,
        max_token_size=8192,
    )(_embedding)

