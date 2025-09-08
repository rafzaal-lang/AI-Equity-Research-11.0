import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "edgar_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

_client = QdrantClient(url=QDRANT_URL)
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ensure_collection():
    try:
        _client.get_collection(QDRANT_COLLECTION)
    except Exception:
        _client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )

def upsert_points(points: List[Dict[str, Any]]):
    ensure_collection()
    qdrant_points = []
    for i, p in enumerate(points):
        qdrant_points.append(PointStruct(
            id=i,
            vector=p["vector"],
            payload=p["payload"]
        ))
    _client.upsert(collection_name=QDRANT_COLLECTION, points=qdrant_points)

def search(query_vector: List[float], k: int = 10, tickers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    ensure_collection()
    filter_conditions = None
    if tickers:
        filter_conditions = {"must": [{"key": "ticker", "match": {"any": tickers}}]}
    
    results = _client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=k,
        query_filter=filter_conditions
    )
    
    return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

def embed_text(text: str) -> List[float]:
    response = _openai.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding

