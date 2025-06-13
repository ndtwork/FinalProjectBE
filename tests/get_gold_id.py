import os
import json
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL_NAME", "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "1000v100"   # Đổi tên collection vật lý nếu muốn

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu", "trust_remote_code": True}
)

project_root = Path(__file__).resolve().parents[1]
EVAL_FILE = project_root / "tests" / "eval_questions.jsonl"
samples = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines()]

for idx, s in enumerate(samples, 1):
    q = s["question"]
    qvec = embedder.embed_query(q)
    # Lấy top 5 chunk Regulation
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=5,
        query_filter={
            "must": [
                {"key": "metadata.document_type", "match": {"value": "Regulation"}}
            ]
        }
    )
    print(f"\nQ{idx}: {q}")
    if not hits:
        print("   No result!")
    for i, chunk in enumerate(hits):
        cid = getattr(chunk, "id", None)
        content = chunk.payload.get("content", "")[:70].replace("\n", " ")
        print(f"   {i+1}. id: {cid}  |  content: {content}")
