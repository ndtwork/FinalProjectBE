from qdrant_client import QdrantClient
import json
import os

# ======== SỬA ĐÚNG THÔNG SỐ NÀY =========
QDRANT_URL = os.getenv("QDRANT_URL", "https://43965d09-2062-4e87-9d29-fed11a204a3c.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bdx9pR6wJycDyXQu6f4Al7xAU3wSiozwLjaDV0FyiLY")

# CÓ THỂ ĐỔI HOẶC LẶP QUA NHIỀU COLLECTION
COLLECTIONS = [
    "1000v100",
    "800v400",
    "800v200",
    "400v200"
]

def export_collection(client, collection_name, out_dir="qdrant_export"):
    print(f"== Đang export collection: {collection_name} ==")
    out_path = f"{out_dir}/{collection_name}.jsonl"
    os.makedirs(out_dir, exist_ok=True)

    all_points = []
    offset = None  # Qdrant mới dùng "offset=None" hoặc "offset=last_point_id"
    page_size = 256  # Tùy dung lượng RAM/Qdrant, nên <500

    total = 0
    while True:
        res, next_page = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=page_size,
            with_payload=True,
            with_vectors=False
        )
        if not res:
            break
        for pt in res:
            all_points.append({
                "id": pt.id,
                "content": pt.payload.get("content"),
                "doc_type": pt.payload.get("metadata", {}).get("document_type"),
                "source_file": pt.payload.get("metadata", {}).get("source_file")
            })
        total += len(res)
        print(f"  ... Lấy {total} điểm")
        if not next_page:
            break
        offset = next_page

    with open(out_path, "w", encoding="utf-8") as f:
        for pt in all_points:
            f.write(json.dumps(pt, ensure_ascii=False) + "\n")
    print(f"=> Xuất {total} chunk ra {out_path}\n")

if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    for col in COLLECTIONS:
        export_collection(client, col)
