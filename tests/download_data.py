# tests/download_data.py

import os
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient

def download_documents_from_qdrant(collection_name: str):
    """
    Lấy tất cả point kèm payload từ Qdrant và in ra.
    Đặc biệt dành cho qdrant-client 1.14.2,
    nơi scroll(...) có thể trả về ScrollResponse hoặc tuple (list, next_offset).
    """
    # 1. Load .env
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=project_root / ".env")

    QDRANT_URL     = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong .env")

    # 2. Khởi tạo client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False
    )

    offset = None   # lần đầu không truyền cursor
    limit  = 100    # mỗi lần lấy tối đa 100 điểm
    total  = 0

    while True:
        # 3. Gọi scroll để fetch batch tiếp theo
        response = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset
        )

        # 4. Xác định points_batch và next_offset dựa vào kiểu trả về
        # Trường hợp scroll trả về tuple (points_list, next_page_offset):
        if isinstance(response, tuple):
            points_batch, next_offset = response
        else:
            # Giả sử response là ScrollResponse
            # Ở đây qdrant-client 1.14.2 trả một ScrollResponse
            # response.result là List[PointStruct]
            points_batch = response.result
            # Lấy next_page_offset (trường này = None khi hết dữ liệu)
            next_offset = getattr(response, "next_page_offset", None)

        # 5. Nếu batch rỗng, đã hết dữ liệu -> dừng
        if not points_batch:
            break

        # 6. In từng record trong batch
        for rec in points_batch:
            # Một số trường hợp rec có thể None, bỏ qua
            if rec is None:
                continue
            # rec.payload là dict bạn đã upsert, lấy content + metadata
            payload = rec.payload or {}
            content = payload.get("content", "<No content>")
            metadata = payload.get("metadata", {})

            total += 1
            print(f"ID: {rec.id}")
            print(f"  document_type: {metadata.get('document_type', '<Unknown>')}")
            print(f"  source_file:   {metadata.get('source_file', '<Unknown>')}")
            print("  Content preview:")
            print(
                f"    {content[:200].replace(chr(10), ' ')}"
                f"{'...' if len(content) > 200 else ''}"
            )
            print("-" * 60)

        # 7. Cập nhật offset = next_offset để lấy batch tiếp theo
        if not next_offset:
            # next_offset = None hoặc 0 → đã hết
            break
        offset = next_offset

    print(f"\nTổng cộng đã tải về {total} chunks từ collection `{collection_name}`.")


if __name__ == "__main__":
    download_documents_from_qdrant(collection_name="quy_che_full")
