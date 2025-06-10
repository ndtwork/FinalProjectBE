# app/routes/rag_admin.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pathlib import Path
import shutil, uuid
from app.utils.rag_utils import client, create_collection, delete_collection, ingest_file

from app.config.rag_config import QDRANT_COLLECTION_NAME  # default nếu muốn
from app.utils.settings import get_active_collection, set_active_collection
from app.utils.security import get_current_admin_user

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_admin_user)]  # ← khóa toàn bộ /admin
)

@router.post("/collections")
async def api_create_collection(
    name: str = Form(...),
    vector_size: int = Form(768),
    distance: str = Form("COSINE")
):
    try:
        create_collection(name, vector_size, distance)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"message": f"Collection `{name}` created."}

@router.delete("/collections/{name}")
async def api_delete_collection(name: str):
    try:
        delete_collection(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Delete failed: {e}")
    return {"message": f"Collection `{name}` deleted."}

@router.get("/collections")
async def list_collections():
    resp = client.get_collections()
    return {"collections": [c.name for c in resp.collections]}

@router.post("/collections/{name}/ingest")
async def api_ingest(
    name: str,
    document_type: str = Form(...),
    file: UploadFile = File(...)
):
    tmp_dir = Path("tmp_upload")
    tmp_dir.mkdir(exist_ok=True)
    tmp_fp = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
    with tmp_fp.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        ingest_file(tmp_fp, document_type, name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest error: {e}")
    finally:
        tmp_fp.unlink()

    return {"message": f"File `{file.filename}` ingested into collection `{name}`"}


@router.get("/settings/active_collection")
async def api_get_active_collection():
    name = get_active_collection() or QDRANT_COLLECTION_NAME
    return {"active_collection": name}

@router.put("/settings/active_collection")
async def api_set_active_collection(
    name: str = Form(..., description="Tên collection admin muốn active")
):
    # Kiểm tra xem collection có tồn tại không
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        raise HTTPException(404, detail=f"Collection `{name}` not found")
    set_active_collection(name)
    return {"message": f"Active collection set to `{name}`"}