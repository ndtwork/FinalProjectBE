# app/routes/rag_admin.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
import shutil, uuid
from app.utils.rag_utils import client, create_collection, delete_collection, ingest_file

router = APIRouter(tags=["admin"], prefix="/admin")

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
