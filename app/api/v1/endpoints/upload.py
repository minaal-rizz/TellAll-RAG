# app/api/v1/endpoints/upload.py

from fastapi import APIRouter, UploadFile, File
from typing import List
import os

from app.services.document_service import process_single_file

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []
    for f in files:
        file_path = os.path.join(UPLOAD_DIR, f.filename)
        try:
            with open(file_path, "wb") as out_file:
                out_file.write(await f.read())
            status = process_single_file(file_path)
            saved_files.append({"filename": f.filename, "status": status})
        except Exception as e:
            return {"error": f"Failed to save {f.filename}: {e}"}
    return {"uploaded": saved_files}
