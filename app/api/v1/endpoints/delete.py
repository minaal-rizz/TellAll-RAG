# app/api/v1/endpoints/delete.py

from fastapi import APIRouter
from typing import List

from app.servicess.document_service import delete_docs, delete_all_docs

router = APIRouter()

@router.post("/delete/")
def delete_selected_docs(filenames: List[str]):
    msg, updated = delete_docs(filenames)
    return {"message": msg, "documents": updated["choices"]}

@router.post("/delete-all/")
async def delete_everything():
    msg, updated = delete_all_docs()
    return {"message": msg, "documents": updated["choices"]}
