# app/api/v1/endpoints/docs.py

from fastapi import APIRouter

from app.services.document_service import get_all_docs

router = APIRouter()

@router.get("/docs/")
async def list_uploaded_docs():
    docs = get_all_docs()
    return {"documents": docs}
