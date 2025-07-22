# app/api/v1/endpoints/ask.py

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
import traceback

from app.services.query_service import run_rag_pipeline

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not question.strip():
        return JSONResponse(content={"error": "Question is empty."}, status_code=400)
    try:
        response = run_rag_pipeline(question)
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"Internal error: {str(e)}"}, status_code=500)
