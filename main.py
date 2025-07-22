# app/main.py

from fastapi import FastAPI
from app.api.v1.endpoints import upload, ask, delete, docs

app = FastAPI(title="TellAll RAG API", version="1.0")

# Register API routers
app.include_router(upload.router)
app.include_router(ask.router)
app.include_router(delete.router)
app.include_router(docs.router)

# Optional health check
@app.get("/")
async def root():
    return {"status": "✅ TellAll RAG API is up and running!"}
