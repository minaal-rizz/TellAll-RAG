# app/core/pinecone_client.py

from pinecone import Pinecone
from app.core.config import PINECONE_API_KEY, PINECONE_INDEX

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    print(f"✅ Pinecone index '{PINECONE_INDEX}' initialized")
except Exception as e:
    print(f"❌ Pinecone initialization failed: {e}")
    index = None
