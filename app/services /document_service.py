# app/services/document_service.py

import os
from datetime import datetime, timezone
import gradio as gr

from app.core.mongo import docs_collection
from app.core.pinecone_client import index
from app.utils.extractors import extract_text
from app.core.config import NAMESPACE

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def process_single_file(file_path: str) -> str:
    filename = os.path.basename(file_path)

    if docs_collection is not None:
        existing_doc = docs_collection.find_one({"filename": filename})
        if existing_doc:
            chunk_ids = existing_doc.get("chunk_ids", [])
            if chunk_ids:
                index.delete(ids=chunk_ids, namespace=NAMESPACE)
            docs_collection.delete_one({"_id": existing_doc["_id"]})

    pages = extract_text(file_path)
    if not pages:
        return f"⚠️ No text found in {filename}"

    all_chunks = []
    for page_num, page_text in enumerate(pages, start=1):
        splits = splitter.split_text(page_text)
        for i, chunk in enumerate(splits):
            if not chunk.strip():
                continue
            embedding = model.encode(chunk).tolist()
            meta = {
                "source": filename,
                "chunk": i + 1,
                "text": chunk,
                "page": page_num
            }
            all_chunks.append({
                "id": f"{filename}-p{page_num}-c{i}",
                "values": embedding,
                "metadata": meta
            })

    index.upsert(vectors=all_chunks, namespace=NAMESPACE)

    docs_collection.insert_one({
        "filename": filename,
        "chunks": len(all_chunks),
        "pages": len(pages),
        "chunk_ids": [chunk["id"] for chunk in all_chunks],
        "uploaded_at": datetime.now(timezone.utc)
    })

    return f"✅ Uploaded {filename} ({len(all_chunks)} chunks)"


def get_all_docs():
    if docs_collection is None:
        return []
    return [doc["filename"] for doc in docs_collection.find()]


def delete_docs(filenames):
    deleted = []
    for f in filenames:
        try:
            if docs_collection is not None:
                docs_collection.delete_many({"filename": f})
            index.delete(filter={"source": f}, namespace=NAMESPACE)
            deleted.append(f)
        except Exception as e:
            print(f"❌ Delete error: {e}")
    updated = get_all_docs()
    return f"🗑️ Deleted: {', '.join(deleted)}", gr.update(choices=updated, value=[])


def delete_all_docs():
    try:
        if docs_collection is not None:
            docs_collection.delete_many({})

        res = index.query(
            vector=[0.0] * 384,
            top_k=1000,
            include_metadata=True,
            namespace=NAMESPACE
        )

        ids_to_delete = [
            m["id"] for m in res.get("matches", [])
            if m["metadata"].get("source") != "__init__"
        ]

        if ids_to_delete:
            index.delete(ids=ids_to_delete, namespace=NAMESPACE)

        return "🗑️ All documents deleted (namespace preserved).", gr.update(choices=[], value=[])
    except Exception as e:
        return f"❌ Delete all failed: {e}", gr.update()
