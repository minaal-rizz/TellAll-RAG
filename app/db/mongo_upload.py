# app/db/mongo_upload.py

from datetime import datetime, timezone
from app.core.mongo import docs_collection


def save_to_mongo(filename: str, chunk_ids: list, pages: int, chunks: int) -> None:
    """
    Save document metadata to MongoDB.

    Args:
        filename (str): Name of the uploaded file
        chunk_ids (list): List of vector chunk IDs
        pages (int): Number of pages/slides/rows
        chunks (int): Number of text chunks created
    """
    if docs_collection is None:
        print("❌ MongoDB not connected.")
        return

    docs_collection.insert_one({
        "filename": filename,
        "chunk_ids": chunk_ids,
        "pages": pages,
        "chunks": chunks,
        "uploaded_at": datetime.now(timezone.utc)
    })


def remove_from_mongo(filename: str) -> list:
    """
    Remove document metadata and return chunk IDs.

    Args:
        filename (str): Name of the document to delete

    Returns:
        list: Chunk IDs removed from MongoDB
    """
    if docs_collection is None:
        return []

    existing_doc = docs_collection.find_one({"filename": filename})
    if existing_doc:
        docs_collection.delete_one({"_id": existing_doc["_id"]})
        return existing_doc.get("chunk_ids", [])
    return []
