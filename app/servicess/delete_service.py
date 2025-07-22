# app/servicess/delete_service.py

import gradio as gr
from app.core.mongo import docs_collection
from app.core.pinecone_client import index
from app.core.config import NAMESPACE


def delete_docs(filenames):
    """
    Deletes selected documents and their associated vector embeddings from MongoDB and Pinecone.

    Args:
        filenames (list[str]): List of filenames to delete.

    Returns:
        tuple: (message string, Gradio UI update for document list)
    """
    deleted = []
    for f in filenames:
        try:
            if docs_collection is not None:
                docs_collection.delete_many({"filename": f})
            index.delete(filter={"source": f}, namespace=NAMESPACE)
            deleted.append(f)
        except Exception as e:
            print(f"❌ Delete error for {f}: {e}")

    updated_docs = get_all_docs()
    return f"🗑️ Deleted: {', '.join(deleted)}", gr.update(choices=updated_docs, value=[])


def delete_all_docs():
    """
    Deletes all documents and associated vectors except '__init__' to preserve Pinecone namespace.

    Returns:
        tuple: (status message, Gradio UI update to clear document list)
    """
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


def get_all_docs():
    """
    Helper to retrieve all filenames from MongoDB.
    """
    try:
        if docs_collection is None:
            return []
        return [doc["filename"] for doc in docs_collection.find()]
    except Exception as e:
        print(f"❌ Mongo fetch error: {e}")
        return []
