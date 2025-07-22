# app/db/pinecone_upload.py

from app.core.pinecone_client import index
from app.core.config import NAMESPACE


def upsert_chunks_to_pinecone(vectors: list):
    """
    Insert chunks into Pinecone under the configured namespace.

    Args:
        vectors (list): List of vectors with id, values, and metadata
    """
    try:
        # Ensure namespace exists by adding and removing a dummy vector
        index.upsert(vectors=[{
            "id": "__init__",
            "values": [0.0] * 384,
            "metadata": {"source": "__init__"}
        }], namespace=NAMESPACE)
        index.delete(ids=["__init__"], namespace=NAMESPACE)

        index.upsert(vectors=vectors, namespace=NAMESPACE)
    except Exception as e:
        print(f"❌ Pinecone upsert failed: {e}")


def delete_chunks_from_pinecone_by_ids(chunk_ids: list):
    if chunk_ids:
        try:
            index.delete(ids=chunk_ids, namespace=NAMESPACE)
        except Exception as e:
            print(f"❌ Failed to delete chunks from Pinecone: {e}")


def delete_chunks_from_pinecone_by_source(source: str):
    try:
        index.delete(filter={"source": source}, namespace=NAMESPACE)
    except Exception as e:
        print(f"❌ Error deleting by source from Pinecone: {e}")
