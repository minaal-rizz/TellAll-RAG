# app/utils/reranker.py

from sentence_transformers import CrossEncoder

# Load CrossEncoder model once
re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_chunks(query, chunks, top_k=5):
    """
    Re-rank retrieved document chunks based on semantic relevance to the query using a CrossEncoder.

    Args:
        query (str): User input query.
        chunks (List[Dict]): Retrieved chunks with 'metadata.text'.
        top_k (int): Number of top-ranked chunks to return.

    Returns:
        List[Dict]: Top-k chunks sorted by semantic relevance.
    """
    texts = [chunk["metadata"].get("text", "") for chunk in chunks]
    pairs = [(query, text) for text in texts if text.strip()]

    if not pairs:
        return []

    scores = re_ranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
