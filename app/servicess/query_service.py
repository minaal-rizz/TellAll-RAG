# app/servicess/query_service.py

from app.core.pinecone_client import index
from app.core.groq_client import groq_client
from app.core.config import NAMESPACE
from app.rag.langchainn import get_vectorstore
from app.utilities.reranker import rerank_chunks


def ask_llm(question, context, model_name="llama3-8b-8192"):
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": (
                    "You are an intelligent assistant. You must answer ALL user questions using only the provided context. "
                    "Give a detailed answer in accordance with the context. If a specific answer cannot be found in the context, clearly state: "
                    "'The context does not contain information about [that part]'. Always mention filename and page/slide/sheet/row references."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
            ],
            max_tokens=512,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LLM error: {e}"


def run_rag_pipeline(q: str) -> str:
    if not q.strip():
        return "⚠️ Please enter a question."

    vectorstore = get_vectorstore(index, NAMESPACE)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents(q)

    matches = [
        {
            "metadata": {**doc.metadata, "text": doc.page_content},
            "id": doc.metadata.get("id", "")
        }
        for doc in docs if doc.page_content.strip()
    ]

    if not matches:
        return "❌ No relevant context found for reranking."

    matches = rerank_chunks(q, matches, top_k=5)

    ctx, refs = "", []
    for m in matches:
        meta = m["metadata"]
        fn = meta.get("source", "Unknown")
        txt = meta.get("text", "[No text found]")

        location = "Location Unknown"
        ext = fn.lower().split(".")[-1]
        if ext in ["ppt", "pptx"] and meta.get("slide"):
            location = f"Slide {meta['slide']}"
        elif ext in ["xls", "xlsx"] and meta.get("sheet") and meta.get("row"):
            location = f"Sheet {meta['sheet']}, Row {meta['row']}"
        elif ext in ["pdf", "docx"] and meta.get("page"):
            location = f"Page {meta['page']}"

        ctx += f"\n--- {fn} ({location}) ---\n{txt}\n"
        refs.append(f"{fn} ({location})")

    ans = ask_llm(q, ctx)
    return f"### 🧠 Answer\n{ans}\n\n---\n\n### 📚 References\n" + "\n".join(f"- {r}" for r in refs)
