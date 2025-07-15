# python app.py

import os
import sys
import gradio as gr
import contextlib
from dotenv import load_dotenv
from pymongo import MongoClient
from functools import lru_cache
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")


client = MongoClient(os.getenv("MONGO_URI"))
db = client["tellall"]
docs = db["documents"]


# hides model loading logs from terminal
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield  # run block silently
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr  # restore outputs

# load and cache embedder + pinecone + llm
@lru_cache(maxsize=1)
def load_clients():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # embed text into vectors
    pc = Pinecone(api_key=PINECONE_API_KEY)  # connect to pinecone
    index = pc.Index(PINECONE_INDEX)  # select index

    with suppress_output():
        llm = Llama(
            model_path="./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",  # path to gguf model
            n_ctx=4096,            # context window
            n_threads=8,           # cpu threads
            n_gpu_layers=35,       # load layers on gpu
            verbose=False          # hide output
        )

    return embedder, index, llm  # return all clients

# get top k similar chunks from pinecone
def get_top_chunks(query, embedder, index, top_k=5):
    vector = embedder.encode(query).tolist()  # encode question into vector
    response = index.query(vector=vector, top_k=top_k, include_metadata=True)  # query pinecone
    return response.get("matches", [])  # return matched chunks

# send context + question to llama
def ask_llm(question, context, llm):
    prompt = f"""<|system|>
You are a precise assistant. You must answer ALL user questions using only the provided context.
If a specific answer cannot be found in the context, clearly state: "The context does not contain information about [that part]".
Always mention the filename and page number where the information was found.
<|end|>


<|user|>
Context:
{context}

Question: {question}
<|end|>

<|assistant|>"""

    result = llm(prompt, max_tokens=384, stop=["<|end|>"])  # generate answer with stop token
    return result["choices"][0]["text"].strip()  # return answer text only

# main que ans pipeline
def rag_chat(question):
    if not question.strip():
        return "⚠️ Please enter a question."  # skip empty questions

    embedder, index, llm = load_clients()  # load all clients
    matches = get_top_chunks(question, embedder, index)  # find relevant chunks

    if not matches:
        return "❌ No relevant documents found."  # nothing matched

    context = ""  # collected context to send to llm
    references = []  # filenames + pages shown to user
    for match in matches:
        meta = match.get("metadata", {})  # get metadata like file and page
        filename = meta.get("source", "Unknown")  # original file name
        page = meta.get("page", "N/A")  # page number
        text = match.get("metadata", {}).get("text", "[No text found]")


        context += f"\n--- From {filename} (Page {page}) ---\n{text}\n"  # append to context
        references.append(f"{filename} (Page {page})")  # add to list of sources

    print("🔍 Final Context Sent to LLaMA:\n", context[:2000])  # print first 2000 chars for debug

    answer = ask_llm(question, context, llm)  # send to llm

    return f"""### 🧠 Answer  
{answer}

---

### 📚 References  
""" + "\n".join(f"- {ref}" for ref in references)  # show answer and source

# gradio interface setup
interface = gr.Interface(
    fn=rag_chat,  # function to call
    inputs=gr.Textbox(label="❓ Ask a question", placeholder="e.g., What’s mentioned about work experience?", lines=2),
    outputs=gr.Markdown(label="📄 Answer with References"),  # show markdown result
    title="🧠 TellAll: RAG-Powered Chatbot",
    description="Ask questions about your indexed PDF/DOCX files. Powered by LLaMA + Pinecone.",
    allow_flagging="never"
)

# launch gradio app
if __name__ == "__main__":
    interface.launch(share=False, debug=False, show_error=True)
