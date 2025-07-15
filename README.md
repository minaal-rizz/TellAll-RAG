# 🧠 TellAll RAG Chat System — File-Aware Question Answering (Gradio UI)

This is a powerful Retrieval-Augmented Generation (RAG) app built with:
- 📦 **MongoDB** for document metadata
- 🌲 **Pinecone** for semantic vector search
- 🤖 **Groq-hosted LLaMA 3** for response generation
- 🧠 **Sentence Transformers** for embeddings + reranking
- 🗂️ Support for **PDF, DOCX, PPTX, XLSX**
- 💬 **Gradio UI** for uploading files, chatting, and managing documents

---

## 🚀 Features

- 📁 Upload documents of different types
- 📄 Text is extracted and chunked with metadata like page/slide/sheet info
- 🧠 Chunks embedded and stored in Pinecone
- 🗂️ Metadata stored in MongoDB, including chunk IDs
- 🔁 Auto-sync between MongoDB and Pinecone at startup
- 💬 Ask questions and get context-aware answers
- 📚 Sources and page references shown with every response
- 🧹 Re-uploads automatically replace old document chunks

---

## 📦 Supported File Types

- `.pdf` (with OCR fallback)
- `.docx` (converted to PDF using Tesseract-compatible OCR)
- `.pptx` (slide-wise text)
- `.xlsx`, `.xls` (sheet and row-based extraction)

---

## 🧰 Tech Stack

| Layer       | Tool / Service                  |
|-------------|----------------------------------|
| Embeddings  | SentenceTransformer (`MiniLM`)   |
| Vector DB   | Pinecone                         |
| LLM         | Groq API (LLaMA 3 - 8B)          |
| OCR         | `pytesseract`, `pdf2image`       |
| UI          | Gradio                           |
| Sync Logic  | Chunk-ID-level Mongo ↔ Pinecone  |
| Reranker    | `cross-encoder/ms-marco-MiniLM`  |

---

## 📂 Project Structure
📁 your_project/
│
├── app.py # Main backend + Gradio interface
├── .env # Environment variables (see below)
├── requirements.txt # Dependencies
└── README.md # You're reading this



---

## ⚙️ Environment Variables (`.env`)

Create a `.env` file in your root directory with the following keys:

PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX=rag-index
NAMESPACE=Default
MONGO_URI=your-mongodb-uri
GROQ_API_KEY=your-groq-key

-----

---

## 📦 Installation
1. **Clone the repo**:
   ```bash
   git clone https://github.com/minaal-rizz/TellAll-RAG.git
   cd rag-chat
----
Install requirements:
pip install -r requirements.txt
# SalesNet-RAG



