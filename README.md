# 🧠 TellAll RAG System

**TellAll** is an AI-powered document Q&A system built with **FastAPI**, **Gradio**, **LangChain**, **Pinecone**, **MongoDB**, and **Groq LLM**. It enables users to upload documents (PDF, DOCX, PPTX, XLSX), extract and embed their content, ask natural language questions, and get contextual answers with source references.

---

## 📁 Project Structure
```
TellAll_RAG/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/              # Versioned route handlers
│   │   │   │   ├── upload.py
│   │   │   │   ├── ask.py
│   │   │   │   ├── delete.py
│   │   │   │   ├── docs.py
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── services/                       # Core service logic
│   │   ├── document_service.py
│   │   ├── query_service.py
│   │   ├── delete_service.py
│   │   └── __init__.py
│   ├── utilities/                      # Text extraction, reranking, etc.
│   │   ├── extractors.py
│   │   ├── reranker.py
│   │   └── __init__.py
│   ├── rag/                            # LangChain integration
│   │   ├── langchainn.py
│   │   └── __init__.py
│   ├── db/                             # Database + vector DB interface
│   │   ├── mongo_upload.py
│   │   ├── pinecone_upload.py
│   │   └── __init__.py
│   ├── core/                           # Configuration and clients
│   │   ├── config.py
│   │   ├── pinecone_client.py
│   │   └── mongo_client.py
│   └── __init__.py
├── frontend/
│   ├── app.py                          # Gradio UI frontend
│   └── __init__.py
├── .env                                # Secrets and API keys
├── requirements.txt
└── README.md
├──--main.py                         # FastAPI app entry point

```

---

## 🚀 Features

✅ Upload multiple document types (PDF, DOCX, PPTX, XLSX)  
✅ Extract and embed content into Pinecone + MongoDB  
✅ Ask questions through natural language  
✅ CrossEncoder reranking improves answer quality  
✅ Groq-powered LLM provides accurate contextual responses  
✅ Document deletion (single or bulk)  
✅ Easy Gradio frontend for user interaction

---

## 🛠️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/minaal-rizz/TellAll-RAG.git
cd TellAll_RAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env`
Create a `.env` file in the root:
```env
MONGO_URI=your-mongo-uri
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-region
PINECONE_INDEX=tellall-index
NAMESPACE=okay
GROQ_API_KEY=your-groq-key
```

### 4. Start the Backend
```bash
uvicorn main:app --reload
```

### 5. Start the Frontend (Gradio)
```bash
python frontend/app.py
```

Access the app at: [http://localhost:7860](http://localhost:7860)

---

## 🧪 API Endpoints
| Method | Endpoint         | Description                     |
|--------|------------------|---------------------------------|
| POST   | `/upload/`       | Upload documents                |
| POST   | `/ask/`          | Ask question (form: `question`) |
| POST   | `/delete/`       | Delete selected files           |
| POST   | `/delete-all/`   | Delete all files                |
| GET    | `/docs/`         | List uploaded documents         |

---

## 🧠 RAG Pipeline
1. **Embed query** using SentenceTransformer
2. **Retrieve** chunks from Pinecone
3. **Re-rank** with CrossEncoder
4. **Ask** Groq LLM strictly based on context
5. **Return** answer + references (file/page/slide/sheet info)


---

## ✨ Credits
- [FastAPI](https://fastapi.tiangolo.com)
- [LangChain](https://www.langchain.com)
- [Pinecone](https://www.pinecone.io)
- [Groq](https://groq.com)
- [Gradio](https://gradio.app)

---
