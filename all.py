
import os
import sys
import openai  # OpenAI client for Groq API
from datetime import datetime, timezone  # for timestamping uploads
from functools import lru_cache  # for caching model and index loading
import contextlib

# Imports for machine learning and document parsing
from sentence_transformers import CrossEncoder  # used for reranking chunks based on query
from sentence_transformers import SentenceTransformer  # used to generate embeddings from text
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splits large texts into manageable chunks

# Gradio and database utilities
import gradio as gr  # UI framework
from pymongo import MongoClient  # MongoDB client for document storage
from pinecone import Pinecone  # Pinecone client for vector search

# Document processing
from pptx import Presentation  # read PowerPoint files
from docx import Document  # read Word documents
import fitz  # PyMuPDF: for reading PDFs
from pdf2image import convert_from_path  # convert PDF to image for OCR fallback
import pytesseract  # OCR engine
import pandas as pd  # for Excel files
import certifi  # SSL certs for secure MongoDB
from groq import Groq  # Groq client for LLM API

# Utilities
import tempfile  # temporary storage for file conversion
import shutil  # file operations
from docx2pdf import convert  # convert DOCX to PDF


from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

# Load environment variables
NAMESPACE = os.getenv("NAMESPACE", "__default__")
print(f"✅ ENV LOADED: NAMESPACE={NAMESPACE}")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")
MONGO_URI = os.getenv("MONGO_URI")
print(f"📂 Using Pinecone namespace: '{NAMESPACE}' on index: '{PINECONE_INDEX}'")

# Initialize Groq LLM client
openai.api_base = "https://api.groq.com/openai/v1"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Connect to MongoDB securely
try:
    client = MongoClient("mongodb+srv://minaalriz4:12345@cluster0.keay9eg.mongodb.net/tellall?retryWrites=true&w=majority", tls=True, tlsCAFile=certifi.where())
    db = client["tellall"]
    docs_collection = db["documents"]
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    docs_collection = None

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
model = SentenceTransformer("all-MiniLM-L6-v2")  # load sentence embedding model
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # define how text is chunked

@lru_cache(maxsize=1)
def load_clients():
    """
    Load and cache the embedding model and Pinecone index.

    Returns:
        tuple: Cached SentenceTransformer model and Pinecone index instance.
    """
    return model, index

def sync_mongo_pinecone():
    """
    Synchronize Pinecone and MongoDB to delete stale vector chunks.

    - Collects chunk IDs from MongoDB
    - Queries Pinecone for existing vector IDs
    - Deletes Pinecone vectors that do not exist in MongoDB

    Returns:
        None
    """
    print("🔄 Syncing MongoDB and Pinecone...")

    if docs_collection is None:
        print("❌ MongoDB not connected.")
        return

    try:
        # collect all chunk ids from Mongo
        mongo_docs = list(docs_collection.find())
        mongo_chunk_ids = set()
        for doc in mongo_docs:
            mongo_chunk_ids.update(doc.get("chunk_ids", []))

        print(f"📦 Found {len(mongo_chunk_ids)} chunk IDs in MongoDB.")

        # query pinecone with dummy vector just to get metadata for up to 1000 items
        res = index.query(
            vector=[0.0]*384,
            top_k=1000,
            include_metadata=True,
            namespace=NAMESPACE
        )
        pinecone_ids = {match["id"] for match in res.get("matches", [])}
        print(f"📱 Found {len(pinecone_ids)} chunk IDs in Pinecone.")

        # find stale chunks that are in Pinecone but not Mongo
        orphaned = pinecone_ids - mongo_chunk_ids
        if orphaned:
            print(f"🩹 Deleting {len(orphaned)} stale Pinecone chunks...")
            index.delete(ids=list(orphaned), namespace=NAMESPACE)
        else:
            print("✅ No stale Pinecone chunks found.")

        print("✅ MongoDB ↔️ Pinecone sync complete.")
    except Exception as e:
        print(f"❌ Sync failed: {e}")

def extract_text(file_path):
    """
    Extract readable text from supported file formats, with support for
    OCR fallback and structure awareness (page/slide/sheet/row).

    Args:
        file_path (str): Path to the uploaded document.

    Returns:
        list: A list of strings, each representing a page/slide/sheet chunk.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()  # get extension

        if ext == ".pdf":
            doc = fitz.open(file_path)  # open PDF
            text = [p.get_text() for p in doc if p.get_text().strip()]  # get text from each page
            return text or [pytesseract.image_to_string(img) for img in convert_from_path(file_path)]

        elif ext == ".docx":
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_pdf_path = os.path.join(tmpdir, "converted.pdf")
                shutil.copy(file_path, os.path.join(tmpdir, "upload.docx"))
                convert(os.path.join(tmpdir, "upload.docx"), tmp_pdf_path)  # convert to PDF
                doc = fitz.open(tmp_pdf_path)
                text = [p.get_text() for p in doc if p.get_text().strip()]
                return text or [pytesseract.image_to_string(img) for img in convert_from_path(tmp_pdf_path)]

        elif ext == ".pptx":
            prs = Presentation(file_path)  # open PPTX
            return [
                f"Slide {i+1}:\n" + "\n".join(
                    shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()
                ) for i, slide in enumerate(prs.slides)
            ]

        elif ext in [".xls", ".xlsx"]:
            text_chunks = []
            dfs = pd.read_excel(file_path, sheet_name=None)  # read all sheets
            for sheet_name, df in dfs.items():
                df = df.fillna("").astype(str)  # clean up and convert to string
                for i, row in df.iterrows():
                    row_text = ", ".join(row.tolist())
                    if row_text.strip():
                        text_chunks.append(f"Sheet: {sheet_name} | Row: {i+1}\n{row_text}")
            return text_chunks

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

    return []

# instantiate reranker model
re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_chunks(query, chunks, top_k=5):
    """
    Re-rank Pinecone search chunks using CrossEncoder for better relevance.

    Args:
        query (str): The user's search query.
        chunks (list): List of chunks retrieved from Pinecone.
        top_k (int): Number of top results to return after re-ranking.

    Returns:
        list: Top-k most relevant chunks based on reranking scores.
    """
    texts = [chunk["metadata"]["text"] for chunk in chunks]  # extract text only
    pairs = [(query, text) for text in texts]  # prepare (query, chunk) pairs
    scores = re_ranker.predict(pairs)  # run reranker
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)  # sort by score
    return [chunk for chunk, _ in ranked[:top_k]]

def process_single_file(file):
    """
    Process a single uploaded file: extract text, split into chunks, embed,
    store in Pinecone, and log metadata in MongoDB.

    Args:
        file (UploadedFile): A file uploaded via Gradio UI.

    Returns:
        str: Upload status message (success/failure with chunk count).
    """
    try:
        filename = os.path.basename(file.name)  # get filename

        # Check and delete existing entries
        if docs_collection:
            existing_doc = docs_collection.find_one({"filename": filename})
            if existing_doc:
                chunk_ids = existing_doc.get("chunk_ids", [])
                if chunk_ids:
                    print(f"🩹 Deleting {len(chunk_ids)} old chunks for {filename} from Pinecone")
                    index.delete(ids=chunk_ids, namespace=NAMESPACE)
                docs_collection.delete_one({"_id": existing_doc["_id"]})

        pages = extract_text(file.name)  # extract pages
        if not pages:
            return f"⚠️ No text found in {filename}"

        all_chunks = []
        for page_num, page_text in enumerate(pages, start=1):
            splits = splitter.split_text(page_text)  # chunk the page text
            for i, chunk in enumerate(splits):
                if not chunk.strip():
                    continue
                embedding = model.encode(chunk).tolist()  # embed chunk
                meta = {
                    "source": filename,
                    "chunk": i + 1,
                    "text": chunk,
                    "page": page_num
                }
                if "Slide" in page_text:
                    meta["slide"] = page_num
                elif "Sheet:" in page_text:
                    try:
                        meta["sheet"] = page_text.split("Sheet:")[1].split("|")[0].strip()
                        meta["row"] = page_num
                    except: pass

                all_chunks.append({
                    "id": f"{filename}-p{page_num}-c{i}",
                    "values": embedding,
                    "metadata": meta
                })

        # init Pinecone namespace if needed
        try:
            index.upsert(vectors=[{
                "id": "__init__",
                "values": [0.0] * 384,
                "metadata": {"source": "__init__"}
            }], namespace=NAMESPACE)
            index.delete(ids=["__init__"], namespace=NAMESPACE)
        except Exception as e:
            print(f"⚠️ Namespace init failed: {e}")

        index.upsert(vectors=all_chunks, namespace=NAMESPACE)  # upload to Pinecone

        # store metadata in MongoDB
        docs_collection.insert_one({
            "filename": filename,
            "chunks": len(all_chunks),
            "pages": len(pages),
            "chunk_ids": [chunk["id"] for chunk in all_chunks],
            "uploaded_at": datetime.now(timezone.utc)
        })

        return f"✅ Uploaded {filename} ({len(all_chunks)} chunks)"
    except Exception as e:
        return f"❌ Upload failed: {e}"

# More docstrings will continue below for other functions.
