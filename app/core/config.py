# app/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

MONGO_URI = os.getenv("MONGO_URI")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NAMESPACE = os.getenv("NAMESPACE", "okay")

# Debug output for sanity check
print("🔐 API KEY:", PINECONE_API_KEY)
print("🌍 ENVIRONMENT:", PINECONE_ENV)
print("📂 INDEX:", PINECONE_INDEX)
print("🧪 Mongo URI:", MONGO_URI)
