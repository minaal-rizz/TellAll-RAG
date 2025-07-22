# app/core/groq_client.py

from groq import Groq
from app.core.config import GROQ_API_KEY

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"❌ Failed to initialize Groq client: {e}")
    groq_client = None
