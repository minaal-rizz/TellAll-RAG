# app/core/mongo.py

import certifi
from pymongo import MongoClient
from app.core.config import MONGO_URI

try:
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    db = client["tellall"]
    docs_collection = db["documents"]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    docs_collection = None
