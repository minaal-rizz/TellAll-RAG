from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "rag-index"
cloud = "aws"
region = "us-east-1"

pc = Pinecone(api_key=pinecone_api_key)

# Delete old index if it exists
if index_name in [index.name for index in pc.list_indexes()]:
    print(f"🔁 Deleting old index: {index_name}")
    pc.delete_index(index_name)

# Create new index
print(f"✅ Creating index: {index_name}")
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec={"serverless": {"cloud": cloud, "region": region}}
)
print("🎉 Index created successfully.")
