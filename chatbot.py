#python chatbot.py


import os
import time
from dotenv import load_dotenv
import pinecone
from llama_cpp import Llama
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
load_dotenv()


# --- Load environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX", "rag-index")

client = MongoClient(os.getenv("MONGO_URI"))
db = client["tellall"]
docs = db["documents"]


# --- Initialize clients
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)
llm = Llama(
    model_path="./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",  # ✅ path to the local model file
    n_ctx=4096,        # max context length (how many tokens the model can "remember" at once)
    n_threads=8,       # number of CPU threads used for processing (higher = faster on CPU)
    n_gpu_layers=35,   # number of layers to offload to GPU (improves speed if GPU available)
    verbose=False      # turn off extra debug/info messages in the console
)



'''llm = Llama(
  model_path="./capybarahermes-2.5-mistral-7b.Q4_K_M.gguf",  # Download the model file first
  n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance. replicas, for con current processes.
  n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available

)'''


# --- Function: Get relevant docs from Pinecone
def get_top_chunks(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return res['matches']

# --- Function: Ask LLaMA
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

# --- CHAT LOOP
print("🤖 Ask me anything from the uploaded documents (type 'exit' to quit):")
while True:
    question = input("\n❓ You: ")
    if question.lower() in ["exit", "quit"]:
        break

    print("🔍 Searching...")
    matches = get_top_chunks(question)
    if not matches:
        print("❌ No relevant information found.")
        continue

    # --- Prepare context
    combined_context = ""
    references= set()  
    for match in matches:
        metadata = match.get('metadata', {})
        filename = metadata.get('source', 'Unknown')
        page=metadata.get('page', 'Unknown')
        text = match.get("metadata", {}).get("text", "[No text found]")

        
        combined_context += f"\n--- From {filename}, Page {page} ---\n{text}\n"
        references.add(f"{filename} (Page {page})")
        

    # --- Get response
    print("💬 Generating answer...")
    answer = ask_llm(question, combined_context)
    print(f"\n🧠 LLama response: {answer}")
    
    print ("filename: " , filename, "page:", page)
    # print("\n📄 References:")
    # for ref in sorted(references):
    #     print(f" . {ref}")

    
