from pymongo import MongoClient
from pinecone import Pinecone
import gradio as gr
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
mongo_uri = os.getenv("MONGO_URI")

# Setup
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(pinecone_index)

client = MongoClient(mongo_uri)
db = client["tellall"]
docs_collection = db["documents"]
def get_all_docs():
    return [doc["filename"] for doc in docs_collection.find()]

def delete_docs(files_to_delete):
    deleted = []

    for filename in files_to_delete:
        docs_collection.delete_many({"filename": filename})
        try:
            index.delete(filter={"source": filename})
        except Exception as e:
            print(f"❌ error deleting from Pinecone for {filename}: {e}")
        deleted.append(filename)

    updated_choices = get_all_docs()
    msg = f"🗑️ Deleted: {', '.join(deleted)}" if deleted else "⚠️ No files deleted."
    return msg, gr.update(choices=updated_choices, value=[])

def delete_all_docs():
    docs_collection.delete_many({})
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"❌ error deleting all from Pinecone: {e}")
    return "🗑️ All documents deleted.", gr.update(choices=[], value=[])

def load_docs():
    filenames = get_all_docs()
    return gr.update(choices=filenames, value=[])


#UI
with gr.Blocks() as delete_tab:
    gr.Markdown("### 🗑️ Manage Documents")
    gr.Markdown("Delete one, multiple, or all documents (and their embeddings).")

    files_to_delete = gr.CheckboxGroup(choices=[], label="Select files to delete")
    delete_btn = gr.Button("Delete Selected")
    delete_all_btn = gr.Button("Delete All Documents")  # New button
    result = gr.Textbox(label="Delete Result")

    def delete_selected(files):
        if not files:
            return "⚠️ No files selected.", gr.update()
        return delete_docs(files)

    delete_tab.load(load_docs, outputs=files_to_delete)
    delete_btn.click(delete_selected, inputs=files_to_delete, outputs=[result, files_to_delete])
    delete_all_btn.click(delete_all_docs, outputs=[result, files_to_delete])  # Connect button to action
