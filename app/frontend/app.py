#app.py 


import gradio as gr
import requests
import mimetypes #guess the MIME type (file type) of a file based on its filename or extension.
import os 
from pydantic import BaseModel
from typing import List


# --------- ENDPOINTS ---------
API_BASE = "http://127.0.0.1:8000"  # Base URL of the FastAPI backend
UPLOAD_ENDPOINT = f"{API_BASE}/upload/"
ASK_ENDPOINT = f"{API_BASE}/ask/"
DOCS_ENDPOINT = f"{API_BASE}/docs/"
DELETE_ENDPOINT = f"{API_BASE}/delete/"
DELETE_ALL_ENDPOINT = f"{API_BASE}/delete-all/"

# --------- CHAT FUNCTION ---------
def ask_backend(question: str) -> str:
    """
    Send a user question to the backend and return the AI's response.

    Args:
        question (str): The user's input question.

    Returns:
        str: The response from the backend or an error message.
    """
    # Check if the input question is empty or just whitespace
    if not question.strip():  
        return "⚠️ Please enter a valid question."

    try:
        # Send a POST request to the ASK_ENDPOINT with the question as form data
        response = requests.post(ASK_ENDPOINT, data={"question": question})

        # If the response is successful (HTTP 200 OK)
        if response.status_code == 200:
            # Try to return the "response" field from the JSON, or a fallback message
            return response.json().get("response", "⚠️ No response returned.")
        else:
            # If not 200, return the error message from the response (or "Unknown error")
            return f"❌ Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        # If an exception occurs (e.g., network error), return the error message
        return f"❌ Request failed: {str(e)}"

# --------- UPLOAD FUNCTION ---------

UPLOAD_ENDPOINT = "http://localhost:8000/upload/"  # FastAPI upload endpoint

def upload_files(files):
    """
    Upload one or more documents to the backend for embedding and storage.

    Args:
        files (list): A list of Gradio File objects selected by the user.

    Returns:
        str: A message indicating upload success or error.
    """
    if not files:
        return "⚠️ No files selected."

    try:
        results = []  # To store status messages for each file

        # Loop over each uploaded file
        for f in files:
            # Detect MIME type (optional but useful for setting headers)
            mime_type, _ = mimetypes.guess_type(f.name)

            # Open file in binary read mode
            with open(f.name, "rb") as file_obj:
                # Send file to FastAPI backend using multipart/form-data
                response = requests.post(
                    UPLOAD_ENDPOINT,
                    files={"files": (f.name, file_obj, mime_type or "application/octet-stream")}
                )

                # Parse JSON response from FastAPI
                data = response.json()

                # Extract filename and status for each file in response
                for item in data.get("uploaded", []):
                    
                    status = item["status"]
                    results.append(f"{status}")  # Only show user-friendly status line

        # Join all results into a final string
        return "\n".join(results)

    except Exception as e:
        # Handle any exception and show error
        return f"❌ Upload error: {str(e)}"
    

# --------- LOAD DOCUMENTS FUNCTION ---------
def load_docs():
    """
    Fetch the list of uploaded documents from the backend.

    Returns:
        gr.update: A Gradio update with new choices for the document list.
    """
    try:
    # Send a GET request to the backend endpoint to fetch the list of uploaded documents
            response = requests.get(DOCS_ENDPOINT)

    # Extract the list of document names from the JSON response,
    # and update the Gradio CheckboxGroup choices with those document names (initial value is empty)
            return gr.update(choices=response.json().get("documents", []), value=[])

    except Exception:
    # If any error occurs (e.g., connection issue, bad response), 
    # return an empty list for both choices and selected values
            return gr.update(choices=[], value=[])


# --------- DELETE SELECTED FILES FUNCTION ---------



def delete_selected(docs):
    """
    Delete selected documents from the backend.

    Args:
        docs (list): List of selected document filenames.

    Returns:
        tuple: Deletion status message and updated document list.
    """

    if not docs:
        return "⚠️ No documents selected.", load_docs()

    try:
        # Send filenames to FastAPI DELETE endpoint
        response = requests.post(DELETE_ENDPOINT, json= docs)

        if response.status_code == 200:
    # If the response from the backend is successful (HTTP 200 OK),
    # parse the JSON response body
            data = response.json()

    # Return a success message (from response or a default),
    # and update the document list in the UI (choices and selected values cleared)
            return data.get("message", "✅ Selected files deleted."), gr.update(choices=data.get("documents", []), value=[])

        else:
    # If the response status is not 200 (e.g., 400, 500), handle it gracefully.
    # Return an error message with the status code and backend error message,
    # and reload the docs list as fallback.
            return f"❌ Delete failed: {response.status_code} - {response.text}", load_docs()

    except Exception as e:
    # Catch any exceptions (e.g., network issues, JSON decoding errors),
    # and return a user-friendly error message along with reloading the docs list.
        return f"❌ Delete error: {str(e)}", load_docs()


    '''if not docs:
        return "⚠️ No documents selected.", load_docs()
    
    try:
        response = requests.post(DELETE_ENDPOINT, json={"filenames": docs})
        if response.status_code == 200:
            message = response.json().get("message", "✅ Selected file(s) deleted.")
            updated_docs = response.json().get("documents", [])
            return message, gr.update(choices=updated_docs, value=[])
        else:
            return f"❌ Delete failed: {response.status_code}", load_docs()
    except Exception as e:
        return f"❌ Delete error: {e}", load_docs()'''


# --------- DELETE ALL FILES FUNCTION ---------
def delete_all():
    """
    Delete all documents stored in the backend.

    Returns:
        tuple: Deletion status message and updated (empty) document list.
    """
    try:
        response = requests.post(DELETE_ALL_ENDPOINT)

        if response.status_code == 200:
            data = response.json()
            return data.get("message", "✅ All documents deleted."), gr.update(choices=data.get("documents", []), value=[])
        else:
            return f"❌ Delete-all failed: {response.status_code} - {response.text}", load_docs()

    except Exception as e:
        return f"❌ Delete-all error: {str(e)}", load_docs()



    '''try:
        response = requests.post(DELETE_ALL_ENDPOINT)
        if response.status_code == 200:
            msg = response.json().get("message", "✅ All documents deleted.")
            updated_docs = response.json().get("documents", [])
            return msg, gr.update(choices=updated_docs, value=[])
        else:
            return f"❌ Delete-all failed: {response.status_code}", load_docs()
    except Exception as e:
        return f"❌ Delete-all error: {e}", load_docs()'''

# --------- GRADIO UI ---------
with gr.Blocks(title="📚 TellAll RAG Interface") as app:
    """
    Main Gradio application that provides a UI for:
    - Uploading documents
    - Asking questions based on uploaded content
    - Managing (deleting) uploaded documents
    """

    # ---- Upload Tab ----
    with gr.Tab("Upload"):
        gr.Markdown("### 📤 Upload Documents")
        file_input = gr.File(
            label="Upload your documents",
            file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".xls"],
            file_count="multiple"
        )
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Upload Status", lines=4)
        upload_btn.click(upload_files, inputs=file_input, outputs=upload_output)


    # ---- Ask Tab ----
    with gr.Tab("Ask"):
        gr.Markdown("### 💬 Ask Questions About Your Files")
        question = gr.Textbox(label="Your Question", lines=3, placeholder="Type your question here...")
        ask_btn = gr.Button("Submit")
        loading_msg = gr.Markdown("⏳ Processing...", visible=False)
        answer_output = gr.Markdown()
        

        def with_loading(q):
            return gr.update(visible=True), None

        def without_loading(q):
            return gr.update(visible=False), ask_backend(q)

        ask_btn.click(with_loading, inputs=question, outputs=[loading_msg, answer_output])
        ask_btn.click(without_loading, inputs=question, outputs=[loading_msg, answer_output])

    # ---- Delete Tab ----
    with gr.Tab("Delete"):
        gr.Markdown("### 🗑️ Delete Documents")
        docs_list = gr.CheckboxGroup(label="Documents", choices=[])
        delete_btn = gr.Button("Delete Selected")
        delete_all_btn = gr.Button("Delete All")
        refresh_btn = gr.Button("🔁 Refresh Documents")
        delete_msg = gr.Textbox(label="Delete Status")

        # Load document list initially when the app starts
        app.load(load_docs, outputs=docs_list)

        # Refresh document list manually
        refresh_btn.click(load_docs, outputs=docs_list)

        # Delete selected documents
        delete_btn.click(delete_selected, inputs=docs_list, outputs=[delete_msg, docs_list])

        # Delete all documents
        delete_all_btn.click(delete_all, outputs=[delete_msg, docs_list])

# --------- LAUNCH APP ---------
if __name__ == "__main__":
    # Launch the Gradio app on port 7860
    # show_api=False disables auto-generated Gradio API documentation at /api
    # share=True creates a public link (useful for demos or mobile access)
    app.launch(server_port=7860, show_api=False, share=True)
