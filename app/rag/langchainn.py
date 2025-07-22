# langchainn.py

from langchain.chains.retrieval_qa.base import RetrievalQA
 #chain that connects retriever and LLM to form a RAG pipeline
from langchain_community.vectorstores import Pinecone as LangchainPinecone # Wraps Pinecone index as a LangChain-compatible vector store for retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings #emb model, used fo text->vec
from langchain_community.chat_models import ChatOpenAI  #works w groq via openai compatible endpoint
import os  #access env var, like keys etc

from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from app.core.config import NAMESPACE
from app.core.pinecone_client import index


def get_vectorstore(index, namespace): #index: pc index obj where embeddings are stored. 
                #namespace: str in pc to scope all vector op
    """
    Wrap the Pinecone index as a LangChain vectorstore.

    This function sets up the vectorstore that LangChain uses to perform 
    similarity search and retrieve relevant document chunks.

    Initializes HuggingFace embeddings using the 'all-MiniLM-L6-v2' model.

    Wraps the provided Pinecone index using LangChain's Pinecone wrapper.
       - This makes Pinecone usable with LangChain tools like retrievers and QA chains.

    Namespaces helps organize embeddings

    Returns:
        langchain.vectorstores.Pinecone: A LangChain-compatible vectorstore that can be used
        for semantic search and question answering.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #pc to store chunks of docs, embed user queries, seaches within given namespace
    return LangchainPinecone(index, embeddings.embed_query, namespace=namespace, text_key="text") #+Retrieves the original text content via the "text" metadata key.


def get_langchain_qa_chain(index, namespace):
    """
    Build a RetrievalQA chain using LangChain, with Groq LLM and Pinecone retriever.
    
    Returns a ready-to-use question-answering chain.
    """
    vectorstore = get_vectorstore(index, namespace)
    llm = ChatOpenAI(temperature=0.0, #control how creative/random llm output is. stick to the retrieved context w 0
                     model="llama3-8b-8192",
                     base_url="https://api.groq.com/openai/v1",
                     api_key=os.environ.get("GROQ_API_KEY"))
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(), #pc retriever. que->embed->top k doc chunks->relevant ans
       # return_source_documents=True  # helpful for debugging
    )
