import os
import io
import logging
import streamlit as st
import PyPDF2
from os import environ
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set Streamlit page configuration
st.set_page_config(page_title="📝 Document Q&A with RAG", layout="wide")
st.title("📝 Document Q&A with Retrieval-Augmented Generation")

# Initialize OpenAI client and embeddings model (cached for performance)
@st.cache_resource
def get_openai_client():
    try:
        return OpenAI(api_key=environ.get('OPENAI_API_KEY'))
    except Exception as e:
        st.error("Failed to initialize OpenAI client.")
        logging.error(e)
        return None

client = get_openai_client()
embedding_model = OpenAIEmbeddings(
    api_key=environ.get('OPENAI_API_KEY'),
    model="openai.text-embedding-3-small"
)

# Initialize session state variables if not already set
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Ask me anything about your uploaded documents."}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents" not in st.session_state:
    st.session_state.documents = {}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error("Error extracting text from PDF.")
        logging.error(e)
        return ""

def extract_text(uploaded_file) -> str:
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_bytes = uploaded_file.getvalue()
        if file_extension == 'pdf':
            return extract_text_from_pdf(file_bytes)
        elif file_extension in ['txt', 'md']:
            return file_bytes.decode("utf-8")
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return ""
    except Exception as e:
        st.error(f"Error processing file: {uploaded_file.name}")
        logging.error(e)
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        logging.info(f"Text split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        st.error("Error splitting text into chunks.")
        logging.error(e)
        return []

def update_vector_store():
    try:
        all_chunks = []
        all_metadatas = []
        for doc_id, doc_info in st.session_state.documents.items():
            for idx, chunk in enumerate(doc_info["chunks"]):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc_info["file_name"],
                    "chunk_id": idx
                })
        if not all_chunks:
            st.session_state.vector_store = None
            return

        st.session_state.vector_store = FAISS.from_texts(
            texts=all_chunks,
            embedding=embedding_model,
            metadatas=all_metadatas
        )
        logging.info("Vector store updated successfully.")
    except Exception as e:
        st.error("Error updating the vector store.")
        logging.error(e)
        st.session_state.vector_store = None

def display_chat_history():
    """Display the conversation history using Streamlit's chat message feature."""
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

def generate_response(query: str) -> str:
    """
    Retrieve relevant document chunks using the vector store,
    then generate an answer using OpenAI's Chat API.
    """
    try:
        # Retrieve the top 3 similar chunks
        search_results = st.session_state.vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in search_results])
        sources = {doc.metadata["source"] for doc in search_results if doc.metadata.get("source")}

        # Build messages for the chat API
        messages = [
            {"role": "system", "content": f"Use the following context to answer the user's question:\n\n{context}"},
            {"role": "user", "content": query}
        ]

        # Create a streaming chat completion
        stream = client.chat.completions.create(
            model="openai.gpt-4o",  # Change model as needed
            messages=messages,
            stream=True
        )

        # Stream the response to the UI
        response = st.write_stream(stream)
        if sources:
            st.write(f"**Sources:** {', '.join(sources)}")
        return response
    except Exception as e:
        error_msg = "Error generating response."
        st.error(error_msg)
        logging.error(e)
        return error_msg


# Display chat history at the top
display_chat_history()

# File uploader: Supports multiple documents (txt, md, pdf)
uploaded_files = st.file_uploader("Upload documents (txt, md, or pdf)", type=["txt", "md", "pdf"], accept_multiple_files=True)

if uploaded_files:
    new_files_added = False
    for uploaded_file in uploaded_files:
        # Create a unique ID for each file to prevent re-processing duplicates
        file_id = f"{uploaded_file.name}_{id(uploaded_file)}"
        if file_id not in st.session_state.documents:
            file_content = extract_text(uploaded_file)
            if file_content:
                chunks = chunk_text(file_content)
                if chunks:
                    st.session_state.documents[file_id] = {
                        "file_name": uploaded_file.name,
                        "content": file_content,
                        "chunks": chunks
                    }
                    new_files_added = True
    # Update the vector store if new files have been processed
    if new_files_added:
        update_vector_store()
        st.success("Files uploaded and processed successfully!")

# Chat input (disabled if no documents have been uploaded)
question = st.chat_input("Ask a question about your documents", disabled=(len(st.session_state.documents) == 0))

if question and st.session_state.vector_store:
    # Append user query to conversation history
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"):
        answer = generate_response(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
