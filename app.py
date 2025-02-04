import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json
from google.ai import generativelanguage as glm
import google.generativeai as genai
from dotenv import load_dotenv
from io import BytesIO  # For handling file download

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize API
if API_KEY:
    genai.configure(api_key=API_KEY)

# Initialize models
MODEL_NAME = "models/embedding-001"
INDEX_FILE = "faiss_index.index"
CONFIG_FILENAME = "config.json"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "system_prompt": "You are a helpful assistant. Answer questions based on the provided context.",
        "stored_pdfs": [],
        "text_chunks": []
    }

# Function to save config as a downloadable JSON file
def save_config(config):
    """Save configuration as a JSON file."""
    json_bytes = json.dumps(config, indent=4).encode('utf-8')
    return BytesIO(json_bytes)

# Function to load config from an uploaded JSON file
def load_config(uploaded_file):
    """Load configuration from a JSON file uploaded by the user."""
    try:
        config_data = json.load(uploaded_file)
        st.session_state.config.update(config_data)  # Update session state directly
        st.sidebar.success("Configuration loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")

# PDF Processing Functions
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    return text

def process_pdf(file):
    text = extract_text_from_pdf(file)
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]  
    st.session_state.config["text_chunks"].extend(chunks)
    update_vector_db(chunks)  
    return chunks

# Vector Database Functions
def initialize_vector_db():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(INDEX_FILE) if os.path.exists(INDEX_FILE) else faiss.IndexFlatL2(384)
    return model, index

embedding_model, faiss_index = initialize_vector_db()

def update_vector_db(texts):
    embeddings = embedding_model.encode(texts)
    faiss_index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(faiss_index, INDEX_FILE)

# Gemini Integration
def generate_response(prompt, context):
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(
            f"{st.session_state.config['system_prompt']}\n\nContext: {context}\n\nQuestion: {prompt}",
            generation_config={"temperature": st.session_state.config["temperature"], "top_p": st.session_state.config["top_p"]}
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# RAG Pipeline
def retrieve_context(query, top_k=15):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    valid_indices = [i for i in indices[0] if i < len(st.session_state.config["text_chunks"])]
    return valid_indices

# URL Processing
def get_pdfs_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return [link.get("href") for link in soup.find_all("a") if link.get("href", "").endswith(".pdf")]

# Streamlit UI
st.title("📄 AI Document Q&A with Gemini")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")

    # Model Settings
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.config.get("temperature", 0.7))
    st.session_state.config["top_p"] = st.slider("Top-p Sampling", 0.0, 1.0, st.session_state.config.get("top_p", 0.9))

    # System Prompt
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config.get("system_prompt", ""))

    # Save and Load Configuration
    config_file = st.file_uploader("Upload Configuration", type=['json'])
    if config_file:
        load_config(config_file)  # Load and update config directly

    if st.button("Update and Download Configuration"):
        config_bytes = save_config(st.session_state.config)
        st.download_button(
            "Download Config",
            data=config_bytes,
            file_name=CONFIG_FILENAME,
            mime="application/json"
        )

# File Upload Section
st.header("Document Management")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
url_input = st.text_input("Or enter URL to scan for PDFs")

# Process URL PDFs
if url_input:
    pdf_links = get_pdfs_from_url(url_input)
    for link in pdf_links:
        if st.button(f"Process {link.split('/')[-1]}"):
            response = requests.get(link)
            with io.BytesIO(response.content) as pdf_file:
                chunks = process_pdf(pdf_file)
                st.session_state.config["stored_pdfs"].append(link)
                st.success("PDF processed successfully!")

# Process Uploaded Files
if uploaded_files:
    for file in uploaded_files:
        with io.BytesIO(file.getvalue()) as pdf_file:
            chunks = process_pdf(pdf_file)
            st.session_state.config["stored_pdfs"].append(file.name)
    st.success(f"Processed {len(uploaded_files)} files")

# Debugging: Check total stored chunks
st.sidebar.write(f"Total text chunks stored: {len(st.session_state.config['text_chunks'])}")

# Chat Interface
st.header("Chat with Documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (English/Swedish)"):
    # Detect language
    try:
        lang = detect(prompt)
    except:
        lang = "en"
    
    # Retrieve context
    context_indices = retrieve_context(prompt)
    context = " ".join([st.session_state.config["text_chunks"][i] for i in context_indices]) if context_indices else "No relevant context found."
    
    # Generate response
    with st.spinner("Generating response..."):
        response = generate_response(prompt, context)
    
    # Display messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)