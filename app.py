import streamlit as st
import os
import io
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json
from google.ai import generativelanguage as glm
import google.generativeai as genai
from dotenv import load_dotenv
from io import BytesIO
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from google.api_core import exceptions
from PyPDF2 import PdfReader
import pdfplumber
import asyncio
import aiohttp
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB")

genai.configure(api_key=API_KEY)

# MongoDB Connection
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client["userembeddings"]
collection = db["embeddings"]

model = genai.GenerativeModel("gemini-2.0-flash")

def initialize_vector_db():
    model_local = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(384)
    return model_local, index

embedding_model, faiss_index = initialize_vector_db()

# Ensure session state variables exist
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "stored_pdfs" not in st.session_state:
    st.session_state.stored_pdfs = []
if "config" not in st.session_state:
    st.session_state.config = {
        "temperature": 0.7,
        "top_p": 1.0,
        "system_prompt": "Answer based on context.",
        "text_chunks": []
    }
if "messages" not in st.session_state:
    st.session_state.messages = []

# Save & Load Config
def save_config(config):
    return BytesIO(json.dumps(config, indent=4).encode('utf-8'))

def load_config(uploaded_file):
    try:
        config_data = json.load(uploaded_file)
        st.session_state.config.update(config_data)
        st.sidebar.success("Configuration loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")

def extract_text(uploaded_file):
    """Extract text from a single PDF."""
    text = ""
    try:
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        with pdfplumber.open(file_bytes) as pdf:
            text = ''.join([page.extract_text() or " " for page in pdf.pages])
    except Exception as e:
        st.error(f"An error occurred with pdfplumber: {e}")

    if not text:
        try:
            file_bytes.seek(0)
            reader = PdfReader(file_bytes)
            text = ''.join([page.extract_text() or " " for page in reader.pages])
        except Exception as e:
            st.error(f"An error occurred with PyPDF2: {e}")

    return text

def process_pdf(file, filename):
    text = extract_text(file)
    # Chunk the text into pieces of 2000 characters
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    st.session_state.config["text_chunks"].extend(chunks)
    update_vector_db(chunks, filename)

# **Store FAISS Embeddings in MongoDB**
def update_vector_db(texts, filename):
    embeddings = embedding_model.encode(texts).tolist()
    documents = [{"filename": filename, "text": text, "embedding": emb} for text, emb in zip(texts, embeddings)]
    collection.insert_many(documents)
    faiss_index.add(np.array(embeddings, dtype="float32"))
    # Optionally, save the faiss index to disk:
    # faiss.write_index(faiss_index, "faiss.index")

def retrieve_context(query, top_k=15):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    stored_docs = list(collection.find({}, {"_id": 0, "embedding": 1, "text": 1}))

    if not stored_docs:
        return []

    embeddings = np.array([doc["embedding"] for doc in stored_docs], dtype="float32")
    texts = [doc["text"] for doc in stored_docs]

    if faiss_index.ntotal == 0:
        faiss_index.add(np.array(embeddings, dtype="float32"))

    top_k = min(top_k, len(texts))  # Avoid requesting more results than available
    distances, indices = faiss_index.search(np.array([query_embedding], dtype="float32"), top_k)

    # Remove duplicates from results
    seen = set()
    unique_texts = []
    for i in indices[0]:
        if i < len(texts) and texts[i] not in seen:
            seen.add(texts[i])
            unique_texts.append(texts[i])

    return unique_texts

import time

def generate_response(prompt, context):
    system_prompt = st.session_state.config["system_prompt"]
    temperature = st.session_state.config["temperature"]
    top_p = st.session_state.config["top_p"]

    input_parts = [system_prompt + context, prompt]
    generation_config = genai.GenerationConfig(
        max_output_tokens=2048,
        temperature=temperature,
        top_p=top_p,
        top_k=32
    )

    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(input_parts, generation_config=generation_config)
            return response.text
        except exceptions.ResourceExhausted:
            st.warning(f"API quota exceeded. Retrying... ({attempt+1}/{retries})")
            time.sleep(5)  # Wait before retrying
        except Exception as e:
            return f"Error generating response: {str(e)}"

    st.error("API quota exceeded. Please try again later.")
    return None

# **Delete Functions**
def delete_file(filename):
    """Delete a specific file and its embeddings from MongoDB."""
    collection.delete_many({"filename": filename})
    st.session_state.file_uploader_key += 1  # Reset file uploader
    st.rerun()

def delete_all_files():
    """Delete all files from MongoDB and FAISS."""
    collection.drop()
    st.session_state.stored_pdfs = []
    st.session_state.file_uploader_key += 1  # Reset file uploader
    st.rerun()

# ========= PDF Link Extraction via URL =========
async def fetch_and_process_pdf_links(url: str):
    """
    Crawls the given URL for PDF links, downloads each PDF, extracts its text, and updates the DB.
    """
    browser_config = BrowserConfig()  # Default browser configuration
    run_config = CrawlerRunConfig(remove_overlay_elements=True)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        # Get internal links; they might be dicts or strings
        internal_links = result.links.get("internal", [])
        pdf_links = []
        for link in internal_links:
            if isinstance(link, dict):
                href = link.get("href", "")
                if ".pdf" in href.lower():
                    pdf_links.append(href)
            elif isinstance(link, str) and ".pdf" in link.lower():
                pdf_links.append(link)
        
        if not pdf_links:
            st.info("No PDF links found on the provided URL.")
            return

        async with aiohttp.ClientSession() as session:
            for pdf_link in pdf_links:
                try:
                    async with session.get(pdf_link) as response:
                        if response.status == 200:
                            pdf_bytes = await response.read()
                            pdf_file = BytesIO(pdf_bytes)
                            filename = os.path.basename(pdf_link)
                            process_pdf(pdf_file, filename)
                            st.success(f"Processed PDF: {filename}")
                        else:
                            st.error(f"Failed to download PDF: {pdf_link}")
                except Exception as e:
                    st.error(f"Error processing {pdf_link}: {e}")
        st.success("Finished processing all PDF links.")

def process_pdf_links_from_url_sync(url: str):
    asyncio.run(fetch_and_process_pdf_links(url))

# ========= Streamlit UI =========

st.title("📄 AI Document Q&A with Gemini")

# **Sidebar Configuration**
with st.sidebar:
    st.header("Configuration")
    st.session_state.config["temperature"] = st.slider("Temperature", 0.0, 1.0, st.session_state.config["temperature"])
    st.session_state.config["top_p"] = st.slider("Top-p Sampling", 0.0, 1.0, st.session_state.config["top_p"])
    st.session_state.config["system_prompt"] = st.text_area("System Prompt", value=st.session_state.config["system_prompt"])

    # Load & Save Config
    config_file = st.file_uploader("Upload Configuration", type=['json'])
    if config_file:
        load_config(config_file)

    if st.button("Update and Download Configuration"):
        config_bytes = save_config(st.session_state.config)
        st.download_button("Download Config", data=config_bytes, file_name="config.json", mime="application/json")

# **File Upload Section**
st.header("📂 Document Management")
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key=st.session_state.file_uploader_key)

if uploaded_files:
    for file in uploaded_files:
        with io.BytesIO(file.getvalue()) as pdf_file:
            process_pdf(pdf_file, file.name)
            st.session_state.stored_pdfs.append(file.name)

    st.success(f"Processed {len(uploaded_files)} file(s)")
    st.session_state.file_uploader_key += 1
    st.rerun()

# **Display Stored Files**
st.subheader("Stored Files in Database")
stored_files = list(collection.distinct("filename"))

if stored_files:
    for filename in stored_files:
        col1, col2 = st.columns([0.8, 0.2])
        col1.write(f"📄 {filename}")
        if col2.button("🗑️ Delete", key=filename):
            delete_file(filename)

    if st.button("🗑️ Delete All Files"):
        delete_all_files()
else:
    st.info("No files stored in the database.")

# ========= PDF Link Extraction Section =========
st.header("🔗 Add PDFs via URL")
pdf_url = st.text_input("Enter a URL to crawl for PDFs:")
if st.button("Extract PDFs from URL"):
    if pdf_url:
        process_pdf_links_from_url_sync(pdf_url)
        st.rerun()
    else:
        st.warning("Please enter a valid URL.")

# **Chat Interface**
st.header("💬 Chat with Documents")
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (English/Swedish)"):
    try:
        lang = detect(prompt)
    except:
        lang = "en"

    retrieved_context = retrieve_context(prompt)
    context = " ".join(retrieved_context) if retrieved_context else "No relevant context found."

    with st.spinner("Generating response..."):
        response = generate_response(prompt, context)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
