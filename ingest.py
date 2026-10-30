# ingest.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import fitz 
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def read_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    doc.close()
    return text

def ingest_pdfs(files):
    all_docs = []

    if not files:
        print("❌ No PDF files received.")
        return

    for file in files:
        file_path = str(file)
        filename = os.path.basename(file_path)
        print(f"📄 Reading: {filename}")
        text = read_pdf_text(file_path).strip()

        if not text:
            print(f"⚠️ Skipping empty PDF: {filename}")
            continue

        #chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_text(text)
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " "]
            )
        chunks = splitter.split_text(text)
        if not chunks:
            print(f"⚠️ No chunks created from: {filename}")
            continue

        docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
        all_docs.extend(docs)

    if not all_docs:
        print("❌ No documents to embed. Check PDF content.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(all_docs, embeddings)
    vectordb.save_local("vectorstore")
    print("✅ Vectorstore saved successfully.")

