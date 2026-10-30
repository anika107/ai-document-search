---
title: ai-document-search
app_file: app.py
sdk: gradio
sdk_version: 4.44.1
---

# AI Document Search

Upload PDF documents, build a searchable knowledge base, and ask questions about their contents. The app extracts PDF text, creates embeddings, stores them in a FAISS index, and uses a Hugging Face text generation model to answer questions with retrieved context.

## Run locally

Use Python 3.9 or newer. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the Gradio app:

```bash
python app.py
```

Open the local URL printed in the terminal.

## Use the app

1. Select one or more PDF files in the upload control.
2. Choose **Build Knowledge Base** and wait for processing to finish.
3. Enter a question and submit it to see an answer and retrieved source excerpts.

The UI handles PDF uploads. The app processes each upload directly and writes the generated FAISS index to `vectorstore/`. That folder is ignored by Git; rebuild the index by uploading PDFs after a fresh deployment or whenever the runtime's local files are reset.

## Deploy to Hugging Face Spaces

Create a Gradio Space and add `app.py`, `config.py`, `ingest.py`, and `requirements.txt`. The YAML header above identifies the Space SDK and app entry point.

The current `config.py` sets the Transformers pipeline to `device=0`, which requires a GPU. For a CPU-only Space, change it to `device=-1` before deploying. CPU inference can be slow, especially on the first request while models are being downloaded and loaded. Free Spaces may sleep when idle, and their local runtime files are not persistent, so uploaded PDFs and the generated index may need to be reprocessed after a restart.

## Project files

- `app.py` — Gradio interface, upload handler, and question answering.
- `ingest.py` — PDF text extraction, chunking, embedding, and FAISS index creation.
- `config.py` — Hugging Face text generation model configuration.
- `requirements.txt` — Python dependencies.
- `vectorstore/` — Generated FAISS index; created by the UI and excluded from Git.

## Notes

- The embedding model is `sentence-transformers/all-MiniLM-L6-v2` (loaded through `HuggingFaceEmbeddings`).
- The answer model is `google/flan-t5-base`.
- This app uses `allow_dangerous_deserialization=True` when loading the local FAISS index. Only load an index created by this app or another trusted source.
