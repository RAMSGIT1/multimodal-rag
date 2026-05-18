---
title: multimodal-rag
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 🚀 Multi-Modal RAG Assistant

# 🚀 Multi-Modal RAG Assistant

A powerful Multi-Modal Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, FAISS, OCR, and multiple LLM providers.

This project supports:

- 📄 PDF document analysis
- 📓 Jupyter Notebook (`.ipynb`) understanding
- 🌐 Website content retrieval
- 🖼️ Image OCR extraction
- 🤖 Multiple LLM providers
- 🔍 Semantic search using FAISS
- 💬 Conversational Q&A
- 📥 Chat export (TXT & PDF)

---

# ✨ Features

## 📄 PDF Processing
- Extracts text using `PyPDFLoader`
- OCR fallback using:
  - `pdf2image`
  - `pytesseract`

## 📓 Notebook Understanding
- Reads `.ipynb` notebook cells
- Supports:
  - markdown cells
  - code cells

## 🌐 Web Scraping
- Extracts clean text from URLs
- Removes:
  - scripts
  - styles
  - unwanted HTML

## 🖼️ Image OCR
- Extracts text from images using:
  - EasyOCR
  - Tesseract OCR

## 🔍 Vector Search
- Uses:
  - Sentence Transformers
  - FAISS Vector Database

## 🤖 Multi-LLM Support
Supports:
- Groq
- OpenAI
- Gemini
- Ollama

## 💬 Conversational RAG
- Semantic retrieval
- Context-aware responses
- Structured answers

## 📥 Export Options
- TXT export
- PDF export

---

# 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Framework | LangChain |
| Vector DB | FAISS |
| Embeddings | sentence-transformers |
| OCR | pytesseract, EasyOCR |
| PDF Parsing | PyPDFLoader |
| Image Processing | Pillow |
| Web Scraping | BeautifulSoup |
| LLM Providers | Groq, OpenAI, Gemini, Ollama |

---

# 📂 Project Structure

```bash
multimodal_rag/
│
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
│
├── modules/
│   ├── loaders.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   └── llm.py
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone YOUR_GITHUB_REPO_URL
cd multimodal_rag
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🖥️ Local Run

```bash
streamlit run app.py
```

---

# 🔑 Supported LLM Providers

## Groq
- openai/gpt-oss-20b
- llama-3.1-8b-instant
- llama-3.3-70b-versatile
- mixtral-8x7b-32768

## OpenAI
- gpt-4o-mini
- gpt-4o

## Gemini
- gemini-1.5-flash

## Ollama
- llama3
- mistral

---

# 📄 Supported Inputs

| Type | Supported |
|---|---|
| PDF | ✅ |
| IPYNB | ✅ |
| URLs | ✅ |
| Images | ✅ |

---

# 🔍 Retrieval Pipeline

1. Load documents
2. Split text into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Generate grounded response using LLM

---

# 🧠 OCR Pipeline

## PDFs
- Try native PDF extraction
- If extraction fails:
  - Convert pages to images
  - Run OCR

## Images
- Extract text directly using OCR

---

# 📥 Export Features

Supports:
- Chat export as TXT
- Chat export as PDF

---

# 🚀 Deployment

## Hugging Face Spaces

### Required Files

- `requirements.txt`
- `packages.txt`

### packages.txt

```txt
tesseract-ocr
poppler-utils
```

### Run Command

HF Spaces automatically detects Streamlit apps.

---

# ⚠️ Important Notes

## Windows
You may need:
- Tesseract OCR
- Poppler

## Linux / Hugging Face
Dependencies are installed automatically via:
- `requirements.txt`
- `packages.txt`

---

# 🔒 Limitations

- Large PDFs may increase response time
- OCR quality depends on image clarity
- Free-tier deployments may sleep after inactivity

---

# 📈 Future Improvements

- Hybrid Retrieval (BM25 + Dense)
- Reranking
- Metadata Filtering
- Conversational Memory
- Qdrant / ChromaDB
- Streaming Token Generation
- Source Citations
- Async OCR Processing
- Multi-query Retrieval
- Parent Document Retrieval

---

# 👨‍💻 Author

Built as a Multi-Modal RAG portfolio project using:
- Streamlit
- LangChain
- FAISS
- OCR
- Multi-provider LLMs

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub.