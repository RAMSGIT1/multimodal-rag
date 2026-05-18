import streamlit as st
import time
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from modules.llm import get_llm
from modules.embeddings import get_embeddings
from modules.loaders import (
    load_pdfs,
    load_urls,
    load_images,
    load_ipynb
)
from modules.vectorstore import create_vectorstore

import warnings
warnings.filterwarnings("ignore")

import re

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted
)


# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="Multi-Modal RAG",
    layout="wide"
)

st.title("🚀 Multi-Modal RAG")


# =========================
# SESSION STATE
# =========================
if "vs" not in st.session_state:
    st.session_state.vs = None

if "chat" not in st.session_state:
    st.session_state.chat = []


# =========================
# EXPORT TXT
# =========================
def export_txt(chat):

    return "\n\n".join(
        [f"{r.upper()}: {m}" for r, m in chat]
    )


# =========================
# EXPORT PDF
# =========================
def clean_text_for_pdf(text):

    # Remove markdown tables
    text = re.sub(r"\|.*\|", "", text)

    # Remove markdown headers
    text = re.sub(r"#+ ", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Replace bullets
    text = text.replace("•", "-")

    return text.strip()


def export_pdf(chat):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    for role, msg in chat:

        clean_msg = clean_text_for_pdf(msg)

        title = Paragraph(
            f"<b>{role.upper()}</b>",
            styles["Heading3"]
        )

        body = Preformatted(
            clean_msg,
            styles["Code"]
        )

        content.append(title)

        content.append(Spacer(1, 6))

        content.append(body)

        content.append(Spacer(1, 14))

    doc.build(content)

    buffer.seek(0)

    return buffer


# =========================
# SAFE CONTEXT
# =========================
def build_context(docs, max_chars=6000):

    context_parts = []

    total_chars = 0

    for d in docs:

        source = d.metadata.get("source", "unknown")

        text = d.page_content.strip()

        if not text:
            continue

        chunk = f"[{source.upper()}]\n{text}\n"

        total_chars += len(chunk)

        if total_chars > max_chars:
            break

        context_parts.append(chunk)

    return "\n".join(context_parts)


def clean_response(text):

    import re

    # Remove markdown tables
    text = re.sub(r'\|', '', text)

    # Remove markdown headers
    text = re.sub(r'#+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove markdown bold
    text = text.replace("**", "")

    # Remove unwanted section names
    text = text.replace("Main Answer", "")
    text = text.replace("Important Insights", "")
    text = text.replace("Final Summary", "")

    # Add spacing before numbered sections
    text = re.sub(r'(\d+\.)', r'\n\n\1', text)

    # Clean repeated blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

provider = st.sidebar.selectbox(
    "Provider",
    ["groq", "openai", "gemini", "ollama"]
)

api_key = st.sidebar.text_input(
    "API Key",
    type="password"
)

model_map = {

    "groq": [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ],

    "openai": [
        "gpt-4o-mini",
        "gpt-4o"
    ],

    "gemini": [
        "gemini-1.5-flash"
    ],

    "ollama": [
        "llama3",
        "mistral"
    ]
}

model = st.sidebar.selectbox(
    "Model",
    model_map[provider]
)


# =========================
# INPUTS
# =========================
pdf_files = st.sidebar.file_uploader(
    "📄 PDFs / IPYNB",
    type=["pdf", "ipynb"],
    accept_multiple_files=True
)

urls_raw = st.sidebar.text_area(
    "🌐 URLs"
)

image_files = st.sidebar.file_uploader(
    "🖼️ Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


# =========================
# PROCESS INPUTS
# =========================
if st.sidebar.button("🚀 Process Inputs"):

    docs = []

    # PDFs + NOTEBOOKS
    for file in pdf_files:

        if file.name.endswith(".pdf"):
            docs += load_pdfs([file])

        elif file.name.endswith(".ipynb"):
            docs += load_ipynb(file)

    # URLS
    if urls_raw:

        urls = [
            u.strip()
            for u in urls_raw.replace("\n", ",").split(",")
            if u.strip()
        ]

        docs += load_urls(urls)

    # IMAGES
    if image_files:
        docs += load_images(image_files)

    if not docs:

        st.error("No documents loaded")

        st.stop()

    embeddings = get_embeddings()

    vectorstore = create_vectorstore(
        docs,
        embeddings
    )

    st.session_state.vs = vectorstore

    st.success(f"Loaded {len(docs)} documents")


# =========================
# CHAT HISTORY
# =========================
for role, msg in st.session_state.chat:

    with st.chat_message(role):
        st.markdown(msg)


# =========================
# CHAT INPUT
# =========================
query = st.chat_input("Ask anything...")

if query:

    if not st.session_state.vs:

        st.warning("Process inputs first")

        st.stop()

    st.session_state.chat.append(("user", query))

    with st.chat_message("user"):
        st.markdown(query)

    llm = get_llm(
        provider,
        model,
        api_key
    )

    # =========================
    # RETRIEVAL
    # =========================
    retriever = st.session_state.vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )
    

    retrieved_docs = retriever.invoke(query)

    context = build_context(retrieved_docs)

    # =========================
    # PROMPT
    # =========================
    prompt = f"""
You are an advanced Multi-Modal RAG assistant.

Answer using ONLY the provided context.

RULES:
- Use clean numbered sections
- Keep answers concise and practical
- Use short bullet points
- Avoid markdown syntax
- Avoid markdown tables
- Avoid bold formatting
- Avoid long paragraphs
- Avoid repetition

CONTEXT:
{context}

QUESTION:
{query}

Provide a direct well-structured answer.
"""

    # =========================
    # RESPONSE
    # =========================
    with st.chat_message("assistant"):

        box = st.empty()

        try:

            res = llm.invoke(prompt)

            cleaned = clean_response(res.content)

            output = ""

            for line in cleaned.split("\n"):

                output += line + "\n"

                box.markdown(output + "▌")

                time.sleep(0.03)

            box.markdown(output)

        except Exception as e:

            output = str(e)

            box.error(output)

    st.session_state.chat.append(
        ("assistant", output)
    )


# =========================
# DOWNLOADS
# =========================
st.sidebar.divider()

st.sidebar.subheader("📥 Download Chat")

if st.session_state.chat:

    st.sidebar.download_button(
        "⬇️ TXT",
        export_txt(st.session_state.chat),
        file_name="chat.txt"
    )

    st.sidebar.download_button(
        "⬇️ PDF",
        export_pdf(st.session_state.chat),
        file_name="chat.pdf"
    )