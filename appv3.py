import streamlit as st
import time
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from modules.llm import get_llm
from modules.embeddings import get_embeddings
from modules.loaders import load_pdfs, load_urls, load_images, load_ipynb
from modules.vectorstore import create_vectorstore

import warnings
warnings.filterwarnings("ignore")
# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Multi-Modal RAG", layout="wide")
st.title("🚀 Multi-Modal RAG ")


# =========================
# SESSION STATE
# =========================
if "vs" not in st.session_state:
    st.session_state.vs = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []


# =========================
# CHAT EXPORT (TXT)
# =========================
def export_txt(chat):
    return "\n\n".join([f"{r.upper()}: {m}" for r, m in chat])


# =========================
# CHAT EXPORT (PDF)
# =========================
def export_pdf(chat):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    for r, m in chat:
        content.append(Paragraph(f"<b>{r.upper()}:</b> {m}", styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)
    buffer.seek(0)
    return buffer


# =========================
# SAFE CONTEXT BUILDER
# =========================
def build_context(docs):

    if not docs:
        return ""

    return "\n\n".join(
        [d.page_content for d in docs if d.page_content][:10]
    )


# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

provider = st.sidebar.selectbox("Provider", ["groq", "openai", "gemini", "ollama"])
api_key = st.sidebar.text_input("API Key", type="password")

model_map = {
    "groq": [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "gemini": ["gemini-1.5-flash"],
    "ollama": ["llama3", "mistral"]
}

model = st.sidebar.selectbox("Model", model_map[provider])


# =========================
# INPUTS
# =========================
pdf_files = st.sidebar.file_uploader("📄 PDFs or IPYNB", type=["pdf", "ipynb"], accept_multiple_files=True)
urls_raw = st.sidebar.text_area("🌐 URLs (comma or newline)")
image_files = st.sidebar.file_uploader("🖼️ Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)


# =========================
# PROCESS INPUTS
# =========================
if st.sidebar.button("🚀 Process Inputs"):

    docs = []

    for file in pdf_files:

        if file.name.endswith(".pdf"):
            docs += load_pdfs([file])

        elif file.name.endswith(".ipynb"):
            docs += load_ipynb(file)

    if urls_raw:
        urls = [u.strip() for u in urls_raw.replace("\n", ",").split(",") if u.strip()]
        docs += load_urls(urls)

    if image_files:
        docs += load_images(image_files)

    if not docs:
        st.error("No documents loaded")
        st.stop()

    # 🔥 IMPORTANT FIX
    all_docs = docs   # store everything

    embeddings = get_embeddings()

    vectorstore = create_vectorstore(all_docs, embeddings)

    # ✅ STORE IN SESSION
    st.session_state.vs = vectorstore
    st.session_state.all_docs = all_docs

    # ✅ DEBUG (VERY IMPORTANT - REMOVE LATER)
    pdf_docs = [d for d in docs if d.metadata.get("source") == "pdf"]

    st.write(f"📄 PDF Pages Loaded: {len(pdf_docs)}")

    for d in pdf_docs[:3]:
        st.write(d.page_content[:300])
        st.write("📄 Sample Loaded Data:")
    for d in all_docs[:3]:
        st.write(d.metadata, d.page_content[:200])

    st.success(f"Loaded {len(all_docs)} documents")



# =========================
# CHAT HISTORY
# =========================
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)


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
        st.write(query)

    llm = get_llm(provider, model, api_key)

    # =========================
    # RETRIEVAL
    # =========================
    retriever = st.session_state.vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )

    docs = retriever.invoke(query)
    docs = docs + st.session_state.all_docs

    context = build_context(docs)

    if not context.strip():
        context = "\n\n".join([d.page_content for d in docs[:5] if d.page_content])


    # =========================
    # SOURCE SEPARATION PROMPT (FINAL FIX)
    # =========================
    # 🔥 ALWAYS INCLUDE PDF (not only retrieved)
    pdf_docs = [d for d in st.session_state.all_docs if d.metadata.get("source") == "pdf"]
    ipynb_docs = [d for d in st.session_state.all_docs if d.metadata.get("source") == "ipynb"]
    web_docs = [d for d in docs if d.metadata.get("source") == "web"]
    img_docs = [d for d in docs if d.metadata.get("source") == "image"]

    pdf_text = "\n".join([d.page_content for d in pdf_docs])
    ipynb_text = "\n".join([d.page_content for d in ipynb_docs])
    web_text = "\n".join([d.page_content for d in web_docs])
    img_text = "\n".join([d.page_content for d in img_docs])
    prompt = f"""
You are an advanced MULTI-MODAL RAG assistant.

INSTRUCTIONS:
- Use all sources (PDF, Web, Image)
- Combine information intelligently
- If something is missing, ignore it (do NOT say Not found)
- Prefer factual explanation

---------------------

📄 PDF CONTENT:
{pdf_text}

📓 IPYNB CONTENT:
{ipynb_text}

🌐 WEB CONTENT:
{web_text}

🖼️ IMAGE CONTENT:
{img_text}

---------------------

QUESTION:
{query}

---------------------

FINAL ANSWER:
- Give a clear explanation
- Compare sources if relevant
- Be simple and accurate
"""


    # =========================
    # RESPONSE
    # =========================
    with st.chat_message("assistant"):
        box = st.empty()

        try:
            res = llm.invoke(prompt)

            output = ""
            for w in res.content.split():
                output += w + " "
                box.markdown(output + "▌")
                time.sleep(0.01)

            box.markdown(output)

        except Exception as e:
            output = str(e)
            box.markdown(output)

    st.session_state.chat.append(("assistant", output))


# =========================
# DOWNLOAD OPTIONS
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