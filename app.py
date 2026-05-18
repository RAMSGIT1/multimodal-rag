import streamlit as st

from modules.llm import get_llm
from modules.embeddings import get_embeddings
from modules.loaders import load_pdfs, load_urls, load_images
from modules.vectorstore import create_vectorstore

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Multi-Modal RAG", layout="wide")
st.title("📚 Multi-Modal RAG")


# =========================
# SESSION STATE
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# PDF EXPORT
# =========================
def generate_chat_pdf(messages, filename="chat.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = []

    for m in messages:
        role = "You" if m["role"] == "user" else "AI"
        content.append(Paragraph(f"{role}: {m['content']}", styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)
    return filename


# =========================
# SIDEBAR - MODEL SELECTION
# =========================
st.sidebar.title("⚙️ Settings")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["groq", "openai", "gemini", "ollama"]
)

if provider == "groq":
    model = st.sidebar.selectbox(
        "Groq Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "openai/gpt-oss-20b"]
    )

elif provider == "openai":
    model = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )

elif provider == "gemini":
    model = st.sidebar.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"]
    )

elif provider == "ollama":
    model = st.sidebar.selectbox(
        "Ollama Model",
        ["llama3", "mistral", "phi3"]
    )


# =========================
# INPUT SECTION
# =========================
pdf_files = st.sidebar.file_uploader("📄 PDFs", type="pdf", accept_multiple_files=True)
url_input = st.sidebar.text_area("🌐 Enter URLs")
image_files = st.sidebar.file_uploader("🖼️ Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)


# =========================
# PROCESS INPUTS
# =========================
if st.sidebar.button("🚀 Process Inputs"):

    docs = []

    if pdf_files:
        docs.extend(load_pdfs(pdf_files))

    if url_input:
        urls = [u.strip() for u in url_input.split("\n") if u.strip()]
        docs.extend(load_urls(urls))

    if image_files:
        docs.extend(load_images(image_files))

    if not docs:
        st.error("No documents uploaded")
        st.stop()

    embeddings = get_embeddings()
    st.session_state.vectorstore = create_vectorstore(docs, embeddings)

    st.success("System Ready 🚀")


# =========================
# TOKEN SAFE UTIL
# =========================
def trim(text, limit=1200):
    if not text:
        return ""
    return text[:limit]


# =========================
# CHAT SECTION
# =========================
st.subheader("💬 Chat")

query = st.text_input("Ask something")
send = st.button("Send")


if send and query:

    if not st.session_state.vectorstore:
        st.warning("Please process inputs first")
        st.stop()

    llm = get_llm(provider, model)

    # =========================
    # FIX 1: SMALL RETRIEVAL (IMPORTANT)
    # =========================
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)


    # =========================
    # FIX 2: CLEAN SOURCE SPLIT
    # =========================
    pdf_docs = []
    img_docs = []
    web_docs = []

    for d in docs:

        src = d.metadata.get("source", "")

        if src == "pdf":
            pdf_docs.append(d)

        elif src == "image":
            img_docs.append(d)

        elif src == "web":
            web_docs.append(d)

        else:
            web_docs.append(d)


    # =========================
    # FIX 3: SAFE TRUNCATION (CRITICAL FOR GROQ)
    # =========================
    pdf_text = trim("\n".join([d.page_content for d in pdf_docs]))
    img_text = trim("\n".join([d.page_content for d in img_docs]))
    web_text = trim("\n".join([d.page_content for d in web_docs]))


    # =========================
    # FIX 4: LIGHTWEIGHT PROMPT
    # =========================
    prompt = f"""
You are a multimodal RAG assistant.

Use ONLY the provided data.

PDF:
{pdf_text}

IMAGE:
{img_text}

WEB:
{web_text}

Question:
{query}

Answer simply:
"""


    response = llm.invoke(prompt)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    st.rerun()


# =========================
# CHAT HISTORY
# =========================
if st.session_state.messages:
    st.subheader("🧠 Chat History")

    for m in st.session_state.messages:
        st.write(("🧑 You:" if m["role"] == "user" else "🤖 AI:"), m["content"])


# =========================
# DOWNLOAD CHAT
# =========================
st.sidebar.markdown("📥 Export Chat")

if st.session_state.messages and st.sidebar.button("Download Chat PDF"):
    file = generate_chat_pdf(st.session_state.messages)

    with open(file, "rb") as f:
        st.sidebar.download_button("Download PDF", f, file_name="chat.pdf")