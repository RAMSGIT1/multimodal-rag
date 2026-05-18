from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def create_vectorstore(docs, embeddings):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    split_docs = splitter.split_documents(docs)

    return FAISS.from_documents(split_docs, embeddings)