import tempfile
import requests
import json
import numpy as np
import easyocr

from bs4 import BeautifulSoup
from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from pdf2image import convert_from_path
import pytesseract


# =========================
# OCR READER
# =========================
reader = easyocr.Reader(['en'], gpu=False)


# =========================
# PDF LOADER
# =========================
def load_pdfs(uploaded_files):

    docs = []

    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        pages = loader.load()

        text_found = False

        for i, p in enumerate(pages):

            text = p.page_content.strip()

            if text and len(text) > 50:

                text_found = True

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "pdf",
                            "file_name": file.name,
                            "page": i + 1
                        }
                    )
                )

        # OCR FALLBACK
        if not text_found:

            images = convert_from_path(
                path,
                first_page=1,
                last_page=5,
                # poppler_path=r"D:\poppler\Library\bin"
            )

            for i, img in enumerate(images):

                text = pytesseract.image_to_string(img)

                if text.strip():

                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": "pdf",
                                "file_name": file.name,
                                "page": i + 1,
                                "ocr": True
                            }
                        )
                    )

    return docs


# =========================
# IPYNB LOADER
# =========================
def load_ipynb(file):

    docs = []

    data = json.load(file)

    for i, cell in enumerate(data.get("cells", [])):

        cell_type = cell.get("cell_type")

        content = "".join(cell.get("source", []))

        if content.strip():

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "ipynb",
                        "cell_type": cell_type,
                        "cell_number": i + 1,
                        "file_name": file.name
                    }
                )
            )

    return docs


# =========================
# URL LOADER
# =========================
def load_urls(urls):

    docs = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for url in urls:

        try:

            r = requests.get(url, headers=headers, timeout=10)

            soup = BeautifulSoup(r.text, "html.parser")

            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)

            docs.append(
                Document(
                    page_content=text[:5000],
                    metadata={
                        "source": "web",
                        "url": url
                    }
                )
            )

        except Exception as e:

            docs.append(
                Document(
                    page_content=f"WEB ERROR: {str(e)}",
                    metadata={
                        "source": "web",
                        "url": url
                    }
                )
            )

    return docs


# =========================
# IMAGE LOADER
# =========================
def load_images(image_files):

    docs = []

    for img in image_files:

        try:

            image = Image.open(img).convert("RGB")

            image_np = np.array(image)

            result = reader.readtext(image_np, detail=0)

            text = " ".join(result)

            if not text.strip():
                text = "No text detected"

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "image",
                        "image_name": img.name
                    }
                )
            )

        except Exception as e:

            docs.append(
                Document(
                    page_content=f"IMAGE ERROR: {str(e)}",
                    metadata={
                        "source": "image",
                        "image_name": img.name
                    }
                )
            )

    return docs