
from langdetect import detect
from deep_translator import GoogleTranslator
import pdfplumber
import docx

def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return normalize_language(extract_pdf_text(uploaded_file))
    elif filename.endswith(".docx"):
        return normalize_language(extract_docx_text(uploaded_file))
    elif filename.endswith(".txt"):
        return normalize_language(uploaded_file.read().decode("utf-8"))
    else:
        raise ValueError("Unsupported file type")


def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


def extract_docx_text(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def normalize_language(text: str) -> str:
    try:
        lang = detect(text)

        if lang == "hi":
            translated = GoogleTranslator(source="hi", target="en").translate(text)
            return translated

        return text

    except Exception as e:
        print("Translation error:", e)
        return text
