# translation.py
import os
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

translator = None 

def get_translator():
    global translator
    if translator is None:
        translator = pipeline(
            "translation",
            model="ai4bharat/indictrans2-indic-en-1B",
            trust_remote_code=True,
            device=-1,  # CPU
            use_auth_token=HF_TOKEN
        )
    return translator

def english(text: str, chunk_size: int = 300) -> str:
    """Translate Hindi → English safely, even for large docs."""
    if not text.strip():
        return ""
    
    translator_instance = get_translator()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translations = []
    
    for chunk in chunks:
        try:
            translated = translator_instance(chunk)
            translations.append(translated[0]['translation_text'])
        except Exception as e:
            translations.append(f"[Error translating: {e}]")
    
    return " ".join(translations)
