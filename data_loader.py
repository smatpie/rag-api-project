from pathlib import Path
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import fitz

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    doc = fitz.open(path)
    full_text = ""
    
    for page in doc:
        extracted = page.get_text("text")
        if isinstance(extracted, str) and extracted:
            full_text += extracted + "\n"
            
    chunks = splitter.split_text(full_text)
    
    print(f"DEBUG: Successfully extracted {len(chunks)} chunks from PDF!")
    return chunks

def embed_texts(texts: list[str])-> list[list[float]]:
    response = client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [item.embedding for item in response.data]
