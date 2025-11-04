import os
from pathlib import Path
import re
import hashlib
import time
from tqdm import tqdm

from config import OPENAI_KEY, EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME,CHUNK_OVERLAP,MAX_TOKENS, INPUT_PDF, PROCESSED_DATA_DIR, LLM_MODEL, BATCH_SIZE
from utils import get_openai_client, get_logger, normalize_whitespace, strip_page_headers_footers, split_and_aggregate_subsections, chunk_text, batched, write_json

import fitz  
import chromadb

logger = get_logger("build_index")
logger.info(f"Starting chroma db index building for document")
logger.info(f"Input pdf: {INPUT_PDF}")
logger.info(f"Chroma directory: {CHROMA_DIR}")
logger.info(f"Collection name: {COLLECTION_NAME}")
logger.info(f"Embedding model: {EMBED_MODEL}")
logger.info(f"Chunk tokens: {MAX_TOKENS}")
logger.info(f"Chunk overlap: {CHUNK_OVERLAP}")
logger.info(f"OpenAI LLM model: {LLM_MODEL}")


def load_pdf(file_path: str) -> str:
    logger.info(f"Loading PDF document from {file_path}")
    text_parts = []
    try:
        with fitz.open(file_path) as doc:
            for page in tqdm(range(len(doc)), desc="Reading PDF pages", unit="page"):
                page_text = doc[page].get_text("text")  
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error loading PDF with PyMuPDF: {e}")
        raise

def clean_text(text: str) -> str:
    logger.info("Cleaning extracted text")
    logger.info("Initial text length: {}".format(len(text)))
    text = normalize_whitespace(text)
    text = strip_page_headers_footers(text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    logger.info("Cleaned text length: {}".format(len(text)))
    return text

def extract_intro_text(cleaned_text: str) -> tuple[str, str]:
    logger.info("Extracting introduction text")
    intro_end_match = re.search(r'(?m)^[1-9]\d*\.\s+[A-Z]', cleaned_text)
    if intro_end_match:
        intro_text = cleaned_text[:intro_end_match.start()]
        main_text = cleaned_text[intro_end_match.start():]
    else:
        logger.warning("No numbered section headings found; treating entire document as main text")
        intro_text = ""
        main_text = cleaned_text
    
    intro_text = intro_text.strip()
    main_text = main_text.strip()
    logger.info(f"Introduction text length: {len(intro_text)} chars, Main body text length: {len(main_text)} chars")
    return (intro_text, main_text)

def mask_pii(text: str) -> tuple[str, dict]:
    logger.info("Masking PII in text")
    pii_map = {}
    counter = {"ADDRESS": 0, "NAME": 0, "PHONE": 0, "DATE": 0}

    patterns = {
        "ADDRESS": re.compile(
            r'\b\d{1,6}\s+[A-Z][\w\s.,-]+(?:Lane|Street|St\.|Road|Rd\.|Avenue|Ave\.|Drive|Dr\.|Court|Ct\.|Boulevard|Blvd\.)\b.*',
            re.IGNORECASE
        ),
        "NAME": re.compile(r'\b(Sean\s+Fogarty|HomeBuyer|Client\s*:\s*[A-Z][a-z]+\s+[A-Z][a-z]+)\b', re.IGNORECASE),
        "PHONE": re.compile(r'\b(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'),
        "DATE": re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b')
    }
    masked_text = text

    for label, pattern in patterns.items():
        matches = list(pattern.finditer(masked_text))
        for m in matches:
            original = m.group(0)
            counter[label] += 1
            placeholder = f"<<{label}_{counter[label]}>>"
            pii_map[placeholder] = original
            masked_text = masked_text.replace(original, placeholder)

    return masked_text, pii_map

def split_text_sections(masked_text:str) -> list[dict]:
    logger.info("Splitting text into sections and subsections")
    sections = split_and_aggregate_subsections(masked_text)
    logger.info(f"Total sections/subsections extracted: {len(sections)}")
    return sections

def section_to_chunks(sections: list[dict], chunk_overlap: int, max_tokens:int, source_pdf:str) -> list:
    all_chunks = []
    for sec in sections:
        sec_id = sec.get("section_id")
        sec_title = sec.get("section_title")
        sub_id = sec.get("subsection_id")
        sub_title = sec.get("subsection_title")
        content = sec.get("content", "")

        if not content.strip():
            continue

        chunks = chunk_text(content, max_tokens, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.sha1(f"{source_pdf}|{sec_id}|{sub_id}|{i}".encode("utf-8")).hexdigest()
            chunk_metadata = {
                "section_id": sec_id,
                "section_title": sec_title,
                "subsection_id": sub_id,
                "subsection_title": sub_title,
                "chunk_index": i,
                "source_pdf": source_pdf

            }
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": chunk_metadata
            })
    return all_chunks

def embed_chunks(chunks:list[dict], batch_size: int) -> list[list[float]]:
    logger.info(f"Embedding {len(chunks)} chunks using model {EMBED_MODEL}")
    client = get_openai_client()
    embeddings = []
    texts = [chunk["text"] for chunk in chunks]
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch in tqdm(batched(texts, batch_size), total=total_batches, desc="Embedding batches", unit="batch"):
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    if embeddings:
        logger.info(f"Generated {len(embeddings)} embeddings (dim = {len(embeddings[0])})")
    else:
        logger.info("Generated 0 embeddings")
    return embeddings

def update_collection(chunks: list[dict], embeddings: list[list[float]]):
    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    count = collection.count()
    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' updated. Total documents in collection: {count}")

def save_artifacts(cleaned_text:str, sections:list[dict], pii_map:dict, chunks: list[dict], embeddings: list[list[float]]):
    write_json(os.path.join(PROCESSED_DATA_DIR, "cleaned_text.json"), {"text": cleaned_text})
    write_json(os.path.join(PROCESSED_DATA_DIR, "pii_map.json"), pii_map)
    write_json(os.path.join(PROCESSED_DATA_DIR, "sections.json"), sections)
    write_json(os.path.join(PROCESSED_DATA_DIR, "chunks.json"), chunks)

    logger.info("Saved processed artifacts to disk")

def main():
    logger.info("Starting RAG index build pipeline")

    if not os.path.exists(INPUT_PDF):
        raise FileNotFoundError(f"Input PDF not found: {INPUT_PDF}")
    if not OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY not set.")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    logger.info(f"Input PDF: {INPUT_PDF}")
    logger.info(f"Embedding model: {EMBED_MODEL}, LLM model: {LLM_MODEL}")
    logger.info(f"Chunk size: {MAX_TOKENS}, overlap: {CHUNK_OVERLAP}")

    raw_text = load_pdf(INPUT_PDF)
    logger.info(f"Loaded {len(raw_text)} characters of raw text")

    cleaned_text = clean_text(raw_text)
    logger.info(f"Cleaned text length: {len(cleaned_text)}")

    intro_text, body_text = extract_intro_text(cleaned_text)
    write_json(os.path.join(PROCESSED_DATA_DIR, "intro_text.json"), {"text": intro_text})
    logger.info("Extracted intro and inspection body text")

    masked_text, pii_map = mask_pii(body_text)
    logger.info("PII masking completed")

    sections = split_text_sections(masked_text)
    logger.info(f"Extracted {len(sections)} sections/subsections")

    source_pdf = os.path.basename(INPUT_PDF)
    chunks = section_to_chunks(sections, CHUNK_OVERLAP, MAX_TOKENS, source_pdf)
    logger.info(f"Created {len(chunks)} total chunks")

    embeddings = embed_chunks(chunks, batch_size=BATCH_SIZE)
    logger.info("Embedding generation complete")

    update_collection(chunks, embeddings)
    logger.info("Vector store update complete")

    save_artifacts(cleaned_text, sections, pii_map, chunks, embeddings)

    logger.info("Index build completed successfully.")


if __name__ == "__main__":
    main()  



