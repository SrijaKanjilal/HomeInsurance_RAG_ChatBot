import argparse
import re
from collections import defaultdict

import chromadb
import openai

from config import LLM_MODEL, TEMPERATURE, MAX_TOKENS, CHROMA_DIR, COLLECTION_NAME, TOP_K, EMBED_MODEL
from utils import get_openai_client, get_logger, get_tokenizer, num_tokens

logger = get_logger("query")

def initialize_clients():
    logger.info("Initializing ChromaDB client and OpenAI client")
    chromadb_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)
    if collection is None:
        raise ValueError(f"ChromaDB collection '{COLLECTION_NAME}' does not exist.")
    logger.info(f"Connected to ChromaDB collection '{COLLECTION_NAME}'")
    logger.info("Size of collection: {}".format(collection.count()))
    if collection.count() == 0:
        logger.warning(f"Index is empty, run build_index.py")
    openai_client = get_openai_client()
    tokenizer = get_tokenizer()
    return (collection, openai_client, tokenizer)

def normalize_query(user_text:str) -> str:
    logger.info("Normalizing user query")
    user_text = user_text.strip()
    user_text = re.sub(r'\s+', ' ', user_text)
    return user_text

def build_metadata_filter(q: str) -> dict | None:
    text = q.strip()
    if not text:
        return None

    where: dict = {}

    # Subsection like 7.1, 10.3, etc.
    m_sub = re.search(r'\b(\d+\.\d+)\b', text)
    if m_sub:
        where["subsection_id"] = m_sub.group(1)

    # Explicit "section 7" style reference
    m_sec = re.search(r'\bsection\s+(\d+)\b', text, flags=re.IGNORECASE)
    if m_sec:
        where["section_id"] = m_sec.group(1)

    m_pdf = re.search(r'([A-Za-z0-9_\-][A-Za-z0-9_\-\.]*\.pdf)\b', text, flags=re.IGNORECASE)
    if m_pdf:
        where["source_pdf"] = m_pdf.group(1)
    return where or None


# def retrieve_top_k(collection, query: str, k: int, where: dict | None = None) -> list[dict]:
def retrieve_top_k(collection, query: str, k: int, where: dict | None = None, query_embedding: list[float] | None = None) -> list[dict]:
    if query_embedding is not None:
        retrieved = collection.query(query_embeddings=[query_embedding], n_results=k, include=["documents","metadatas","distances"], where=where)
    else:
        retrieved = collection.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"], where=where)
    
    # retrieved = collection.query(
    #     query_texts=[query], 
    #     n_results=k, 
    #     include=["documents","metadatas","distances"], 
    #     where=where)
    # ids = retrieved["ids"][0]
    documents = retrieved["documents"][0]
    metadatas = retrieved["metadatas"][0]
    distances = retrieved["distances"][0]

    ids = None
    if "ids" in retrieved and retrieved["ids"]:
        ids = retrieved["ids"][0]
    elif "data" in retrieved and isinstance(retrieved["data"], list) and len(retrieved["data"]) > 0:
        first_data = retrieved["data"][0]
        if isinstance(first_data, dict) and "ids" in first_data:
            ids = first_data["ids"]
    if not ids:
        ids = []
        for i, m in enumerate(metadatas):
            if isinstance(m, dict):
                candidate = m.get("id") or m.get("subsection_id") or m.get("doc_id") or m.get("source")
            else:
                candidate = None
            ids.append(candidate if candidate is not None else f"idx_{i}")

    results = []
    for id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        logger.info(f"Retrieved chunk ID: {id}, Distance: {dist:.4f}, Section: {meta.get('section_title')}, Subsection: {meta.get('subsection_title')}")
        results.append({"id": id, "text": doc, "meta": meta, "distance": dist})

    if not results:
        logger.info("No results retrieved from ChromaDB.")
        return []
    
    logger.info("Top distances: " + ", ".join(f"{x['distance']:.3f}" for x in results[:5]))    
    return results

def group_chunks(retrieved: list[dict], tokenizer, max_context_tokens:int) -> tuple[list[dict], int]:
    if not retrieved:
        return ([], 0)

    groups = defaultdict(list)
    for r in retrieved:
        m = r["meta"]
        key = (
            m.get("source_pdf"),
            m.get("section_id"),
            m.get("section_title"),
            m.get("subsection_id"),
            m.get("subsection_title")
        )
        groups[key].append(r)
    
    grouped = []
    for key, items in groups.items():
        items = sorted(items, key=lambda x: x.get("distance", 1e9))

        seen_texts = set()
        merged_parts = []
        for i in items:
            txt = (i.get("text") or "").strip()
            if not txt or txt in seen_texts:
                continue
            seen_texts.add(txt)
            merged_parts.append(txt)

        if not merged_parts:
            continue
        
        merged_text = "\n".join(merged_parts).strip()
        best = items[0]
        meta = best.get("meta", {}).copy()
        score = best.get("distance", 1e9)
        source_pdf, sec_id, sec_title, sub_id, sub_title = key
        
        label_sec = f"Section {sec_id}" if sec_id else "Section ?"
        label_sub = f"{sub_id}" if sub_id else ""
        label_title = " — ".join([t for t in [sec_title, sub_title] if t])
        header_bits = [label_sec]
        if label_sub:
            header_bits.append(label_sub)
        if label_title:
            header_bits.append(label_title)
        if source_pdf:
            header_bits.append(f"({source_pdf})")
        header = " — ".join(header_bits)

        grouped.append({
            "header": header,
            "text": merged_text,
            "meta": meta,
            "score": score,
        })

    if not grouped:
        return [], 0
    
    grouped.sort(key=lambda x: x["score"])
    context_blocks, total_tokens = [], 0
    for g in grouped:
        t = num_tokens(g["header"] + "\n" + g["text"])
        if total_tokens + t > max_context_tokens:
            continue
        context_blocks.append(g)
        total_tokens += t

    return context_blocks, total_tokens

def build_prompts(question:str, context_blocks:list[dict]) -> list[dict]:
    system = ("You are a helpful assistant to an underwriter. "
              "Based on the inspection reports used for input, help answer any queries the underwriter may have about the property. " 
              "Answer ONLY based on the 'Context' derived from PDF document(s) given as input. "
              "If 'Context' is insufficient or no relevant fact is found to answer the question, say 'I don't have enough information to answer this question'. "
              "Do not hallucinate or infer based on external information. Do not invent any facts. "
              "Answers to the questions should be based on the context document only. " 
              "Keep the placeholders for personal information masking such as (<<NAME_1>>) intact. "
              "Provide correct citations to the context in your answers in the format [source: <section id>, subsection id (if any)]. "
              "Use a concise and professional tone in your responses. "
    )

    parts = []
    for block in context_blocks:
        header = block.get("header", "").strip()
        text   = (block.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"{header}\n{text}")
    
    context_str = "\n\n".join(parts).strip() if parts else "No relevant context found."

    user = (
            f"Question:\n{question.strip()}\n\n"
            f"Context:\n{context_str}\n\n"
            "Answer using only the context above."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    return messages

def format_sources(context_blocks: list[dict]) -> list[dict]:
    sources = []
    for b in context_blocks:
        m = b.get("meta", {}) or {}
        sources.append({
            "section_id": m.get("section_id"),
            "subsection_id": m.get("subsection_id"),
            "section_title": m.get("section_title"),
            "subsection_title": m.get("subsection_title"),
            "source_pdf": m.get("source_pdf"),
            "distance": b.get("score"),
        })
    return sources

def generate_answer(question: str, k: int = TOP_K, context_budget_tokens: int = 3000, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> dict:
    collection, openai_client, tokenizer = initialize_clients()

    q = normalize_query(question)
    if not q:
        return {
            "answer": "Please provide a non-empty question.",
            "sources": [],
            "context_tokens": 0,
            "used_k": 0,
            "abstained": True,
        }
    where = build_metadata_filter(q)
    # compute query embedding with the same model used for indexing
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[q])
    query_emb = resp.data[0].embedding
    logger.info("Query emb dim: %d", len(query_emb))
    hits = retrieve_top_k(collection, q, k, where=where, query_embedding=query_emb)
    # hits = retrieve_top_k(collection, q, k, where=where)

    if not hits:
        logger.info("No hits returned by retriever.")
        return {
            "answer": "I don't have enough information in the report.",
            "sources": [],
            "context_tokens": 0,
            "used_k": 0,
            "abstained": True,
        }

    context_blocks, total_tokens = group_chunks(hits, tokenizer, context_budget_tokens)
    if not context_blocks:
        logger.info("No context blocks after grouping/token budgeting.")
        return {
            "answer": "I don't have enough information in the report.",
            "sources": [],
            "context_tokens": 0,
            "used_k": len(hits),
            "abstained": True,
        }

    messages = build_prompts(q, context_blocks)    
    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not resp.choices or not resp.choices[0].message or not resp.choices[0].message.content:
        return {
        "answer": "I couldn't generate an answer.",
        "sources": [],
        "context_tokens": total_tokens,
        "used_k": len(hits),
        "abstained": True,
    }

    
    answers = resp.choices[0].message.content.strip()

    sources = format_sources(context_blocks)

    return {
        "answer": answers,
        "sources": sources,
        "context_tokens": total_tokens,
        "used_k": len(hits),
        "abstained": False,
    }

def main():
    print("\n--- Home Inspection Report Query ---\n")
    question = input("Enter your question about the inspection report: ").strip()
    if not question:
        print("Please enter a non-empty question.")
        return

    k = TOP_K   
    context_budget = 3000

    print("\nProcessing your query...\n")

    try:
        result = generate_answer(
            question=question,
            k=k,
            context_budget_tokens=context_budget,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    if result["sources"]:
        for i, s in enumerate(result["sources"], 1):
            sec = s.get("section_id") or "?"
            sub = s.get("subsection_id") or ""
            st  = s.get("section_title") or ""
            sst = s.get("subsection_title") or ""
            pdf = s.get("source_pdf") or ""
            dist = s.get("distance")
            dist_str = f"{dist:.3f}" if isinstance(dist, (int, float)) else "n/a"
            label_bits = [f"Section {sec}"]
            if sub: label_bits.append(str(sub))
            title = " — ".join([t for t in [st, sst] if t])
            if title: label_bits.append(title)
            if pdf: label_bits.append(f"({pdf})")
            print(f"{i}. {' — '.join(label_bits)}  [distance: {dist_str}]")
    else:
        print("(none)")

    print(f"\nContext tokens used: {result.get('context_tokens', 0)}")
    print(f"Top-k retrieved: {result.get('used_k', 0)}")
    if result.get("abstained"):
        logger.info("Abstained due to insufficient context.")


if __name__ == "__main__":
    main()