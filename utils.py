import logging
import tiktoken
import re
import os
import json

from openai import OpenAI
from config import OPENAI_KEY
import fitz

_ENCODER = None

def get_openai_client():
    if not OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=OPENAI_KEY)

def get_logger(name: str = "rag"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def get_tokenizer():
    global _ENCODER 
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER

def num_tokens(text: str) -> int:
    enc = get_tokenizer()
    return len(enc.encode(text))

def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r'(?<![.?!:])\n(?![-•*\d]|[A-Z]{2,})', ' ',text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [ln.strip() for ln in text.split('\n')]
    text = "\n".join(lines)
    return text.strip()

def strip_page_headers_footers(text: str) -> str:
    lines = text.split('\n')
    filtered = []
    patterns = [
        re.compile(r'(?i)^\s*Fogarty Inspection Services Group\s*$'),
        re.compile(r'(?i)^\s*HomeBuyer\s*$'),
        re.compile(r'(?i)^\s*\d{1,6}\s+.*\b(Lane|Street|St\.|Road|Rd\.|Drive|Dr\.|Court|Ct\.|Avenue|Ave\.|Boulevard|Blvd\.)\b.*$'),
        re.compile(r'(?i)^\s*Page\s+\d+\s+of\s+\d+\s*$')
    ]
    for line in lines:
        line_stripped = line.strip()
        if any(pat.match(line_stripped) for pat in patterns):
            continue
        filtered.append(line)
    return '\n'.join(filtered)

# def split_sections(text: str) -> list[dict]:
#     SEC_RE = re.compile(r'^(?P<id>\d+)\.\s+(?P<title>.+)$')
#     SUB_RE = re.compile(r'^(?P<id>\d+\.\d+)\s+(?P<title>[A-Z0-9/\- &()]+)$')

#     sections = []
#     current_sec = None          
#     current_sub = None          
#     buffer = []
#     in_main_intro = False       # NEW: track “between section and first subsection”

#     for raw in text.split("\n"):
#         line = raw.strip()
#         if not line:
#             continue

#         m_sub = SUB_RE.match(line)
#         if m_sub:
#             if current_sub and buffer:
#                 sections.append({
#                     "section_id":   current_sec[0] if current_sec else None,
#                     "section_title":current_sec[1] if current_sec else None,
#                     "subsection_id":   current_sub[0],
#                     "subsection_title": current_sub[1],
#                     "content": "\n".join(buffer).strip()
#                 })
#                 buffer = []

#             # if we were in main-intro (no subsection yet), DROP that buffer
#             if in_main_intro:
#                 buffer = []
#                 in_main_intro = False

#             # start a new subsection (keep header line inside content for context)
#             current_sub = (m_sub.group("id"), m_sub.group("title"))
#             buffer.append(line)
#             continue

#         m_sec = SEC_RE.match(line)
#         if m_sec:
#             if current_sub and buffer:
#                 sections.append({
#                     "section_id":   current_sec[0] if current_sec else None,
#                     "section_title":current_sec[1] if current_sec else None,
#                     "subsection_id":   current_sub[0],
#                     "subsection_title": current_sub[1],
#                     "content": "\n".join(buffer).strip()
#                 })
#                 buffer = []

#             current_sec = (m_sec.group("id"), m_sec.group("title"))
#             current_sub = None
#             buffer = []         
#             in_main_intro = True
#             continue

#         if current_sub:
#             buffer.append(line)
#         else:
#             # we are either before the first section, or in main-intro so IGNORE
#             pass

#     if current_sub and buffer:
#         sections.append({
#             "section_id":   current_sec[0] if current_sec else None,
#             "section_title":current_sec[1] if current_sec else None,
#             "subsection_id":   current_sub[0],
#             "subsection_title": current_sub[1],
#             "content": "\n".join(buffer).strip()
#         })

#     return sections

def split_and_aggregate_subsections(input_src: str) -> list[dict]:
    """
    Parse PDF and aggregate all text belonging to the same base subsection ID (e.g., '4.1')
    into a single output entry. This ensures exactly one entry per unique X.Y subsection id.

    Returns: list of dicts:
      {
        "section_id": "4",
        "section_title": "Roofing",
        "subsection_id": "4.1",
        "subsection_title": "Roof Covering",
        "content": "full text for this subsection (concatenated across page breaks)"
      }
    """

    SEC_RE = re.compile(r'^(?P<id>\d+)\.\s+(?P<title>.+)$')
    SUB_RE = re.compile(r'^(?P<id>\d+\.\d+)\s+(?P<title>.+\S)$')

    # Decide whether input_src is a path to a PDF file or raw text
    if isinstance(input_src, str) and os.path.isfile(input_src):
        # It's a real PDF path -> extract text from PDF pages
        doc = fitz.open(input_src)
        pages = [p.get_text("text") for p in doc]
        text = "\n".join(pages)
    else:
        # Treat input_src as already-extracted raw text (safe default)
        text = input_src

    merged = {}  # base_id -> {section_id, section_title, subsection_title, content_lines}
    current_section = None
    current_base = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m_sub = SUB_RE.match(line)
        if m_sub:
            base = m_sub.group("id")
            title = m_sub.group("title").strip()
            # infer parent section if missing
            if current_section is None:
                inferred = base.split(".")[0]
                current_section = (inferred, None)
            if base not in merged:
                merged[base] = {
                    "section_id": current_section[0] if current_section else None,
                    "section_title": current_section[1] if current_section else None,
                    "subsection_id": base,
                    "subsection_title": title,
                    "content_lines": []
                }
                # include header once at start
                merged[base]["content_lines"].append(line)
            else:
                # repeated occurrence (e.g. same header repeated on new page):
                # add a small page-break marker so content parts remain separable.
                merged[base]["content_lines"].append("\n--PAGE-BREAK--\n")
                # if title differs from first captured, also record it for clarity
                if title and title != merged[base]["subsection_title"]:
                    merged[base]["content_lines"].append(line)
            current_base = base
            continue

        m_sec = SEC_RE.match(line)
        if m_sec:
            current_section = (m_sec.group("id"), m_sec.group("title").strip())
            continue

        # regular text lines appended to the currently active subsection
        if current_base:
            merged[current_base]["content_lines"].append(line)
        else:
            # text before first subsection (intro) is ignored
            pass

    # finalize aggregated entries (sorted by numeric subsection id)
    entries = []
    def sort_key(x):
        major, minor = x.split(".")
        return (int(major), float(x))
    for base in sorted(merged.keys(), key=lambda x: (int(x.split(".")[0]), float(x))):
        e = merged[base]
        content = "\n".join(e["content_lines"]).strip()
        entries.append({
            "section_id": e["section_id"],
            "section_title": e["section_title"],
            "subsection_id": e["subsection_id"],
            "subsection_title": e["subsection_title"],
            "content": content
        })
    return entries

def chunk_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    enc = get_tokenizer()
    tokens = enc.encode(text)
    total_tokens = len(tokens)
    if total_tokens <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunk_text = chunk_text.strip()
        chunks.append(chunk_text)
        if end == total_tokens:
            break
        start += max_tokens - overlap
    return chunks

def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def read_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def write_json(file_path: str, obj):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        return None