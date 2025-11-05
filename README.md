## HomeInsurance_RAG_ChatBot

A small Retrieval-Augmented Generation (RAG) project that indexes a home inspection PDF into a Chroma vector DB and answers user questions grounded in that report using OpenAI embeddings + an LLM.

## Repo contents (key files)

* `build_index.py` — build/update Chroma index from PDF
* `query.py` — retrieval + answer generation
* `utils.py` — helpers (PDF extraction, splitting, chunking, masking)
* `config.py` — paths & model settings
* `app.py` — minimal Streamlit UI to produce answer to any question

## Requirements

* Python 3.10+
* Packages: `openai`, `chromadb`, `pymupdf`, `tiktoken`, `streamlit`

## Quick setup

1. Create & activate a venv and install dependencies.
2. Set `OPENAI_API_KEY` in your environment.
3. Put the PDF at the path referenced by `config.py` or update `INPUT_PDF`.

## To build the index in the vector DB

Run:

```
python build_index.py
```

## To query the database and produce the answer

Run:

```
python query.py
```

## To run the Streamlit UI

Run:

```
streamlit run app.py
```

## Config note

Check `config.py` for `INPUT_PDF`, `CHROMA_DIR`, `EMBED_MODEL`, and `LLM_MODEL`. Keep `EMBED_MODEL` consistent between indexing and querying.
