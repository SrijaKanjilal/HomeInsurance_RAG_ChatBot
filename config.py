# imports and environment setup
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_KEY:
    print(f"OpenAI API Key successfully retrieved.")
else:
    print("OPENAI_API_KEY environment variable not found. Please ensure it is set correctly in your environment.")

# Paths
DATA_DIR = "D:/Projects/RAG/HomeInsurance_RAG_ChatBot/data/"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
INPUT_PDF = "D:/Projects/RAG/HomeInsurance_RAG_ChatBot/data/raw/SeanFogartyHomeInspectionReport.pdf"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# Model parameters
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
TEMPERATURE = 0.2
MAX_TOKENS = 500

# Pipeline parameters
MAX_TOKENS = 500
CHUNK_OVERLAP = 50
TOP_K = 10
COLLECTION_NAME = "home_insurance_docs"
BATCH_SIZE = 128    