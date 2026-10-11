from pathlib import Path

# ============================================================
# BASE DIRECTORY — this is your project root folder
# ============================================================
BASE_DIR = Path(__file__).parent

# ============================================================
# FOLDER PATHS  (created automatically when pipeline runs)
# ============================================================
PDFS_DIR    = BASE_DIR / "pdfs"        # put your PDF books here
IMAGES_DIR  = BASE_DIR / "images"      # full page JPGs saved here
JSON_DIR    = BASE_DIR / "json"        # OCR text per page saved here
QDRANT_DIR  = BASE_DIR / "qdrant_db"   # local vector database saved here

# ============================================================
# OCR SETTINGS
# ============================================================
OCR_LANG = "hin"          # Hindi — change to "guj" for Gujarati
DPI      = 400            # higher = better OCR, but slower

# ============================================================
# WINDOWS PATHS  — adjust these to match your installation
# ============================================================
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH  = r"D:\poppler-26.02.0\Library\bin"

# ============================================================
# EMBEDDING MODEL  (downloads automatically on first run)
# ============================================================
EMBEDDING_MODEL = "BAAI/bge-m3"   # stronger multilingual model — supports Hindi, higher quality than e5-small
EMBEDDING_DIM   = 1024            # vector size for bge-m3

# ============================================================
# QDRANT SETTINGS
# ============================================================
QDRANT_HOST      = "localhost"
QDRANT_PORT      = 6333
COLLECTION_NAME  = "book_paragraphs"

# ============================================================
# CHUNKING
# ============================================================
MAX_CHUNK_SIZE   = 500   # max characters per section chunk (paragraph-aware splitting, not a hard cutoff)
MIN_CHUNK_LENGTH = 25     # दादाश्री turns shorter than this are dropped as OCR-noise fragments

# ============================================================
# PAGE FILTER — only keep pages that look like actual
# Prashnkarta/Dadashree dialogue content. Pages without either
# keyword (cover, publisher info, index, etc.) are discarded
# before an image or JSON entry is ever created for them.
# ============================================================
KEEP_KEYWORDS = ["प्रश्नकर्ता", "दादाश्री"]