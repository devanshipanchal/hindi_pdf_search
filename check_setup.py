import sys
import subprocess
from pathlib import Path

print("=" * 60)
print("DIAGNOSTICS SCRIPT")
print("=" * 60)

# 1. Check Python version
print(f"Python Version: {sys.version}")

# 2. Check imports
print("\nChecking Python packages:")
packages = [
    "streamlit", "pdf2image", "pytesseract", "PIL", "cv2",
    "sentence_transformers", "qdrant_client"
]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError as e:
        print(f"  [FAIL] {pkg} (Error: {e})")

# 3. Check Tesseract
print("\nChecking Tesseract OCR:")
try:
    from config import TESSERACT_CMD, OCR_LANG
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    version = pytesseract.get_tesseract_version()
    print(f"  [OK] Tesseract version: {version}")
    print(f"  [OK] Tesseract path: {TESSERACT_CMD}")
except Exception as e:
    print(f"  [FAIL] Tesseract check failed. Error: {e}")

# 4. Check Poppler
print("\nChecking Poppler:")
try:
    from config import POPPLER_PATH
    poppler_bin = Path(POPPLER_PATH) / "pdftoppm.exe"
    if poppler_bin.exists():
        print(f"  [OK] Poppler found at: {POPPLER_PATH}")
    else:
        print(f"  [FAIL] pdftoppm.exe not found at: {POPPLER_PATH}")
except Exception as e:
    print(f"  [FAIL] Poppler check failed. Error: {e}")

# 5. Check Qdrant Connection
print("\nChecking Qdrant Database:")
try:
    from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
    from qdrant_client import QdrantClient
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = client.get_collections()
    print(f"  [OK] Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"  [OK] Existing collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"  [FAIL] Could not connect to Qdrant. Error: {e}")

print("\n" + "=" * 60)
