# Hindi PDF Semantic Search

A complete, intelligent pipeline to OCR Hindi PDF books and search them semantically using advanced AI embeddings — in Hindi, Hinglish, or English.

You ask a question in natural language (Devanagari, romanized Hindi, or English), the system finds the most relevant paragraphs across all your books with precise semantic understanding, flags questions that aren't actually about your books, and highlights exactly where the answer appears on the original page image.

---

## Features

- ✅ Convert any Hindi PDF into searchable text via Tesseract OCR (400 DPI high-quality scanning)
- ✅ Semantic (meaning-based) search powered by `BAAI/bge-m3` multilingual embeddings (1024-dim, superior Hindi understanding)
- ✅ **Multi-script query input** — ask in Devanagari, Hinglish ("manushya jivan ka dhyey kya he"), or English; romanized queries are transliterated to Devanagari before embedding, since the index itself is built entirely in Devanagari
- ✅ **Off-topic detection** — queries unrelated to the indexed books (e.g. general chit-chat) are caught by a relevance gate and shown a clear message, instead of returning a misleading low-confidence match
- ✅ Adaptive result scoring — returns exactly as many results as the question deserves (no fixed limit)
- ✅ Fuzzy OCR text matching — handles OCR drift between indexing and display time for reliable highlighting
- ✅ Visual highlighting of matched text on the original book page (cropped paragraph zoom or full-page overlay)
- ✅ Dark-themed Streamlit web UI with sidebar controls for fine-tuning search behavior
- ✅ CLI search tool with OpenCV image viewer for terminal-based searches
- ✅ Intelligent page filtering — indexes only content pages (filters out contact/publisher info)

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Web UI | Streamlit | Fast prototyping, built-in caching, dark theme support |
| PDF → Images | pdf2image + Poppler | Reliable PDF parsing, 400 DPI per-page conversion |
| OCR | Tesseract (`hin` language pack) | Industry-standard Hindi text recognition |
| Embeddings | `BAAI/bge-m3` (1024-dim) | Superior multilingual quality, no prefix needed, high Hindi accuracy |
| Transliteration | `indic-transliteration` (ITRANS scheme) | Converts romanized Hindi queries to Devanagari before search |
| Vector DB | Qdrant (Docker) | Fast cosine similarity search, local persistence, clean API |
| Image Processing | OpenCV + Pillow | Word-level coordinate extraction for precise highlighting |
| ML Framework | PyTorch (via SentenceTransformers) | Automatic GPU support if available, reliable inference |

---

## Project Structure

```
hindi_pdf_search/
├── pdfs/                   # Place your Hindi PDF books here
├── images/                 # Auto-generated: page images per book
│   └── <book_name>/
│       ├── page_001.jpg
│       └── ...
├── json/                   # Auto-generated: OCR text per book
│   └── <book_name>.json
├── qdrant_db/              # Local Qdrant vector database (Docker persisted)
│
├── config.py               # All settings: paths, model, OCR language, Qdrant
├── convert_pdf.py          # PDF pages → JPG images (confirm: does this also run OCR? see note below)
├── build_index.py          # JSON text → vector embeddings → Qdrant
├── run_all.py              # Runs the pipeline steps automatically in sequence
├── search_books.py         # CLI search with full-page image viewer
├── app.py                  # Streamlit web search UI (recommended — has off-topic detection + multi-script input)
│
├── requirements.txt        # All Python dependencies
├── check_setup.py          # Diagnostics: verify all dependencies are working
├── check_points.py         # Verify Qdrant collection contents
├── diagnose.py             # Additional diagnostics
├── recreate_test.py        # One-off utility: test Qdrant collection recreation
│
└── README.md               # This file
```

> **⚠️ To verify:** the previous README referenced a separate `extract_text.py` for the OCR step, but that file isn't present in the actual project folder. Confirm whether `convert_pdf.py` now handles both PDF→image conversion *and* OCR, or whether a file is missing/renamed — update `run_all.py`'s step list and this README's pipeline description accordingly once confirmed.

---

## Prerequisites

Install the following before running the pipeline:

### 1. Python packages

A `requirements.txt` is included. Install the pinned ML packages first (they need a specific CPU index URL on Windows), then the rest:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

> **Why pinned versions?** Letting `torch`/`transformers`/`sentence-transformers` float to "latest" has repeatedly pulled in unstable/bleeding-edge builds that break on import (missing attributes, broken internal imports). `requirements.txt` pins `torch==2.4.1`, `torchvision==0.19.1`, `transformers==4.44.2`, `accelerate==0.34.2`, and `sentence-transformers==3.1.1` together as a known-working combination.

### 2. Tesseract OCR (Windows)

Download from: https://github.com/UB-Mannheim/tesseract/wiki

- Install to: `C:\Program Files\Tesseract-OCR\`
- During install, select the **Hindi** language data (`hin`)
- Update `TESSERACT_CMD` in `config.py` if installed elsewhere

### 3. Poppler (Windows)

Download from: https://github.com/oschwartz10612/poppler-windows/releases

- Extract to match your `POPPLER_PATH` in `config.py`
- Update `POPPLER_PATH` in `config.py` if extracted elsewhere

### 4. Qdrant (Docker)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Qdrant must be running before you run `build_index.py` or `app.py`.

---

## Quick Start

### Step 1 — Activate Virtual Environment

```bash
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` appear at the start of your terminal prompt.

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

> **If `streamlit.exe` or other `.exe` files get blocked** by a "This file has been blocked by an Application Control policy" error, run the tool as a Python module instead, which usually routes around it: `python -m streamlit run app.py` instead of `streamlit run app.py`.

### Step 2 — Add your PDFs

Place your Hindi PDF books inside the `pdfs/` folder.

### Option A — Run the full pipeline automatically

```bash
python run_all.py
```

### Option B — Run steps individually

**Convert PDFs to images (and OCR — see note above about confirming this)**

```bash
python convert_pdf.py
```

**Build the search index**

```bash
python build_index.py
```

Splits each page into paragraph-aware chunks, generates multilingual embeddings, and uploads everything to Qdrant. **This always deletes and recreates the Qdrant collection from scratch.**

> First run downloads the embedding model (~1.5 GB for BGE-M3, one-time only).

### Step 3 — Search your books

**Option A — Web UI (recommended)**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Dark-themed interface, multi-script query input (Devanagari/Hinglish/English), off-topic detection, and highlighted page images.

**Option B — Command-line search**
```bash
python search_books.py
```
Interactive terminal loop. Type a query, see top results printed, then choose which page image to open in a viewer window with the matching paragraph highlighted in red.

> **Note:** `search_books.py` currently has its own hard-coded `HARD_MIN = 0.25` and does not yet include the off-topic relevance gate that `app.py` has. Worth syncing the two if you want consistent behavior between the CLI and web versions.

---

## Adding New Books

1. Put the new PDF in `pdfs/`
2. Run `python convert_pdf.py` (already-processed books are skipped)
3. Run `python build_index.py` (this rebuilds the full Qdrant collection from scratch)
4. Restart `app.py`

Or simply:
```bash
python run_all.py
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `OCR_LANG` | `hin` | Tesseract language code. Change to `guj` for Gujarati, `eng` for English |
| `DPI` | `400` | Image resolution for PDF conversion. Higher = better OCR, slower |
| `TESSERACT_CMD` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Path to Tesseract binary |
| `POPPLER_PATH` | `D:\poppler-26.02.0\Library\bin` | Path to Poppler `bin` folder |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence embedding model (higher quality than e5-small) |
| `EMBEDDING_DIM` | `1024` | Vector dimension (must match model) |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `book_paragraphs` | Qdrant collection name |
| `MAX_CHUNK_SIZE` | `500` | Max characters per paragraph-aware section chunk |
| `MIN_CHUNK_LENGTH` | `25` | Drop chunks shorter than this (OCR noise) |
| `KEEP_KEYWORDS` | `["प्रश्नकर्ता", "दादाश्री"]` | Only index pages containing these keywords |

---

## Diagnostics

Check that all dependencies are installed and working:

```bash
python check_setup.py
```

Verify your Qdrant collection has data:

```bash
python check_points.py
```

If `check_points.py` (or the Qdrant dashboard at `http://localhost:6333/dashboard#/collections`) shows **0 points**, the index was never actually built — run `python build_index.py`.

---

## Search UI Controls (Sidebar, in `app.py`)

| Control | Range | Default | Effect |
|---|---|---|---|
| **Answer grouping gap** | 0.02–0.30 | 0.12 | How many results appear per query. Smaller = only closest matches, Larger = includes more loosely related results |
| **Minimum score cutoff** | 0.10–0.60 | 0.40 | Drop individual results below this similarity score (filters out noise among otherwise-relevant results) |
| **View mode** | 2 options | Cropped | Cropped Paragraph shows zoomed text block; Full Page Highlighted overlays box on entire page |

**Score interpretation:**
- 🟢 **0.7–1.0** = Excellent semantic match
- 🟡 **0.5–0.7** = Good relevant match
- 🟠 **0.3–0.5** = Broad or weak correlation
- 🔴 **Below the minimum cutoff** = Noise or not relevant

### Off-topic detection (separate from the slider above)

`RELEVANCE_THRESHOLD` (currently `0.4`, set in `app.py`, not adjustable from the sidebar) is a **separate gate** from "Minimum score cutoff." It answers a different question:

- `RELEVANCE_THRESHOLD` → "Is this query even about the books at all?" Checked once, against the single best-scoring candidate, **before** any per-result filtering happens.
- `hard_min` (the sidebar slider) → "Among results that *are* about the books, which are strong enough to show?"

If the best candidate's score is below `RELEVANCE_THRESHOLD`, the app shows a Hindi message explaining the query seems unrelated to the Dada Bhagwan books, regardless of what the sidebar slider is set to.

### Multi-script query input

Queries typed in plain Latin script (detected via `looks_romanized()`) are transliterated from romanized Hindi (ITRANS-style spelling) to Devanagari via the `indic-transliteration` library before being embedded. This matters because the index itself is built entirely from Devanagari OCR text — searching with a mismatched script otherwise produces much weaker similarity scores. Devanagari queries pass through unchanged.

---

## Currently Indexed Books

| Book | Pages | Content |
|---|---|---|
| manav-dharma_hindi | 42 | Prashnkarta/Dadashree dialogue on human ethics |
| pratikraman_hindi | 114 | Prashnkarta/Dadashree dialogue on self-reflection |

> Update this table after re-running `build_index.py` with new books — point counts can be checked at `http://localhost:6333/dashboard#/collections`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Virtual environment not activating | Make sure you're in the project folder and run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `(venv)` not showing in prompt | The virtual environment didn't activate. Try: `python -m venv venv` then `.\venv\Scripts\Activate.ps1` |
| `ModuleNotFoundError` when running scripts | Make sure `(venv)` shows in your prompt. If not, activate venv first |
| `ImportError: DLL load failed` / "blocked by an Application Control policy" | Windows Smart App Control or a managed-device policy is blocking an unsigned `.exe`/`.dll`. Try running via `python -m <tool>` instead of the `.exe` directly; on a managed/work device this needs IT to allowlist the path |
| `AttributeError: module 'torch' has no attribute '_utils'` or `NameError: name 'nn' is not defined` | Broken/mismatched `torch`/`transformers` versions — delete `venv`, recreate it, and install the exact pinned versions from `requirements.txt` |
| `ModuleNotFoundError: No module named 'torchvision'` | Run `python -m pip install torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu` |
| Qdrant connection failed | Start Docker and run: `docker run -d -p 6333:6333 qdrant/qdrant` |
| Streamlit command not found / blocked | Activate venv and verify with `pip list`; try `python -m streamlit run app.py` |
| Tesseract not found | Check `TESSERACT_CMD` in `config.py` points to your installation |
| Poppler not found | Check `POPPLER_PATH` in `config.py` points to your extraction |
| Web UI won't start | Check port 8501 is available; stop any existing Streamlit instance |
| Search returns zero results at any confidence level | Check `http://localhost:6333/dashboard#/collections` — if points count is 0, `build_index.py` was never actually run (or ran against an empty/broken environment); re-run it |
| Search finds nothing for a Hinglish/romanized query specifically | Confirm `looks_romanized()` and `transliterate_to_devanagari()` are actually being called inside `run_search()` — these were written once but not wired in initially; check the `search_query` variable is what actually gets passed to `model.encode()` |
| An off-topic query (e.g. unrelated small talk) still returns a confident-looking answer | Check `RELEVANCE_THRESHOLD` in `app.py` is being checked against `best_score` returned from `run_search()`, not against the post-filter `results` list — if `hard_min` already filtered everything out, `results` will be empty and the off-topic branch needs `best_score`, not `results`, to detect this case |

---

## Performance & Timing

### First Run (Complete Pipeline)

| Step | Time | Why |
|---|---|---|
| Download BGE-M3 model | 5-15 min | ~1.5 GB one-time download |
| Convert PDFs → Images | 2-5 min | 400 DPI per-page conversion |
| Run Hindi OCR | 10-30 min | Tesseract text extraction per page |
| Generate embeddings | 5-10 min | 1024-dim vectors per paragraph |
| Upload to Qdrant | 1-2 min | Index storage |
| **Total First Run** | **25-60 min** | One-time investment |

### Subsequent Runs
- **Just searching** (no new books): **3-5 seconds** — Model and index cached ✅
- **Adding new book**: **5-15 minutes** — Only new book processed, old books skipped ✅

### Why It's Fast After First Run
- ✅ Model cached in memory (`@st.cache_resource`)
- ✅ Images skipped if already exist (checked in `convert_pdf.py`)
- ✅ Qdrant persisted on disk (only loaded once)
- ✅ Image highlighting cached (`@st.cache_data`)

---

## Notes

- The embedding model is downloaded automatically on first run of `build_index.py` or `app.py` (BGE-M3 ~1.5GB, one-time only)
- OCR quality depends on PDF scan quality. Higher DPI generally improves results, but increases processing time
- Rebuilding the index always recreates the Qdrant collection from scratch — this ensures a clean, deduplicated index
- The project supports any Indic language supported by Tesseract — just change `OCR_LANG` in `config.py` and install the language pack
- Only pages containing `प्रश्नकर्ता` (Prashnkarta) or `दादाश्री` (Dadashree) keywords are indexed — pages with contact numbers, publisher info, and cover pages are automatically filtered out
- The fuzzy OCR matching algorithm (`difflib.SequenceMatcher` with `autojunk=False`) handles the fact that re-OCRing a page at display time rarely produces byte-identical output to the OCR captured at indexing time — this ensures highlighting is reliable despite OCR drift
- BGE-M3 requires no query/passage prefixes (unlike `multilingual-e5-small` which required them) — just use plain text
- Hinglish/romanized-Hindi transliteration quality depends on the `indic-transliteration` ITRANS scheme, which expects fairly consistent spelling conventions. Casual/inconsistent Hinglish typing may transliterate imperfectly — a neural alternative (`ai4bharat-transliteration`, the IndicXlit model) is a possible upgrade if this becomes a real problem with actual users


#   h i n d i _ p d f _ s e a r c h  
 