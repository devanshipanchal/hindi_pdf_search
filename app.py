"""
app.py — Modern, Premium Streamlit search interface for your Hindi PDF books.

HOW TO RUN:
    1. Make sure Docker is open and your Qdrant container is running
    2. Open terminal in your project folder
    3. Type:  streamlit run app.py
    4. Browser opens at http://localhost:8501
"""
import re
import difflib

import cv2
import pytesseract

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from pytesseract import Output
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TESSERACT_CMD,
    OCR_LANG,
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Minimum score the TOP result must reach for a query to count as
# "about this book collection" at all.
RELEVANCE_THRESHOLD = 0.4


def is_query_relevant(results):
    """True if the top result's score suggests this query actually
    belongs to the book collection."""
    if not results:
        return False
    return results[0].score >= RELEVANCE_THRESHOLD


# ============================================================
# ONE-TIME SETUP (CACHED)
# ============================================================

@st.cache_resource
def load_model():
    """Load the SentenceTransformer model once and keep it in cache."""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_client():
    """Connect to Qdrant Docker container once and keep the client in cache."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ============================================================
# TEXT NORMALIZATION HELPER
# ============================================================

def normalize(text):
    """Normalize text by stripping spaces to improve OCR word-matching."""
    return "".join(text.split())

def looks_romanized(text):
    """
    True if the query is plain Latin letters/numbers (no Devanagari).
    Used to decide whether we should ALSO try a Devanagari transliteration
    of the query — a Devanagari query doesn't need this extra step.
    """
    return bool(re.fullmatch(r"[a-zA-Z0-9\s\?\.,!']+", text.strip()))


def transliterate_to_devanagari(text):
    """
    Convert romanized Hindi (ITRANS-style, e.g. 'pratikraman') into
    Devanagari script. Returns None if transliteration fails or produces
    nothing useful — caller should fall back to the original query.
    """
    try:
        result = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return result if result and result.strip() else None
    except Exception:
        return None


def line_belongs_to_chunk(line_norm, text_norm, min_ratio=0.7, min_fuzzy_len=15):
    """
    True if this OCR'd line is (close to) part of the chunk text.

    Re-OCRing the page at display time rarely comes out byte-identical
    to the OCR captured at indexing time (punctuation/spacing drift),
    so an exact substring check silently drops lines and produces a
    highlight box that only covers part of the paragraph. This allows
    a fuzzy match instead: the line counts if most of its characters
    form a contiguous match somewhere inside the chunk text.

    Fuzzy matching is skipped for short lines (< min_fuzzy_len chars):
    a short running header/title line can coincidentally share one
    common word with the chunk (e.g. "प्रतिक्रमण", this book's own
    title, appearing in a page header) and cross the ratio threshold
    by accident. Short lines must match exactly instead.
    """
    if not line_norm:
        return False
    if line_norm in text_norm:
        return True
    if len(line_norm) < min_fuzzy_len:
        return False

    # autojunk=False: difflib's default autojunk heuristic treats any
    # character repeated often in a long string as "junk" and skips it
    # during matching — Devanagari matras repeat constantly, so with
    # autojunk left on, this silently breaks nearly every match.
    match = difflib.SequenceMatcher(None, line_norm, text_norm, autojunk=False).find_longest_match(
        0, len(line_norm), 0, len(text_norm)
    )
    return (match.size / len(line_norm)) >= min_ratio


# ============================================================
# IMAGE HIGHLIGHTING AND CROPPING
# ============================================================

@st.cache_data(show_spinner=False)
def get_highlighted_image(image_path, display_text, mode="Cropped Paragraph"):
    """
    Find where the matched sentences exist on the original page image.
    Mode:
      - "Cropped Paragraph": Crops the matching text block with a red border.
      - "Full Page Highlighted": Shows the complete page image with a red bounding box.
    """
    try:
        pil_image = Image.open(image_path)
    except Exception:
        return None

    # Retrieve word coordinates using Tesseract OCR
    data = pytesseract.image_to_data(
        pil_image,
        lang=OCR_LANG,
        config="--psm 6",
        output_type=Output.DICT,
    )

    line_groups = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_groups.setdefault(key, []).append(i)

    text_norm     = normalize(display_text)
    matched_boxes = []

    for indices in line_groups.values():
        line_text = " ".join(data["text"][i] for i in indices).strip()
        line_norm = normalize(line_text)

        # Check if this line is part of our matching display text
        if line_belongs_to_chunk(line_norm, text_norm):
            xs      = [data["left"][i] for i in indices]
            ys      = [data["top"][i] for i in indices]
            rights  = [data["left"][i] + data["width"][i] for i in indices]
            bottoms = [data["top"][i] + data["height"][i] for i in indices]
            matched_boxes.append([
                min(xs), min(ys),
                max(rights) - min(xs),
                max(bottoms) - min(ys),
            ])

    if not matched_boxes:
        # If OCR fails to pinpoint, fallback to original full page in Full mode, or None in Crop mode
        if mode == "Full Page Highlighted":
            cv_image = cv2.imread(image_path)
            if cv_image is not None:
                return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return None

    xs      = [b[0] for b in matched_boxes]
    ys      = [b[1] for b in matched_boxes]
    rights  = [b[0] + b[2] for b in matched_boxes]
    bottoms = [b[1] + b[3] for b in matched_boxes]

    cv_image = cv2.imread(image_path)
    if cv_image is None:
        return None

    img_h, img_w = cv_image.shape[:2]
    pad = 16   # Padding for highlighted box

    x  = max(0,     min(xs)      - pad)
    y  = max(0,     min(ys)      - pad)
    x2 = min(img_w, max(rights)  + pad)
    y2 = min(img_h, max(bottoms) + pad)

    if mode == "Cropped Paragraph":
        # Crop to the highlighted block
        crop = cv_image[y:y2, x:x2].copy()
        cv2.rectangle(
            crop,
            (0, 0),
            (crop.shape[1] - 1, crop.shape[0] - 1),
            (239, 68, 68),  # Red outline (RGB/BGR conversion compatible)
            4,
        )
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        # Render on full page
        highlighted = cv_image.copy()
        cv2.rectangle(
            highlighted,
            (x, y),
            (x2, y2),
            (0, 0, 255),  # Red box in BGR (rendered as red in OpenCV BGR)
            5,
        )
        return cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def run_search(query, model, client, top_k=20, score_gap=0.12, hard_min=0.25):
    # If the query is plain Latin script (Hinglish or English), try
    # converting it to Devanagari first — the index was built entirely
    # in Devanagari, so matching script gives much better similarity scores.
    # bge-m3 doesn't require query:/passage: prefixes (unlike e5 models)
    
    search_query = query
    if looks_romanized(query):
        transliterated = transliterate_to_devanagari(query)
        if transliterated:
           search_query = transliterated
    query_vector = model.encode(
        search_query,
        normalize_embeddings=True,
    ).tolist()

    # Fetch a large pool of candidates
    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    if not candidates:
        return [], 0.0

    # Find the top score
    best_score = candidates[0].score

    # Keep results that are within score_gap of the best AND above hard_min
    adaptive_cutoff = max(best_score - score_gap, hard_min)
    filtered = [r for r in candidates if r.score >= adaptive_cutoff]
    return filtered,best_score


# ============================================================
# MODERN USER INTERFACE LAYOUT
# ============================================================

st.set_page_config(
    page_title="Hindi Book Semantic Search",
    page_icon="📖",
    layout="wide",
)

# Custom premium styling via markdown
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* Elegant Title and Headers */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px !important;
    }
    
    /* Search Box styling */
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
        background-color: #1e293b !important;
    }
    
    /* Modern Badges for metadata */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        color: #ffffff;
    }
    .badge-book { background-color: #0284c7; }
    .badge-page { background-color: #0d9488; }
    .badge-score { background-color: #ea580c; }
    
    /* Custom Result Card container */
    .result-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Quote styled answers */
    .answer-box {
        background-color: #0f172a;
        border-left: 4px solid #6366f1;
        padding: 12px 18px;
        border-radius: 6px;
        margin-top: 10px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e2e8f0;
    }
    
        /* Off-topic warning box */
    .offtopic-box {
        background-color: #1e1a2e;
        border: 1px solid #6366f1;
        border-left: 5px solid #a855f7;
        border-radius: 10px;
        padding: 20px 24px;
        margin-top: 20px;
        color: #c4b5fd;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
</style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.title("📖 Hindi Book Semantic Search")
st.caption(
    "Ask a question in natural Hindi or English. The engine scans your books, "
    "understands the context, and shows you exactly where the answer resides."
)

st.markdown("---")

# --- SIDEBAR (Controls & Preferences) ---
with st.sidebar:
    st.header("⚙️ Search Control Center")
    st.markdown("Customize how the search operates:")

    score_gap = st.slider(
        "Answer grouping gap",
        min_value=0.02,
        max_value=0.30,
        value=0.12,
        step=0.02,
        help=(
            "Controls how many results appear. "
            "Smaller = only the most closely related answers. "
            "Larger = includes more loosely related results. "
            "The system automatically finds how many answers fit each question."
        ),
    )

    hard_min = st.slider(
        "Minimum score cutoff",
        min_value=0.10,
        max_value=0.60,
        value=0.40,
        step=0.05,
        help="Results below this score are always dropped as noise.",
    )

    st.markdown("---")
    st.header("🖼️ Image Rendering Settings")

    view_mode = st.radio(
        "Select Image View Mode",
        options=["Cropped Paragraph", "Full Page Highlighted"],
        index=0,
        help="Cropped extracts just the text area. Full Page draws a box on the entire page image.",
    )

    st.markdown("---")
    st.markdown(
        "**Score Reference:**\n"
        "- 🟢 **0.7 – 1.0** : Excellent match\n"
        "- 🟡 **0.5 – 0.7** : Good relevant match\n"
        "- 🟠 **0.3 – 0.5** : Broad/weak correlation\n"
        "- 🔴 **Below 0.25** : Noise or not relevant"
    )


# --- SEARCH BAR ---
query = st.text_input(
    "🔍 Enter your question:",
    placeholder="e.g.  मनुष्य जीवन का अंतिम ध्येय क्या है?",
)

if query.strip():
    with st.spinner("Analyzing semantics & searching database..."):
        try:
            model  = load_model()
            client = load_client()
            results,best_score = run_search(
                query.strip(), model, client,
                top_k=20,
                score_gap=score_gap,
                hard_min=hard_min,
            )
        except Exception as e:
            st.error(f"Failed to connect to Qdrant. Verify that your Docker container is running! Error details: {e}")
            results, best_score = [],0.0

    if query.strip() and best_score < RELEVANCE_THRESHOLD:
        st.markdown(f"""
        <div class="offtopic-box">
            🙏 <strong>यह प्रश्न इस पुस्तक के विषय से संबंधित नहीं लगता।</strong><br><br>
            इस खोज इंजन में केवल <strong>दादाश्री की पुस्तकों</strong> से संबंधित प्रश्नों के उत्तर मिलेंगे —
            जैसे आत्मा, कर्म, क्षमापना, अहंकार, प्रतिक्रमण आदि।<br><br>
            <em>कृपया पुस्तक के विषय से संबंधित प्रश्न पूछें।</em>
            <br><br>
            <span style="font-size:0.85rem; color:#7c6faf;">
                (Best match score: {round(best_score, 4)} — minimum required: {RELEVANCE_THRESHOLD})
            </span>
        </div>
        """, unsafe_allow_html=True)

    elif query.strip() and results:
        top_score = round(results[0].score, 4)
        st.success(f"🎉 Found **{len(results)}** relevant answer(s) — top confidence score: **{top_score}**")

        # Loop through search results
        for rank, result in enumerate(results, start=1):
            payload    = result.payload or {}
            score      = round(result.score, 4)
            pdf_name   = payload.get("pdf_name",   "unknown")
            page       = payload.get("page",        0)
            image_path = payload.get("image_path", "")
            full_text  = payload.get("text",        "")

            # Each chunk is already one full दादाश्री answer turn —
            # show it whole, no sentence-level trimming.
            display_text = full_text

            # Draw the visual container for each result
            st.markdown(f"""
            <div class="result-container">
                <span class="badge badge-book">📚 {pdf_name}</span>
                <span class="badge badge-page">📄 Page {page}</span>
                <span class="badge badge-score">🎯 Score: {score}</span>
            </div>
            """, unsafe_allow_html=True)

            # Two-column layout: Left = Text answer & Details, Right = Highlighted Image
            col1, col2 = st.columns([1, 1], gap="medium")

            with col1:
                st.markdown("### 💬 दादाश्री's Answer:")
                st.markdown(f'<div class="answer-box">{display_text}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("### 🖼️ Page Highlight:")
                if image_path:
                    with st.spinner("Processing page image..."):
                        highlighted_img = get_highlighted_image(image_path, display_text, mode=view_mode)

                    if highlighted_img is not None:
                        st.image(
                            highlighted_img,
                            caption=f"{pdf_name} (Page {page}) - Mode: {view_mode}",
                            use_container_width=True,
                        )
                    else:
                        st.warning("Could not pinpoint exact highlight on the page image. Displaying text only.")
                else:
                    st.info("No page image path stored for this record.")

            st.markdown("<hr style='border: 1px solid #334155; margin: 40px 0;'>", unsafe_allow_html=True)

    elif query.strip():
        st.warning(
            f"No matching entries met your Minimum match confidence of {hard_min}. "
            "Try lowering the confidence slider in the sidebar!"
        )



