
"""
STEP 3 — Split page text into paragraphs, embed each paragraph,
          and store in a local Qdrant vector database.

Reads:   json/<book_name>.json  (created by step 2)
Output:  qdrant_db/             (local vector database)

Each stored point contains:
  - vector:  meaning-based embedding of the paragraph text
  - payload: { pdf_name, page, image_path, text }

MODEL: BAAI/bge-m3
  - Strong multilingual retrieval quality
  - Fully supports Hindi (Devanagari script)
  - 1024-dimensional embeddings
  - Heavier than e5-small; slower on CPU but more accurate
"""

import json
import re
import sys
from collections import defaultdict
from uuid import uuid4

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    JSON_DIR,
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_LENGTH,
)


# Matches a speaker label at the START of a paragraph, e.g. "प्रश्नकर्ता :" or "दादाश्री :"
SPEAKER_START_PATTERN = re.compile(r"^(प्रश्नकर्ता\s*[:ः]|दादाश्री\s*[:ः])\s*")

# Matches a short numbered section/topic heading paragraph, e.g.
# "4. निकालें कषाय की कोठरी में से" or "22. निकाल, चिकनी फाइलों का"
HEADING_PATTERN = re.compile(r"^\d{1,3}\.\s+\S")

# Matches a running page header — book title and page number, in either
# order depending on left/right page, e.g. "76 प्रतिक्रमण", "22 मानव धर्म",
# "प्रतिक्रमण 29", "मानव धर्म 7". Used to strip the header wherever it
# lands in the OCR'd text: sometimes its own \n\n paragraph, sometimes
# glued onto the start of the next line with no blank line at the page
# break — so this is applied line-by-line, not just at paragraph edges.
PAGE_HEADER_PATTERN = re.compile(
    r"^\s*(\d{1,4}\s+(मानव\s*धर्म|प्रतिक्रमण)|(मानव\s*धर्म|प्रतिक्रमण)\s*\d{1,4})\s*[\]\)।.,]*\s*$"
)

# ---------------------------------------------------------------------------
# NEW: sentence-boundary detection for smart chunking.
#
# Used only by the oversized-single-paragraph fallback splitter below, so a
# hard split (Rule 3) still ends on a complete sentence (Rule 4) instead of
# cutting exactly at MAX_CHUNK_SIZE.
# ---------------------------------------------------------------------------

# Devanagari + English sentence-ending punctuation.
SENTENCE_END_CHARS = "।॥…?!."

# Trailing characters that may follow a sentence-ending punctuation mark and
# should be swept up into the same cut (closing quotes/brackets, or a run of
# punctuation like "..." or "?!").
SENTENCE_TRAILING_CHARS = SENTENCE_END_CHARS + "\"'”’)]"

# How many extra characters we're willing to look past MAX_CHUNK_SIZE to find
# a clean sentence boundary before giving up and doing a hard cut (Rule 5).
SENTENCE_BOUNDARY_TOLERANCE = 100


def strip_page_headers(text):
    """Remove running-header lines (e.g. '76 प्रतिक्रमण') from a page's
    OCR text, checking each line rather than only whole \\n\\n paragraphs,
    since a header sometimes ends up glued to adjacent text with no
    blank line separating it at the page break."""
    lines = text.split("\n")
    kept = [line for line in lines if not PAGE_HEADER_PATTERN.match(line)]
    return "\n".join(kept)


def build_book_text(pages):
    """
    Concatenate one book's kept pages (already sorted by page number)
    into a single text blob, and record where each page starts within
    that blob — so a दादाश्री turn that continues onto the next page
    (very common — OCR/page breaks don't respect sentence endings)
    still becomes ONE chunk instead of being cut off at the page edge.

    Returns (combined_text, page_starts) where page_starts is a list
    of (offset, page_number, image_path) sorted by offset — used to
    look up which page/image a given position in combined_text falls on.
    """
    combined = ""
    page_starts = []

    for page in pages:
        page_starts.append((len(combined), page["page"], page["image_path"]))
        combined += strip_page_headers(page["text"]) + "\n\n"

    return combined, page_starts


def page_for_offset(offset, page_starts):
    """Find which page (and its image) a character offset falls into."""
    page_number, image_path = page_starts[0][1], page_starts[0][2]
    for start, pnum, img in page_starts:
        if start <= offset:
            page_number, image_path = pnum, img
        else:
            break
    return page_number, image_path


def iter_paragraphs_with_offsets(combined_text):
    """Yield (offset, paragraph_text) for each \\n\\n-separated paragraph,
    where offset is that paragraph's exact start position in combined_text."""
    offset = 0
    for part in combined_text.split("\n\n"):
        yield offset, part
        offset += len(part) + 2   # +2 accounts for the "\n\n" separator consumed by split


# ---------------------------------------------------------------------------
# NEW: sentence-boundary-aware hard splitter.
#
# Only ever called on a single "piece" of text that is, by itself, longer
# than MAX_CHUNK_SIZE (Rule 3). Instead of blindly slicing at a fixed
# character count (the old behaviour, which produced mid-sentence cuts),
# this walks forward from the ideal cut point and looks for the nearest
# real sentence ending within a small tolerance window (Rule 4), falling
# back to a hard cut only if none is found (Rule 5).
# ---------------------------------------------------------------------------
def _find_sentence_boundary(text, ideal_end, tolerance):
    """
    Search text[ideal_end : ideal_end + tolerance] for the first sentence-
    ending character. Returns the index just AFTER that boundary (and any
    immediately-following punctuation/closing quotes), or None if no
    boundary is found within the tolerance window.
    """
    search_end = min(len(text), ideal_end + tolerance)

    for i in range(ideal_end, search_end):
        if text[i] in SENTENCE_END_CHARS:
            # Sweep up any trailing punctuation/quotes glued to the same
            # sentence ending (e.g. "?!", "...", or a closing quote mark)
            # so we don't leave a stray character at the start of the
            # next chunk.
            j = i + 1
            while j < len(text) and text[j] in SENTENCE_TRAILING_CHARS:
                j += 1
            return j

    return None  # no boundary nearby — caller falls back to a hard cut


def hard_split_on_sentence_boundaries(text, start_offset, max_size,
                                       tolerance=SENTENCE_BOUNDARY_TOLERANCE):
    """
    Split a single oversized piece of text (len(text) > max_size) into
    <= max_size-ish pieces, preferring to end each piece on a complete
    sentence rather than cutting at a fixed character count.

    For each piece:
      - Target cut point is `pos + max_size` (Rule 1's ~500 target).
      - We look forward up to `tolerance` extra characters for a sentence
        ending (Rule 4) and cut right after it if found.
      - If nothing is found in that window, we hard-cut at the target
        point as a last resort (Rule 5).

    Returns a list of (piece_text, piece_offset), where piece_offset is
    that piece's real start position in the ORIGINAL combined_text
    (start_offset + local position) — this keeps page/image mapping
    accurate even when one long paragraph is sliced into several pieces
    that may land on different pages (Rule 6).
    """
    pieces = []
    pos = 0
    n = len(text)

    while pos < n:
        remaining = n - pos

        # Remainder already fits — take it all, no need to search further.
        if remaining <= max_size:
            sub = text[pos:].strip()
            if sub:
                pieces.append((sub, start_offset + pos))
            break

        ideal_end = pos + max_size
        cut = _find_sentence_boundary(text, ideal_end, tolerance)
        if cut is None:
            # Rule 5 fallback: no sentence boundary nearby, hard cut.
            cut = ideal_end
        # Guard against a boundary search returning a non-advancing cut
        # (shouldn't happen given ideal_end > pos, but keeps the loop safe).
        cut = max(cut, pos + 1)

        sub = text[pos:cut].strip()
        if sub:
            pieces.append((sub, start_offset + pos))
        pos = cut

    return pieces


def split_long_texts_by_paragraphs(texts_with_offsets, max_size):
    """
    Group (offset, paragraph_text) items into ~max_size pieces, preferring
    to cut between paragraphs rather than mid-sentence.

    CHANGED (Rules 1 & 2): a paragraph is now added to the current group
    FIRST, and only checked afterwards. If the running total has reached
    or exceeded max_size, the group is closed right there — with the
    paragraph that pushed it over still fully included, never truncated.
    (Previously the check happened before adding, which could bump that
    paragraph into the next chunk instead of keeping it with the group
    it naturally belongs to.)

    Returns a list of (piece_text, first_offset) — first_offset is the
    offset of whatever paragraph STARTS that piece, so each piece can
    later be mapped to the page it actually appears on (critical: a
    long section can span many original pages, so every piece needs
    its own page lookup instead of inheriting the section's first page).
    """
    pieces  = []
    current_parts   = []
    current_offset  = None
    current_len     = 0

    def flush():
        if current_parts:
            pieces.append(("\n\n".join(current_parts).strip(), current_offset))

    for offset, para in texts_with_offsets:
        added_len = len(para) + (2 if current_parts else 0)

        if current_offset is None:
            current_offset = offset

        # Rule 1 & 2: always add the whole paragraph first — we never
        # truncate a paragraph to make it fit.
        current_parts.append(para)
        current_len += added_len

        if current_len >= max_size:
            # Reached (or passed) the target size. Close the chunk now,
            # keeping the paragraph we just added intact — this is what
            # produces the "430 + 120 = 550, keep it all together"
            # behaviour instead of pushing the 120-char paragraph into
            # the next chunk.
            flush()
            current_parts, current_offset, current_len = [], None, 0

    flush()  # trailing partial group, if any

    # Rule 3: a single paragraph that is, by itself, larger than max_size
    # still needs to be split — but now via sentence-boundary-aware
    # splitting (Rule 4) instead of a blind fixed-size slice, and with
    # correct per-sub-piece offsets (Rule 6) instead of all sub-pieces
    # inheriting the same starting offset.
    final = []
    for piece_text, offset in pieces:
        if len(piece_text) > max_size:
            final.extend(hard_split_on_sentence_boundaries(piece_text, offset, max_size))
        else:
            final.append((piece_text, offset))

    return final


def split_into_dadashree_chunks(combined_text, page_starts):
    """
    Split a whole book's concatenated text into one chunk per topic
    SECTION (delimited by numbered headings like "4. निकालें कषाय की
    कोठरी में से"), where each chunk is ALL of दादाश्री's replies within
    that section joined together — not just a single speaker turn.

    Rationale: this book is written as rapid back-and-forth dialogue,
    so a single दादाश्री turn is often just one short sentence. Grouping
    by section gives a full, paragraph-length answer per topic instead
    of a one-liner, at the cost of a search result sometimes covering
    more than one exchange. प्रश्नकर्ता's questions are never included
    in the indexed/displayed text — only दादाश्री's own words.

    A paragraph with no speaker label continues whichever speaker's
    turn came before it (OCR line breaks split a single दादाश्री answer
    into several \\n\\n-separated paragraphs without repeating the label).

    Returns a list of (chunk_text, page_number, image_path) — each
    piece is mapped to the page IT actually starts on, since a section
    can span many original pages.
    """
    sections = []                 # list of [(offset, content), ...] per section
    current_section = []
    last_speaker     = None

    def close_section():
        if current_section:
            sections.append(current_section)

    for offset, raw_para in iter_paragraphs_with_offsets(combined_text):
        para = raw_para.strip()
        if not para:
            continue

        if HEADING_PATTERN.match(para) and len(para) <= 80:
            close_section()
            current_section = []
            last_speaker    = None
            continue

        if PAGE_HEADER_PATTERN.match(para) and len(para) <= 40:
            continue   # running page header (e.g. "76 प्रतिक्रमण") — drop, keep current speaker/section

        m = SPEAKER_START_PATTERN.match(para)
        if m:
            marker  = m.group(1)
            content = para[m.end():].strip()
            last_speaker = "दादाश्री" if "दादाश्री" in marker else "प्रश्नकर्ता"
        else:
            content = para   # continuation of whichever speaker was last

        if last_speaker == "दादाश्री" and content:
            current_section.append((offset, content))

    close_section()

    results = []
    for section in sections:
        for piece_text, piece_offset in split_long_texts_by_paragraphs(section, MAX_CHUNK_SIZE):

            # Skip stray fragments: too short, or reads like a leaked
            # question/heading (ends in "?") rather than a real answer.
            if len(piece_text) < MIN_CHUNK_LENGTH or piece_text.rstrip().endswith("?"):
                continue

            page_number, image_path = page_for_offset(piece_offset, page_starts)
            results.append((piece_text, page_number, image_path))

    return results


def load_pages_by_book():
    """Load all OCR JSON files, grouped by book and sorted by page number."""
    pages_by_book = defaultdict(list)
    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {JSON_DIR}")
        print("Please run convert_pdf.py first.")
        return {}

    for json_file in json_files:
        print(f"  Loading: {json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            pages = json.load(f)
        for page in pages:
            pages_by_book[page["pdf_name"]].append(page)

    for pdf_name in pages_by_book:
        pages_by_book[pdf_name].sort(key=lambda p: p["page"])

    return pages_by_book


def main():
    print("=" * 60)
    print("STEP 3 — Store in Qdrant")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step A: Load OCR data from JSON files, grouped by book
    # ----------------------------------------------------------
    print("\nLoading OCR data from json/ folder...")
    pages_by_book = load_pages_by_book()

    if not pages_by_book:
        return

    total_pages_loaded = sum(len(pages) for pages in pages_by_book.values())
    print(f"Total pages loaded: {total_pages_loaded} across {len(pages_by_book)} book(s)")

    # ----------------------------------------------------------
    # Step B: Connect to Qdrant Docker & recreate collection
    # ----------------------------------------------------------
    print(f"\nConnecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Delete & recreate the collection so we start fresh
    print(f"Recreating collection '{COLLECTION_NAME}' (size={EMBEDDING_DIM})...")
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    print("Collection created successfully!")

    # ----------------------------------------------------------
    # Step C: Load the embedding model
    # ----------------------------------------------------------
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(Downloading on first run — about 470MB, one-time only)")
    sys.stdout.flush()

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully!")
    sys.stdout.flush()

    # ----------------------------------------------------------
    # Step D: Process each book as one continuous text — this lets a
    # दादाश्री turn that continues onto the next page still become
    # ONE chunk, then embed and upload, book by book.
    # ----------------------------------------------------------
    total_chunks = 0
    total_pages_processed = 0

    print(f"\nProcessing and indexing {len(pages_by_book)} book(s)...")
    print("-" * 60)

    for pdf_name, pages in pages_by_book.items():

        combined_text, page_starts = build_book_text(pages)
        book_chunks = split_into_dadashree_chunks(combined_text, page_starts)

        if not book_chunks:
            print(f"  [SKIP] No दादाश्री chunks found in {pdf_name}")
            sys.stdout.flush()
            continue

        chunk_texts = [c[0] for c in book_chunks]

        # bge-m3 doesn't require query:/passage: prefixes (unlike e5 models)
        embeddings = model.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Build Qdrant point objects and upload in one batch per book
        points = []
        for (chunk_text, page_number, image_path), embedding in zip(book_chunks, embeddings):
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload={
                    "pdf_name":   pdf_name,
                    "page":       page_number,
                    "image_path": image_path,
                    "text":       chunk_text,
                },
            )
            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        total_chunks      += len(book_chunks)
        total_pages_processed += len(pages)

        print(f"  [OK] {pdf_name}: {len(pages)} pages -> {len(book_chunks)} chunks | Total so far: {total_chunks}")
        sys.stdout.flush()

    # ----------------------------------------------------------
    # Step E: Done!
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 3 COMPLETE!")
    print(f"  Pages processed : {total_pages_processed}")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Stored in       : Qdrant collection '{COLLECTION_NAME}'")
    print("\nNow run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()