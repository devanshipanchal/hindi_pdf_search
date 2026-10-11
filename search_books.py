"""
STEP 4 — Search your books using natural language (Hindi or English).

How it works:
  1. You type a question or phrase
  2. The question is converted to a meaning-vector
  3. Qdrant finds the most similar paragraphs
  4. Results are printed: book name, page, text, image path
  5. The FULL PAGE image opens in a viewer window, with a red box
     drawn around the matching paragraph

FIX APPLIED:
  - Removed "search_query: " prefix from model.encode() call.
    The index was built WITHOUT any prefix (plain chunk text),
    so the query MUST also be encoded without a prefix — otherwise
    the two vectors are in different spaces and cosine similarity
    scores collapse to ~0.20 even for on-topic queries.

  - Added hard_min filter (same as app.py) so truly off-topic
    queries show "no results" instead of returning the least-bad match.
"""

import difflib

import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
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

# Minimum score the TOP result must have.
# Below this → query is off-topic for this book collection.
HARD_MIN = 0.25


def normalize(text):
    """Squash away all spaces/newlines so small OCR spacing
    differences don't break the text-matching below."""
    return "".join(text.split())


def line_belongs_to_chunk(line_norm, text_norm, min_ratio=0.7, min_fuzzy_len=15):
    """True if this OCR'd line is (close to) part of the chunk text."""
    if not line_norm:
        return False
    if line_norm in text_norm:
        return True
    if len(line_norm) < min_fuzzy_len:
        return False

    match = difflib.SequenceMatcher(None, line_norm, text_norm, autojunk=False).find_longest_match(
        0, len(line_norm), 0, len(text_norm)
    )
    return (match.size / len(line_norm)) >= min_ratio


def find_highlight_bbox(image_path, chunk_text):
    """Look at the page image right now, find where each line of
    text sits, and return the box covering the lines that belong to
    the matched chunk: [x, y, w, h]."""

    image = Image.open(image_path)

    data = pytesseract.image_to_data(
        image,
        lang=OCR_LANG,
        config="--psm 6",
        output_type=Output.DICT,
    )

    line_groups = {}
    n_words = len(data["text"])

    for i in range(n_words):
        word = data["text"][i].strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_groups.setdefault(key, []).append(i)

    chunk_norm = normalize(chunk_text)
    matched_boxes = []

    for indices in line_groups.values():
        words = [data["text"][i] for i in indices]
        line_text = " ".join(words).strip()
        line_norm = normalize(line_text)

        if line_belongs_to_chunk(line_norm, chunk_norm):
            xs      = [data["left"][i] for i in indices]
            ys      = [data["top"][i] for i in indices]
            rights  = [data["left"][i] + data["width"][i] for i in indices]
            bottoms = [data["top"][i] + data["height"][i] for i in indices]
            matched_boxes.append([
                min(xs), min(ys),
                max(rights) - min(xs), max(bottoms) - min(ys),
            ])

    if not matched_boxes:
        return None

    xs      = [b[0] for b in matched_boxes]
    ys      = [b[1] for b in matched_boxes]
    rights  = [b[0] + b[2] for b in matched_boxes]
    bottoms = [b[1] + b[3] for b in matched_boxes]

    x, y = min(xs), min(ys)
    w, h = max(rights) - x, max(bottoms) - y

    return [x, y, w, h]


def open_full_page_image(image_path, highlight_bbox=None):
    """Open the full page image using OpenCV with optional red highlight box."""

    image = cv2.imread(image_path)

    if image is None:
        print(f"  Could not open image: {image_path}")
        return

    if highlight_bbox:
        x, y, w, h = highlight_bbox
        pad = 6
        cv2.rectangle(
            image,
            (max(0, x - pad), max(0, y - pad)),
            (x + w + pad, y + h + pad),
            (0, 0, 255),
            4,
        )

    window_title = f"Full Page — {image_path}"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    height, width = image.shape[:2]
    max_height = 900
    if height > max_height:
        scale  = max_height / height
        width  = int(width * scale)
        height = max_height

    cv2.resizeWindow(window_title, width, height)
    cv2.imshow(window_title, image)

    print(f"\n  Image opened. Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def search(query, model, client, top_k=10):
    """
    Embed the query WITHOUT any prefix and find top matching paragraphs.

    FIX: old code used f"search_query: {query}" as the input to encode().
    The index was built with plain chunk text (no prefix), so adding a
    prefix at query time shifts the vector into a different region of the
    embedding space — scores drop to ~0.20 for on-topic queries and the
    hard_min filter silently drops everything.

    bge-m3 does NOT require query/passage prefixes (unlike e5 models).
    Plain text works correctly for both indexing and querying.
    """
    query_vector = model.encode(
        query,                      # ← plain query, no prefix
        normalize_embeddings=True,
    ).tolist()

    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    # Filter out results below the hard minimum threshold
    return [r for r in candidates if r.score >= HARD_MIN]


def main():
    print("=" * 60)
    print("STEP 4 — Search Your Books")
    print("=" * 60)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("\nReady! Type your search query below.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    while True:

        try:
            query = input("Enter search query: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print(f"\nSearching for: {query}")
        print("-" * 60)

        results = search(query, model, client, top_k=10)

        if not results:
            print("No relevant results found.")
            print(f"(All scores were below minimum threshold of {HARD_MIN})")
            print("Try a different query related to the book's topics.")
            continue

        # Show top 3 results max in terminal
        display_results = results[:3]

        for rank, result in enumerate(display_results, start=1):

            payload     = result.payload or {}
            score       = round(result.score, 4)
            pdf_name    = payload.get("pdf_name",   "unknown")
            page        = payload.get("page",        0)
            image_path  = payload.get("image_path", "")
            text        = payload.get("text",        "")

            print(f"\nResult {rank}")
            print(f"  Book       : {pdf_name}")
            print(f"  Page       : {page}")
            print(f"  Score      : {score}  (1.0 = perfect match)")
            print(f"  Image path : {image_path}")
            print(f"  Text       :")
            print(f"    {text[:400]}{'...' if len(text) > 400 else ''}")

        print("\n" + "-" * 60)

        while True:
            choice = input(
                f"Open full page image for result (1-{len(display_results)}), or press Enter to skip: "
            ).strip()

            if not choice:
                break

            if choice.isdigit() and 1 <= int(choice) <= len(display_results):
                selected = display_results[int(choice) - 1]
                selected_payload = selected.payload or {}
                image_path = selected_payload.get("image_path", "")
                chunk_text = selected_payload.get("text", "")

                if image_path:
                    highlight_bbox = find_highlight_bbox(image_path, chunk_text)
                    if highlight_bbox is None:
                        print("  (Couldn't pinpoint the exact spot — showing full page, no box.)")
                    open_full_page_image(image_path, highlight_bbox)
                else:
                    print("  No image path stored for this result.")
                break
            else:
                print(f"  Please enter a number between 1 and {len(display_results)}.")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()