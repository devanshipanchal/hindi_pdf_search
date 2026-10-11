"""
STEP 1 — Convert each PDF page to an image, OCR it, and keep the
          page only if it looks like real Prashnkarta/Dadashree
          dialogue content (see KEEP_KEYWORDS in config.py).

Pages that don't match are discarded immediately: no image is
saved to disk and no entry is written to the JSON.

Reads:   pdfs/<book_name>.pdf
Output:  images/<book_name>/page_NNN.jpg   (kept pages only)
         json/<book_name>.json             (kept pages only)

Each JSON file is a list of page objects:
  [
    {
      "pdf_name":   "my_book.pdf",
      "page":       12,
      "image_path": "images/my_book/page_012.jpg",
      "text":       "...hindi text..."
    },
    ...
  ]

Page numbers are the original page number in the source PDF —
gaps are expected wherever a page was filtered out.
"""

import json

import pytesseract
from pdf2image import convert_from_path

from config import (
    PDFS_DIR,
    IMAGES_DIR,
    JSON_DIR,
    DPI,
    POPPLER_PATH,
    TESSERACT_CMD,
    OCR_LANG,
    KEEP_KEYWORDS,
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def run_ocr_on_image(image):
    """Run Tesseract OCR on one in-memory image. Returns cleaned text."""

    text = pytesseract.image_to_string(
        image,
        lang=OCR_LANG,
        config="--psm 6",     # treat image as a single block of text
    )

    text = text.replace("\x0c", " ").strip()

    return text


def matches_keep_keywords(text):
    """True if the page text contains any of the dialogue keywords."""
    return any(keyword in text for keyword in KEEP_KEYWORDS)


def process_pdf(pdf_path):
    """Convert, OCR and filter all pages of one PDF. Returns kept page dicts."""

    book_name  = pdf_path.stem
    pdf_name   = pdf_path.name
    out_folder = IMAGES_DIR / book_name
    json_file  = JSON_DIR / f"{book_name}.json"

    if json_file.exists():
        print(f"\n  Skipping (already done): {pdf_name}")
        return []

    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {pdf_name}")

    pages = convert_from_path(pdf_path, dpi=DPI, poppler_path=POPPLER_PATH)
    print(f"  Total pages in PDF: {len(pages)}")

    last_page_number = len(pages)
    kept_pages = []

    for page_number, page_image in enumerate(pages, start=1):

        if page_number == last_page_number:
            print(f"  Page {page_number}: skipped (last page of book)")
            continue

        text = run_ocr_on_image(page_image)

        if not matches_keep_keywords(text):
            print(f"  Page {page_number}: skipped (no dialogue keywords)")
            continue

        image_filename = f"page_{page_number:03d}.jpg"
        image_path = out_folder / image_filename
        page_image.save(image_path, "JPEG")

        word_count = len(text.split())
        print(f"  Page {page_number}: kept ({word_count} words) -> {image_filename}")

        relative_image_path = str(image_path).replace("\\", "/")

        kept_pages.append({
            "pdf_name":   pdf_name,
            "page":       page_number,
            "image_path": relative_image_path,
            "text":       text,
        })

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(kept_pages, f, ensure_ascii=False, indent=2)

    print(f"  Kept {len(kept_pages)}/{len(pages)} pages. Saved: {json_file}")

    return kept_pages


def main():
    print("=" * 60)
    print("STEP 1 — PDF to Filtered Pages (image + OCR + keyword filter)")
    print("=" * 60)

    if not PDFS_DIR.exists():
        PDFS_DIR.mkdir(parents=True)
        print(f"\nCreated folder: {PDFS_DIR}")
        print("Please put your PDF books inside the 'pdfs' folder and run again.")
        return

    pdf_files = sorted(PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"\nNo PDFs found in: {PDFS_DIR}")
        print("Please put your PDF books inside the 'pdfs' folder.")
        return

    print(f"\nFound {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    for pdf_path in pdf_files:
        process_pdf(pdf_path)

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE — Filtered pages saved as images + JSON.")
    print("Next: run  python build_index.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
