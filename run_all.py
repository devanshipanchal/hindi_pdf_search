"""
RUN PIPELINE — Runs steps 1 and 2 in order automatically.

Use this the first time you set up the project, or whenever you
add new PDF books to the pdfs/ folder.

After this finishes, run  python step4_search.py  to start searching.
"""

import convert_pdf
import build_index


def main():
    print("\n" + "=" * 60)
    print("RUNNING FULL PIPELINE")
    print("=" * 60)

    print("\n--- STEP 1: PDF to Filtered Pages (image + OCR + keyword filter) ---")
    convert_pdf.main()

    print("\n--- STEP 2: Store in Qdrant ---")
    build_index.main()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("Now run:  python step4_search.py")
    print("=" * 60)


if __name__ == "__main__":
    main()