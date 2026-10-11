"""
Quick diagnostic — paste this in your project folder and run:
    python diagnose.py
"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
COLLECTION_NAME = "book_paragraphs"
EMBEDDING_MODEL = "BAAI/bge-m3"

print("=" * 55)
print("DIAGNOSTIC")
print("=" * 55)

# Step 1: connect
client = QdrantClient(path="./qdrant_db")

# Step 2: collection info
info = client.get_collection(COLLECTION_NAME)
count = info.points_count
print(f"\nTotal chunks stored in Qdrant : {count}")

if count == 0:
    print("\n❌  Collection is EMPTY.")
    print("   Fix: run   python build_index.py   first.")
    exit()

# Step 3: show 3 sample chunks
print("\nSample stored chunks (first 3):")
sample = client.scroll(collection_name=COLLECTION_NAME, limit=3, with_payload=True)[0]
for i, pt in enumerate(sample, 1):
    payload = pt.payload or {}
    text = payload.get("text", "")[:120]
    page = payload.get("page", "?")
    pdf  = payload.get("pdf_name", "?")
    print(f"\n  [{i}] Book={pdf}  Page={page}")
    print(f"       Text: {text}...")

# Step 4: test search
print("\n" + "-" * 55)
print("Test search: 'धर्म क्या है?'")
model = SentenceTransformer(EMBEDDING_MODEL)
vec   = model.encode("धर्म क्या है?", normalize_embeddings=True).tolist()
hits  = client.query_points(
    collection_name=COLLECTION_NAME,
    query=vec,
    limit=5,
).points

print(f"Top 5 scores returned:")
for h in hits:
    payload = h.payload or {}
    print(f"  score={h.score:.4f}  text={payload['text'][:80]}...")

print("\n" + "=" * 55)