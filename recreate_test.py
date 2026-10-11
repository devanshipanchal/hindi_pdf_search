from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

try:
    client = QdrantClient(host="localhost", port=6333)
    print("Testing connection...")
    collections = client.get_collections()
    print("Collections:", [c.name for c in collections.collections])
    
    print("Recreating collection test_col...")
    client.recreate_collection(
        collection_name="test_col",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print("Successfully recreated test_col!")
    
    # Clean up test_col
    client.delete_collection("test_col")
    print("Cleaned up test_col.")
except Exception as e:
    print("Error:", e)
