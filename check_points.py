from qdrant_client import QdrantClient
from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection status: {info.status}")
    print(f"Points count: {info.points_count}")
    
    # Try fetching 1 point to see payload structure
    res = client.scroll(collection_name=COLLECTION_NAME, limit=1)
    if res[0]:
        print("\nFirst point payload sample:")
        print(res[0][0].payload)
    else:
        print("\nNo points found in collection.")
except Exception as e:
    print(f"Error checking collection details: {e}")
