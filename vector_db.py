from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self,ids,vectors,payloads):
        points= [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)


    def search(self, query_vector, top_k: int = 5):
        # 1. Get the full response object
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        
        # 2. Extract the list of points explicitly
        results = response.points 
        
        # DEBUG: See if we actually got anything
        print(f"DEBUG: Qdrant returned {len(results)} matches.")

        context = []
        sources = set()

        for result in results:
            # 3. Access payload directly from the result object
            payload = result.payload or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                context.append(text)
                sources.add(source)

        return {
            'context': context,
            'sources': list(sources)
        }