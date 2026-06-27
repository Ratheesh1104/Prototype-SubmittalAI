import chromadb


class ChromaManager:

    def __init__(self, db_path):

        self.client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name="submittal_review"
        )

    def add_chunks(self, chunks, embeddings):

        ids = []
        docs = []
        metas = []

        for chunk in chunks:

            ids.append(chunk["chunk_id"])

            docs.append(chunk["text"])

            metas.append(
                {
                    "document_type": chunk["document_type"],
                    "page_number": chunk["page_number"]
                }
            )

        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings.tolist(),
            metadatas=metas
        )

    def query(
        self,
        embedding,
        top_k=10
    ):

        return self.collection.query(

            query_embeddings=[embedding],

            n_results=top_k

        )