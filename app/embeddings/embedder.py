from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self, model_name):

        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

    def encode(self, texts):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )