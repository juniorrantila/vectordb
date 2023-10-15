from typing import List
import numpy as np
import faiss

class VectorSearch:
    """
    A class to perform vector search using different methods (MRPT, Faiss, or scikit-learn).
    """

    def __init__(self, embeddings: list[list[float]]):
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.embeddings = embeddings 

    def run_faiss(self, vector, k=15):
        """
        Search for the most similar vectors using Faiss method.
        """
        _, indices = self.index.search(x=np.array([vector]), k=k)
        return indices[0]

    def search_vectors(self, query_embedding: List[float], top_n: int) -> List[int]:
        """
        Searches for the most similar vectors to the query_embedding in the given embeddings.

        :param query_embedding: a list of floats representing the query vector.
        :param embeddings: a list of vectors to be searched, where each vector is a list of floats.
        :param top_n: the number of most similar vectors to return.
        :return: a list of indices of the top_n most similar vectors in the embeddings.
        """
        return self.run_faiss(query_embedding, top_n)
