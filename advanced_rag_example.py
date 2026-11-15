"""Advanced RAG example: hybrid search + cross-encoder re-ranking."""
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List

class HybridRerankRetriever:
    def __init__(self, documents: List[Document], persist_dir: str = "./chroma_db"):
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_documents(documents, self.embedding, persist_directory=persist_dir)
        self.bm25 = BM25Retriever.from_documents(documents)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    def retrieve(self, query: str, k: int = 5, fetch_k: int = 20):
        # Vector search
        vector_docs = self.vectorstore.similarity_search(query, k=fetch_k)
        # BM25 search
        self.bm25.k = fetch_k
        bm25_docs = self.bm25.get_relevant_documents(query)
        
        # Merge and deduplicate
        seen = set()
        all_docs = []
        for d in vector_docs + bm25_docs:
            key = d.metadata.get("source", "") + d.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_docs.append(d)
        
        # Re-rank
        pairs = [(query, d.page_content) for d in all_docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]