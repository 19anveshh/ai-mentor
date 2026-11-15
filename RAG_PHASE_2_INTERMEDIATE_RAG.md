# Phase 2 — Intermediate RAG: Advanced Retrieval (Weeks 5–8)

## Goal
Fix naive RAG's retrieval problems: missing context, wrong chunks, irrelevant results, and poor precision. Add hybrid search, re-ranking, query transformation, and context compression.

---

## Week 5 — Query Transformation

### Day 1: Multi-Query Retrieval
Generate multiple variations of the user query, retrieve for each, fuse results.

```python
# intermediate_rag/multi_query.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

multi_query_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant. Generate 3 different versions of the following question
    to improve retrieval. Return them as a newline-separated list.
    
    Question: {question}"""
)

def generate_queries(question: str, llm) -> list[str]:
    chain = multi_query_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    return [q.strip() for q in result.split("\n") if q.strip()]
```

### Day 2: HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer, embed it, use that embedding for retrieval.

```python
# intermediate_rag/hyde.py
hyde_prompt = ChatPromptTemplate.from_template(
    "Write a hypothetical paragraph that answers this question: {question}"
)

def hyde_retrieve(question: str, llm, vectorstore, k: int = 5):
    hypothetical = llm.invoke(hyde_prompt.format(question=question))
    return vectorstore.similarity_search(hypothetical.content, k=k)
```

### Day 3: Query Decomposition
Break complex questions into sub-questions, retrieve for each, synthesize.

### Day 4: Query Expansion
Add synonyms, expand acronyms, correct spelling before retrieval.

### Day 5: Comparison Experiment
Run the same 20 eval questions with: plain query, multi-query, HyDE, decomposition. Compare hit@k.

**Deliverable:** `EVAL_REPORTS/query_transformation_comparison.json`

---

## Week 6 — Re-ranking + Hybrid Search

### Day 1: Cross-Encoder Re-ranking
Take top-50 from vector search, re-rank with cross-encoder (bge-reranker, Cohere rerank), keep top-5.

```python
# intermediate_rag/reranker.py
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank(query: str, documents, top_n: int = 5):
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]
```

### Day 2: BM25 + Hybrid Search
BM25 for keyword matching + dense vector search, combined with Reciprocal Rank Fusion (RRF).

```python
# intermediate_rag/hybrid.py
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever, k: int = 5):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.k = k
    
    def _get_relevant_documents(self, query):
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        # RRF fusion
        fused = {}
        for rank, doc in enumerate(vector_docs + bm25_docs):
            doc_id = doc.metadata.get("source", str(id(doc)))
            fused[doc_id] = fused.get(doc_id, 0) + 1 / (rank + 60)
        sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:self.k]]
```

### Day 3: Weighted Hybrid Search
Combine vector similarity score and BM25 score with tunable weights.

### Day 4: Metadata Filtering
Pre-filter (before search) vs post-filter (after search) by category, date, source.

### Day 5: Hybrid + Re-ranking Pipeline
Combine BM25 + vector + cross-encoder re-ranking in one retriever.

**Deliverable:** `intermediate_rag/hybrid_retriever.py` with eval comparison

---

## Week 7 — Advanced Splitting + Context Compression

### Day 1: Semantic Chunking
Split by meaning, not just character count. Use embedding-based semantic boundaries.

### Day 2: Parent-Child Retrieval
Store small chunks for retrieval, return larger parent context to LLM.

### Day 3: Document-Aware Splitting
Respect document structure (sections, tables, headers) when splitting.

### Day 4: Context Compression
LongLLMLingua, contextual compression, lost-in-the-middle awareness.

### Day 5: Compression Pipeline
Compress retrieved context before sending to LLM, measure token savings vs quality loss.

**Deliverable:** `intermediate_rag/compression_demo.py` with before/after token counts

---

## Week 8 — Evaluation Framework

### Day 1: Build Custom Eval Dataset
50+ QA pairs with ground-truth answers and relevant source documents.

### Day 2: Retrieval Metrics
Hit@k, MRR, NDCG, precision@k, recall@k — implement from scratch.

### Day 3: Generation Metrics
Faithfulness, answer relevancy, semantic similarity, citation accuracy.

### Day 4: Full Pipeline Evaluation
Run eval on naive RAG vs hybrid + re-ranking RAG.

### Day 5: Report + Gate
Write eval report, log cost/latency, update `ARCHITECTURE.md`.

**Deliverable:** `EVAL_REPORTS/intermediate_rag_eval.json`

---

## Phase 2 Hard Gate
- [ ] Retrieval hit@k > 0.8 on custom eval set
- [ ] Hybrid search (BM25 + vector) implemented
- [ ] Cross-encoder re-ranking implemented
- [ ] Query transformation (multi-query or HyDE) implemented
- [ ] Eval report with retrieval + generation metrics
- [ ] Cost/latency numbers logged
