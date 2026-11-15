# Phase 3 — Advanced RAG: Modular & Agentic (Weeks 9–12)

## Goal
RAG that reasons, routes, and self-corrects. Build agentic systems that decide when to retrieve, how many times, and whether the answer is good enough.

---

## Week 9 — Modular RAG + Router Patterns

### Day 1: Modular Architecture
Retrieval → Re-rank → Compress → Generate as swappable modules.

```python
# advanced_rag/modular_rag.py
from typing import Protocol
from dataclasses import dataclass

@dataclass
class RAGResult:
    answer: str
    sources: list
    retrieval_count: int
    latency_ms: float

class RetrieverModule(Protocol):
    def retrieve(self, query: str) -> list: ...

class RerankerModule(Protocol):
    def rerank(self, query: str, docs: list) -> list: ...

class GeneratorModule(Protocol):
    def generate(self, query: str, context: str) -> str: ...

class ModularRAG:
    def __init__(self, retriever, reranker, generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
    
    def run(self, query: str) -> RAGResult:
        import time
        start = time.time()
        docs = self.retriever.retrieve(query)
        docs = self.reranker.rerank(query, docs)
        context = "\n".join(d.page_content for d in docs)
        answer = self.generator.generate(query, context)
        latency = (time.time() - start) * 1000
        return RAGResult(answer, docs, 1, latency)
```

### Day 2: Intelligent Router
Route queries to: vector DB, keyword search, SQL database, or LLM-only.

```python
# advanced_rag/router.py
ROUTER_PROMPT = """Classify this question into: VECTOR, KEYWORD, SQL, or LLM_ONLY.
VECTOR = semantic search over documents
KEYWORD = exact term search
SQL = structured database query
LLM_ONLY = general knowledge, no retrieval needed

Question: {question}
Category:"""
```

### Day 3: Multi-Source Retrieval
Query multiple sources in parallel, merge and deduplicate results.

### Day 4: Router Evaluation
Test routing accuracy on 100+ labeled questions.

**Deliverable:** `advanced_rag/router.py` with accuracy metrics

---

## Week 10 — Agentic RAG (LangGraph)

### Day 1: LangGraph Basics
State graph, nodes, edges, conditional edges.

```python
# advanced_rag/agentic_rag.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class RAGState(TypedDict):
    question: str
    documents: list
    answer: str
    needs_retrieval: bool
    retry_count: int

def retrieve_node(state: RAGState) -> RAGState:
    docs = vectorstore.similarity_search(state["question"], k=5)
    return {**state, "documents": docs, "needs_retrieval": False}

def generate_node(state: RAGState) -> RAGState:
    context = "\n".join(d.page_content for d in state["documents"])
    answer = llm.invoke(f"Answer: {state['question']}\nContext: {context}")
    return {**state, "answer": answer}

def should_retrieve(state: RAGState) -> str:
    if state.get("retry_count", 0) > 2:
        return END
    return "retrieve"

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_conditional_edges("generate", should_retrieve)
```

### Day 2: Retrieve → Critique → Rewrite → Re-retrieve
Agent critiques its own answer, decides if more retrieval is needed.

### Day 3: Multi-Hop Reasoning
Decompose complex questions, retrieve for each sub-question, synthesize.

### Day 4: Self-RAG / Adaptive Retrieval
Self-reflection, adaptive retrieval based on confidence scores.

**Deliverable:** `advanced_rag/agentic_rag.py` — self-correcting RAG loop

---

## Week 11 — Multi-Modal + Graph RAG

### Day 1: Document Layout Parsing
Unstructured, Marker, or Azure Document Intelligence for tables + images.

### Day 2: Multi-Modal Embeddings
CLIP, ColPali, or multimodal embedding models for text + images.

### Day 3: Table Extraction + RAG
Extract tables from PDFs, convert to structured data, query with text-to-SQL + RAG.

### Day 4: Graph RAG
Knowledge graph construction (entity extraction), graph traversal + vector search hybrid.

```python
# advanced_rag/graph_rag.py
# Extract entities → build graph → traverse + vector search
```

**Deliverable:** `advanced_rag/multimodal_rag.py` or `graph_rag.py`

---

## Week 12 — Long-Context + Stress Testing

### Day 1: Large Context Windows
Test with 32k, 128k context models; needle-in-haystack evaluation.

### Day 2: Lost-in-the-Middle
Test retrieval when relevant info is at beginning, middle, or end of context.

### Day 3: Context Window Stress Test
Gradually increase document count, measure quality degradation.

### Day 4: Final Polish + Gate
Complete eval report, cost/latency numbers, architecture documentation.

**Deliverable:** `EVAL_REPORTS/advanced_rag_eval.json`

---

## Phase 3 Hard Gate
- [ ] Agentic RAG with self-correction loop (LangGraph)
- [ ] Multi-hop reasoning implemented
- [ ] Router with 4+ categories working
- [ ] Multi-modal OR Graph RAG implemented
- [ ] Long-context stress test completed
- [ ] Eval report with end-to-end metrics
