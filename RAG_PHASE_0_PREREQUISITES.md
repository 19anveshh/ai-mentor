# Phase 0 — Prerequisites (Weeks 1–2)

## Goal
Build the foundational skills you need before touching RAG: Python fluency, ML fundamentals, LLM basics, vector database concepts, and framework basics.

---

## Week 1 — Python Fluency

### Topics

| Sub-topic | What to learn | Why it matters for RAG |
|-----------|--------------|----------------------|
| Data structures | Lists, dicts, sets, tuples, comprehensions, `collections` module | You'll manipulate document metadata and chunks constantly |
| Functions | Type hints, docstrings, `*args`/`**kwargs`, decorators, `async def` | RAG pipelines use async for concurrent embedding and retrieval |
| OOP | Classes, `__init__`, dunder methods, inheritance, `@property` | You'll wrap models and retrievers in classes (like `ModelWrapper`) |
| Virtual envs | `uv venv`, `uv pip install`, `requirements.txt`, `.venv` | Isolate dependencies for each RAG experiment |
| File I/O | `pathlib`, reading/writing text/JSON/CSV, glob patterns | Load documents from disk, save eval results |
| Error handling | `try/except`, custom exceptions, logging | Production RAG needs robust error handling |

### Daily Tasks

**Day 1:** Python data structures + comprehensions
- Write a script that processes a list of documents and groups them by metadata category

**Day 2:** Functions with type hints + OOP
- Create a `Document` class with `page_content`, `metadata`, and a `.word_count()` property

**Day 3:** Async/await + virtual environments
- Set up `.venv` with `uv`, install `langchain`, write an async function that calls an embedding model

**Day 4:** File I/O + pathlib
- Build a document loader that reads all `.txt` and `.md` files from a folder

**Day 5:** Error handling + logging
- Add try/except and logging to all previous scripts

**Days 6–7:** Sunday ritual (60 min) — Review → Plan → Learn → Commit

### Deliverable
- `prerequisites/python_basics/` folder with 5+ scripts
- `requirements.txt` with all dependencies
- Git repo with daily commits

---

## Week 2 — ML Fundamentals + LLM Basics

### Topics

| Sub-topic | What to learn | Why it matters for RAG |
|-----------|--------------|----------------------|
| Embeddings | What are embeddings, cosine similarity, Euclidean distance, dimensionality | The core of vector search in RAG |
| Nearest neighbors | k-NN, approximate nearest neighbor (ANN), HNSW, IVF | How vector stores find similar chunks |
| Train/test split | Why we split, stratification, data leakage (you already saw this) | Evaluating RAG retrieval quality |
| Metrics | Accuracy, precision, recall, F1, RMSE, MRR, NDCG, hit@k | Measuring RAG performance |
| Tokenization | Token vs word, subword tokenization, context window | Chunking strategy and context limits |
| LLM basics | Temperature, top-p, system prompts, few-shot, hallucination | Generation step in RAG |
| Prompt engineering | Zero-shot vs few-shot, chain-of-thought, RAG prompt templates | Getting good answers from the LLM |

### Daily Tasks

**Day 1:** Embeddings + cosine similarity
- Use `sentence-transformers` to embed 10 sentences, compute cosine similarity matrix

**Day 2:** Train/test split + metrics
- Revisit your iris pipeline; compute precision, recall, F1 per class

**Day 3:** Tokenization + context window
- Count tokens in a document using `tiktoken`, demonstrate chunking

**Day 4:** LLM prompting
- Write system prompts for a RAG QA bot, test few-shot examples

**Day 5:** Vector database concept
- Load 100 docs into Chroma or FAISS, run similarity search

**Days 6–7:** Sunday ritual + commit all work

### Deliverable
- `prerequisites/ml_fundamentals/` notebooks/scripts
- Embedding similarity demo
- Vector DB quickstart with 100 documents

---

## Exit Criteria for Phase 0
- [ ] Can write Python classes with type hints and async
- [ ] Can explain embeddings and cosine similarity
- [ ] Can load documents, split them, embed them, and query a vector store
- [ ] Can explain when to use accuracy vs F1 vs RMSE
- [ ] Git repo with consistent commits

---

## Recommended Resources
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Chollet) — ML basics
- "The Annotated Transformer" — attention/transformers
- LangChain documentation — framework basics
- Chroma/FAISS documentation — vector stores
