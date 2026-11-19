"""Multi-Query RAG Project — Phase 2, Week 5 (Query Transformation)

This script demonstrates query rewriting for RAG:
1. Load a PDF and split it into chunks
2. Embed chunks and store in FAISS vector database
3. Take a user question and rewrite it into 3 alternative queries using an LLM
4. Search FAISS for each query and merge unique results
5. Generate a final answer using all retrieved context

Run:
    python projects/phase_2_intermediate/multi_query_rag.py

Requirements:
    .env file with GOOGLE_API_KEY
    uv sync (to install dependencies)
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

# 1. Load environment variables
load_dotenv()

# Path to your PDF document
doc_path = Path("pdfcoffee.com_never-split-the-difference-negotiating-as-if-your-life-depended-on-itpdf-pdf-free.pdf")
index_path = Path("faiss_index")

# 2. Initialize embeddings (local, free — runs on CPU)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 3. Load saved FAISS index if it exists, otherwise build and save it
if index_path.exists():
    vector_store = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )
else:
    loader = PyPDFLoader(str(doc_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(str(index_path))

# 4. Initialize the LLM (used for both query rewriting and answering)
model = init_chat_model(
    model="gemini-3.8-flash",
    model_provider="google_genai",
    temperature=0.7,
)

# 5. Multi-Query Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
query = "How do I stay calm and get someone to agree with me?"

# Step 5a: Ask the LLM to generate 3 alternative search queries
rewrite_prompt = f"""Generate 3 different search queries to find relevant information for this question.
Return ONLY the queries, one per line, with no numbering or bullet points.

Question: {query}
"""
rewrite_response = model.invoke(rewrite_prompt)

# Step 5b: Turn the LLM output into a clean Python list (always keep the original query)
queries = [query]
for line in rewrite_response.content.splitlines():
    cleaned = line.strip()
    if cleaned:
        queries.append(cleaned)

print("Generated queries:")
for q in queries:
    print(" -", q)

# Step 5c: Search FAISS for each query and remove duplicate chunks
seen_contents = set()
unique_docs = []
total_retrieved = 0

for q in queries:
    docs = retriever.invoke(q)
    total_retrieved += len(docs)
    for doc in docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)

print(f"\nRetrieved {total_retrieved} chunks total -> kept {len(unique_docs)} unique chunks.\n")

# 6. Build context and generate the final answer
context_text = "\n\n".join(d.page_content for d in unique_docs)

prompt_content = f"""Only answer according to the given information below.

Context:
{context_text}

Question: {query}
"""

response = model.invoke(prompt_content)
print("Answer:\n", response.content)
