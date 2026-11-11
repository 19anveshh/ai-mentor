# Week 2 - Basic RAG (Retrieval Pipeline)
# Steps: Load PDF -> Split chunks -> Embed -> Store in FAISS -> Search -> Display

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Store chunks and embeddings in FAISS
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 5. Search
results = vector_store.similarity_search(
    "What is machine learning?",
    k=3
)

# 6. Display results
for result in results:
    print(result.page_content)
    print(result.metadata)
    print("----------------")
