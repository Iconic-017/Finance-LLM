import os
import time
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf, text_splitter, download_embedding_model

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --------------------------
# Config
# --------------------------
INDEX_NAME = "finance-chatbot"
NAMESPACE = "finance"
BATCH_SIZE = 64
EMBED_DIM = 384  # all-MiniLM-L6-v2

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found. Please set it in your .env file.")

# --------------------------
# Init Pinecone
# --------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Handle differences in list_indexes return type
existing = pc.list_indexes()
if existing and isinstance(existing[0], dict):
    existing_names = [i.get("name") for i in existing]
else:
    existing_names = existing

# Create index if not present
if INDEX_NAME not in existing_names:
    print(f"⚡ Index '{INDEX_NAME}' not found — creating...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    print("✅ Index created successfully.")
else:
    print(f"ℹ️ Index '{INDEX_NAME}' already exists. Skipping creation.")

index = pc.Index(INDEX_NAME)

# --------------------------
# Load & preprocess documents
# --------------------------
print("📂 Loading PDFs from 'data/' ...")
extracted_docs = load_pdf("data/")
print(f"✅ Loaded {len(extracted_docs)} documents.")

print("✂️ Splitting into chunks...")
text_chunks = text_splitter(extracted_docs)
print(f"📊 Produced {len(text_chunks)} chunks.")

# Deduplicate & clean
texts, metadatas, seen = [], [], set()
for i, doc in enumerate(text_chunks):
    text = (doc.page_content or "").strip()
    if not text:
        continue
    h = hash(text)
    if h in seen:
        continue
    seen.add(h)

    meta = dict(doc.metadata) if hasattr(doc, "metadata") else {}
    meta.update({
        "source": meta.get("source", ""),
        "chunk_id": i,
        "length": len(text),
    })

    texts.append(text)
    metadatas.append(meta)

print(f"🧹 Final chunk count (after cleaning): {len(texts)}")

# --------------------------
# Embeddings
# --------------------------
print("🔍 Loading embedding model...")
embeddings = download_embedding_model()

# --------------------------
# Upload to Pinecone
# --------------------------
print("🚀 Uploading vectors to Pinecone...")
t0 = time.time()

docsearch = PineconeVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
    index_name=INDEX_NAME,
    metadatas=metadatas,
    namespace=NAMESPACE,
    batch_size=BATCH_SIZE,
)

t1 = time.time()
print(f"✅ Upload completed in {(t1 - t0):.2f}s")
print(f"📦 {len(texts)} vectors indexed in namespace '{NAMESPACE}'.")

# --------------------------
# Sanity check
# --------------------------
if metadatas:
    print("🔎 Sample metadata for first chunk:")
    print(json.dumps(metadatas[0], indent=2))
