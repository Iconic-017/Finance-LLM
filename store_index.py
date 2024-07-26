from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from src.helper import load_pdf , text_splitter , download_embedding_model


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
index_name = "medical-chatbot"

#data extraction
extracted_data = load_pdf("data/")

# storing text chunks
text_chunks = text_splitter(extracted_data)

# embedding model object
embeddings = download_embedding_model()

# intializing the pinecone
pc = Pinecone(api_key="1840affb-9ae0-426c-920e-df290252ede2")
index = pc.Index("medical-chatbot")
#creating embeddings for each of the text_chunks and storing in pinecorn_db
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings , index_name=index_name)
