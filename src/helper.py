from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Extract data from pdf:
def load_pdf(data):
  loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
  documents = loader.load()
  return documents 

# data split and chunk creation
def text_splitter(extracted_data):
  text_split=RecursiveCharacterTextSplitter(chunk_size=500 , chunk_overlap=20)
  text_chunks = text_split.split_documents(extracted_data)
  return text_chunks


# function to download embedding model
def download_embedding_model():
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embeddings