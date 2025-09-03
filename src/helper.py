import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import yfinance as yf


import nltk

# Ensure required tokenizers are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")




# ✅ Download NLTK resources (only runs once)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------
# 1) Load PDFs
# ---------------------------
def load_pdf(data_dir):
    """
    Load all PDFs from a directory.
    """
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# ---------------------------
# 2) Preprocess text
# ---------------------------
def preprocess_text(text):
    """
    Clean text: remove special chars, lowercase, remove stopwords, lemmatize.
    """
    # Lowercase
    text = text.lower()

    # Remove non-alphanumeric chars (except . , ? !)
    text = re.sub(r"[^a-z0-9\s.,?!]", " ", text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords + lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words
    ]

    return " ".join(cleaned_tokens)


# ---------------------------
# 3) Split into chunks
# ---------------------------
def text_splitter(extracted_data):
    """
    Split cleaned documents into chunks with semantic-friendly size.
    """
    # Clean each doc before splitting
    for doc in extracted_data:
        doc.page_content = preprocess_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,    # smaller chunks → faster retrieval
        chunk_overlap=50
    )
    text_chunks = splitter.split_documents(extracted_data)
    return text_chunks


# ---------------------------
# 4) Embedding model
# ---------------------------
def download_embedding_model():
    """
    Load sentence-transformers embedding model.
    Optimized for CPU, normalizes embeddings for cosine similarity.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings



def get_financial_data(query: str) -> str:
    """
    Try to fetch live financial data using yfinance as a fallback
    when the RAG system doesn't know the answer.
    Supports stock prices, market cap, PE ratio, etc.
    """
    try:
        # --- Extract ticker symbol from query ---
        match = re.search(r"\b[A-Z]{1,5}\b", query)
        if not match:
            return "Sorry, I couldn't identify a ticker symbol in your question."

        ticker = match.group(0)
        stock = yf.Ticker(ticker)
        info = stock.info

        query_lower = query.lower()
        response = None

        # --- Handle common financial queries ---
        if "price" in query_lower or "stock" in query_lower:
            response = f"{ticker} current price is ${info.get('currentPrice', 'N/A')}"
        elif "market cap" in query_lower:
            response = f"{ticker} market cap is {info.get('marketCap', 'N/A')}"
        elif "pe ratio" in query_lower:
            response = f"{ticker} P/E ratio is {info.get('trailingPE', 'N/A')}"
        elif "dividend" in query_lower:
            response = f"{ticker} dividend yield is {info.get('dividendYield', 'N/A')}"
        else:
            response = f"I found some data for {ticker}: Price ${info.get('currentPrice', 'N/A')}, Market Cap {info.get('marketCap', 'N/A')}"

        return response

    except Exception as e:
        return f"⚠️ Could not fetch financial data: {str(e)}"
