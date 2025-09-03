import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import CTransformers

from src.helper import download_embedding_model, get_financial_data
from src.prompt import prompt_template


# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- Environment -------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "finance-chatbot"
NAMESPACE = "finance"

# ------------------- Embeddings -------------------
embeddings = download_embedding_model()

# ------------------- Pinecone VectorStore -------------------
print("🔗 Connecting to Pinecone index...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

# ------------------- Prompt -------------------
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# ------------------- Local LLM -------------------
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",   # make sure model file exists
    model_type="llama",
    config={
        "max_new_tokens": 512,
        "temperature": 0.5,      # lower temp = more deterministic answers
    },
    local_files_only=True
)

# ------------------- Optimized Retriever -------------------
retriever = docsearch.as_retriever(
    search_type="mmr",   # ✅ Maximal Marginal Relevance: balances relevance + diversity
    search_kwargs={
        "k": 3,          # ✅ return top 3 chunks (speed + accuracy)
        "fetch_k": 20,   # ✅ fetch 20, then rerank (tradeoff between speed & context)
    }
)

# ------------------- RetrievalQA Chain -------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


# ------------------- Routes -------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]

    try:
        # --- Quick check: if user asks about stock price/market data, skip RAG ---
        keywords = ["price", "stock", "market cap", "pe ratio", "dividend"]
        if any(word in user_input.lower() for word in keywords):
            fallback = get_financial_data(user_input)
            return fallback

        # --- Otherwise, go through RAG ---
        result = qa.invoke({"query": user_input})
        response = result["result"]

        # If LLM doesn't know, fallback
        if "i don't know" in response.lower():
            fallback = get_financial_data(user_input)
            return fallback

        return response

    except Exception as e:
        return f"An error occurred: {str(e)}"



# ------------------- Run -------------------
if __name__ == "__main__":
    app.run(debug=True)
