from flask import Flask , render_template , request , jsonify
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_community.llms import CTransformers
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embedding_model
from dotenv import load_dotenv
from src.prompt import *
import os

app =Flask(__name__)  # defining flask object

load_dotenv()
os.environ['PINECONE_API_KEY'] = "1840affb-9ae0-426c-920e-df290252ede2"
index_name = "medical-chatbot"

# embedding model object
embeddings = download_embedding_model()

# intializing the pinecone
pc = Pinecone(api_key="1840affb-9ae0-426c-920e-df290252ede2")
index = pc.Index("medical-chatbot")

#loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name,embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs = {"prompt":PROMPT}

# loading our model:
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type = "llama",
                    config = {'max_new_tokens':512,
                              'temperature':0.8})

# defining question-answer object
qa = RetrievalQA.from_chain_type(
  llm = llm,
  chain_type = "stuff",
  retriever = docsearch.as_retriever(search_kwargs = {'k' : 2}),
  return_source_documents = True,
  chain_type_kwargs = chain_type_kwargs
)

@app.route("/")
def index():
  return render_template("chat.html")

@app.route("/get", methods = ["GET","POST"])
def chat():
  msg = request.form["msg"]
  input = msg
  print(input)
  result = qa.invoke({"query": input})
  print("response: ",result["result"])
  return str(result["result"])

if(__name__ == "__main__"):
  app.run(debug=True)

