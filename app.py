from flask import Flask, render_template, request
from helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from prompt import system_prompt
import os

app = Flask(__name__)

# Step 1: Load and process PDF
pdf_path = "data"
documents = load_pdf_file(pdf_path)
chunks = text_split(documents)

# Step 2: Embeddings
embeddings = download_hugging_face_embeddings()

# Step 3: Create or Load FAISS index
index_path = "faiss_index"

if os.path.exists(index_path):
    print("🔁 Loading existing FAISS index...")
    docsearch = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("🧠 Creating FAISS index for the first time...")
    docsearch = FAISS.from_documents(chunks, embeddings)
    docsearch.save_local(index_path)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Step 4: LLM using Ollama (local)
llm = Ollama(model="mistral")

# Step 5: Prompt + RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    print("User:", user_msg)
    response = rag_chain.invoke({"input": user_msg})
    bot_reply = response["answer"]
    print("Bot:", bot_reply)
    return bot_reply

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
