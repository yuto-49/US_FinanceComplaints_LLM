from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama

# 1. Load embedding model (same one used to create the index)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load vectorstore
vectorstore = FAISS.load_local(
    "vectorstore_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# 3. Connect to Ollama running DeepSeek
llm = ChatOllama(model="deepseek-coder")

# 4. Set up RAG-style chatbot
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 5. Chat loop
print("🧠 Chatbot is ready! Ask anything based on the complaint dataset.")
while True:
    query = input("\n🧑 You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.invoke({"query": query})
    print(f"🤖 Bot: {response['result']}")
    print("📚 Retrieved documents:")
    for doc in response['source_documents']:
         print("-", doc.page_content[:200])  # Show first 100 chars
    print("🔍 End of documents.")
