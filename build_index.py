import numpy
import os
import sys
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Step 1: Load CSV
df = pd.read_csv("./data/consumer_complaints.csv", low_memory=False)

# Step 2: Combine all columns into a single text block per row
def row_to_text(row):
    return " | ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])

texts = df.apply(row_to_text, axis=1).tolist()

# Step 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.create_documents(texts, metadatas=[{"source": f"row_{i}"} for i in range(len(texts))])

# Step 4: Convert to embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Starting embedding...")
vectorstore = FAISS.from_documents(documents, embedding_model)

# Step 5: Save the vectorstore
vectorstore.save_local("vectorstore_index")
print("✅ Embedding complete!")
print("✅ FAISS vectorstore created and saved to 'vectorstore_index/'")