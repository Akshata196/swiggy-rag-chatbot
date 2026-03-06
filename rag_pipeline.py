
"""
Swiggy Annual Report RAG Application
------------------------------------
This script builds a Retrieval-Augmented Generation (RAG) system that answers
questions using the Swiggy Annual Report PDF.

Pipeline:
PDF → Chunking → Embeddings → Vector DB → Retriever → LLM → Answer
"""

import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -------------------------------
# 1. Initialize Groq Client
# -------------------------------

def initialize_llm():
    api_key = os.getenv("GROQ_API_KEY")
    
    

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Set it in environment variables.")

    return Groq(api_key=api_key)


# -------------------------------
# 2. Load PDF
# -------------------------------

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF.")
    return documents


# -------------------------------
# 3. Chunk Text
# -------------------------------

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks


# -------------------------------
# 4. Create Embeddings
# -------------------------------

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")
    return embeddings


# -------------------------------
# 5. Build Vector Database
# -------------------------------

def create_vector_db(chunks, embeddings):

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )
    vector_db.persist()

    print("Vector database created.")
    return vector_db


# -------------------------------
# 6. Create Retriever
# -------------------------------

def create_retriever(vector_db):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    print("Retriever ready.")
    return retriever


# -------------------------------
# 7. RAG Query
# -------------------------------

def generate_answer(client, retriever, query):

    results = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are an AI assistant answering questions about the Swiggy Annual Report.

Use ONLY the provided context to answer.

If the answer is not present in the document, say:
"Answer not found in the document."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# -------------------------------
# 8. Main Application Loop
# -------------------------------

def main():

    print("\n=== Swiggy Annual Report RAG Assistant ===\n")

    client = initialize_llm()

    documents = load_documents("data/swiggy_annual_report.pdf")

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vector_db = create_vector_db(chunks, embeddings)

    retriever = create_retriever(vector_db)

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("Exiting application.")
            break

        answer = generate_answer(client, retriever, query)

        print("\nAnswer:\n")
        print(answer)


# -------------------------------
# Run Program
# -------------------------------

if __name__ == "__main__":
    main()
