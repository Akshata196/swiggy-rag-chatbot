
import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

      


# ---------------- UI ---------------- #

st.set_page_config(
    page_title="Swiggy RAG Chatbot",
    page_icon="📊",
    layout="wide"
)


# -------- Sidebar -------- #

with st.sidebar:
    st.title("📊 Swiggy RAG Assistant")

    st.write("Ask questions about the **Swiggy Annual Report**.")

    st.markdown("### Example Questions")
    st.markdown("""
    - What is Swiggy total income in FY2024?
    - How many cities does Swiggy operate in?
    - What services does Swiggy provide?
    - What was the net loss in FY2024?
    """)

# -------- Main UI -------- #


st.title("📊 Swiggy Annual Report AI Assistant (RAG Chatbot)")
st.markdown("Ask questions about the **Swiggy Annual Report** and get answers directly from the document.")

st.divider()

# ---------------- Load Data ---------------- #

@st.cache_resource
def load_rag_pipeline():

    loader = PyPDFLoader("data/swiggy_annual_report.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    retriever = vector_db.as_retriever(search_kwargs={"k":3})

    return retriever

retriever = load_rag_pipeline()

# ---------------- Chat Interface ---------------- #

query = st.chat_input("Ask something about the Swiggy Annual Report...")

if query:

    st.chat_message("user").write(query)

    results = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Answer the question ONLY using the context below.
If the answer is not present say 'Answer not found in the document'.

Context:
{context}

Question:
{query}
"""

    with st.spinner("Analyzing report..."):

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )

    answer = response.choices[0].message.content

    st.chat_message("assistant").write(answer)

