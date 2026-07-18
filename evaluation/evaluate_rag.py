import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Load PDF
# -----------------------------
loader = PyPDFLoader("data/swiggy_annual_report.pdf")
documents = loader.load()

# -----------------------------
# Split Documents
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

# -----------------------------
# Embeddings
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load Existing Vector DB
# -----------------------------
vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k":3})

# -----------------------------
# Load Evaluation Dataset
# -----------------------------
df = pd.read_csv("evaluation/rag_evaluation_dataset.csv")

generated_answers = []
retrieved_contexts = []

print(f"Total Questions: {len(df)}")

# -----------------------------
# Run Evaluation
# -----------------------------
for i, row in df.iterrows():

    question = row["question"]

    print(f"\nProcessing {i+1}/{len(df)}")

    # Retrieve Context
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt
    prompt = f"""
Answer ONLY using the context below.

If the answer is not present, reply:
Answer not found in the document.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    generated_answers.append(answer)
    retrieved_contexts.append(context)

# -----------------------------
# Save Results
# -----------------------------
df["generated_answer"] = generated_answers
df["retrieved_context"] = retrieved_contexts

df.to_csv(
    "evaluation/evaluation_results.csv",
    index=False
)

print("\nEvaluation file created successfully!")
print("Saved as evaluation/evaluation_results.csv")