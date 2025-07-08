# rag_module.py

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def build_rag():
    # Example knowledge base (you can expand this)
    docs = [
        "Cardiomegaly is enlargement of the heart and may require echocardiogram.",
        "Pneumonia treatment involves antibiotics and monitoring.",
        "Pleural effusion may require drainage and imaging follow-up."
    ]

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector index
    vector_db = FAISS.from_texts(docs, embeddings)

    # Define basic LLM pipeline
    qa_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Retrieval-based QA chain
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())
