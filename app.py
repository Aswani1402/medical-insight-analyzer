# app.py

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from insight_engine import generate_insight, advanced_nlp_insight
from rag_module import build_rag
from transformers import pipeline
from glob import glob

# Load data
caption_df = pd.read_csv(r"C:\Users\Aswini_Ayappan\OneDrive - Amrita university\3rd year\sem 6\nlp_projectnote4\data\patient_10007_progression_captions.csv")
ner_df = pd.read_json(r"C:\Users\Aswini_Ayappan\OneDrive - Amrita university\3rd year\sem 6\nlp_projectnote4\data\organ_severity_output.json")
re_df = pd.read_json(r"C:\Users\Aswini_Ayappan\OneDrive - Amrita university\3rd year\sem 6\nlp_projectnote4\data\all_medical_relations_structured_only.json")

# Rename columns to ensure merging works
ner_df.rename(columns={"filename": "Image Index", "disease": "Disease", "organ": "Organ"}, inplace=True)
re_df.rename(columns={"filename": "Image Index"}, inplace=True)

# Optional debug (see columns)
st.write("✅ Caption DF columns:", caption_df.columns.tolist())
st.write("✅ NER DF columns:", ner_df.columns.tolist())
st.write("✅ RE DF columns:", re_df.columns.tolist())

# Merge all
merged_df = caption_df.merge(ner_df, on="Image Index", how="left")
merged_df = merged_df.merge(re_df, on="Image Index", how="left")

# Add insights
merged_df["Insight"] = merged_df.apply(lambda r: generate_insight(r["Disease"], r["Organ"], r["Caption"]), axis=1)
merged_df["Advanced Insight"] = merged_df.apply(lambda r: advanced_nlp_insight(r["Caption"], r["Disease"]), axis=1)

# Build RAG model
qa_chain = build_rag()

# Optional summarization model (LLM)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🩺 Medical Insight Analyzer (Notebook 5 Final)")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🖼️ Scan Viewer", "🧠 NLP & LLM", "📚 RAG QA", "📊 Summary Charts", "🧾 Documentation"])

with tab1:
    st.header("📌 Select a Scan")
    selected_scan = st.selectbox("Choose Scan:", merged_df["Scan"])
    row = merged_df[merged_df["Scan"] == selected_scan].iloc[0]


    image_index = row['Image Index'].replace('.png', '').replace('.jpg', '')
    search_path = f"data/images_*/images/{image_index}.*"
    matched_files = glob(search_path)

    if matched_files:
        st.image(matched_files[0], width=350, caption="Chest X-ray")
    else:
        st.warning(f"Image for {image_index} not found in subfolders.")



    st.subheader("🧬 Caption")
    st.code(row["Caption"])

    st.subheader("🦠 Disease & Organ")
    st.write(f"**Disease:** {row['Disease']}")
    st.write(f"**Organ:** {row['Organ']}")

with tab2:
    st.header("🧠 NLP Insight")
    st.info(row["Insight"])

    st.subheader("🔍 Advanced Caption Analysis")
    st.success(row["Advanced Insight"])

    st.subheader("📝 LLM-Based Patient Summary (All Captions)")
    patient_rows = merged_df[merged_df["Scan"].str.contains(row["Scan"].split()[0])]
    all_captions = " ".join(patient_rows["Caption"].tolist())
    if all_captions.strip():
        summary = summarizer(all_captions, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        st.write(summary)
    else:
        st.warning("No captions available.")

with tab3:
    st.header("💬 Ask a Medical Question (RAG)")
    user_question = st.text_input("Ask about this condition (e.g., treatment for cardiomegaly):")
    if user_question:
        answer = qa_chain.run(user_question)
        st.success(answer)

with tab4:
    st.header("📊 Disease Distribution")
    disease_counts = merged_df["Disease"].value_counts()
    st.bar_chart(disease_counts)

    st.subheader("🗂️ Full Data Table")
    st.dataframe(merged_df[["Scan", "Caption", "Disease", "Insight", "Advanced Insight"]])

    st.download_button("📤 Download Full CSV", merged_df.to_csv(index=False), file_name="final_patient_insights.csv")

with tab5:
    st.header("📄 Documentation & Explanation")
    st.markdown("""
    ### 📘 Overview
    This tool combines medical NLP, LLMs, and RAG to generate insights from chest X-rays using NIH data.

    ### 🔬 NLP Pipeline
    - Caption ➜ Named Entity Recognition (NER)
    - Relation Extraction ➜ Disease–Organ pairs
    - Insight Generation ➜ Rule-based and template-driven

    ### 🧠 LLM Technologies
    - `facebook/bart-large-cnn` for summarization
    - `gpt2` via LangChain for text generation
    - FAISS vector database for Retrieval-Augmented Generation (RAG)

    ### 📚 RAG Use Case
    A small medical knowledge base is indexed using FAISS.
    Queries like "What is pleural effusion?" return generated answers.

    ### 🖼️ Dataset
    NIH Chest X-ray Dataset (Sample subset) with ~5–10 scans, JSON metadata.

    ---  
    ✅ Developed using: `streamlit`, `pandas`, `transformers`, `langchain`, `faiss-cpu`, `sentence-transformers`
    """)

