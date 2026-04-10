import streamlit as st
import os
from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    preprocess,
    compute_similarity
)

st.set_page_config(page_title="Resume Shortlister", layout="wide")

st.title("📄 AI Resume Shortlisting System")

# Upload JD
jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])

# Upload resumes
resume_files = st.file_uploader(
    "Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True
)

top_n = st.slider("Select number of top candidates", 1, 10, 3)

if st.button("🔍 Analyze Resumes"):

    if not jd_file or not resume_files:
        st.warning("Please upload JD and resumes")
    else:
        # -------------------------------
        # Process JD
        # -------------------------------
        if jd_file.name.endswith(".pdf"):
            jd_text = extract_text_from_pdf(jd_file)
        elif jd_file.name.endswith(".docx"):
            jd_text = extract_text_from_docx(jd_file)
        else:
            jd_text = jd_file.read().decode("utf-8")

        jd_text = preprocess(jd_text)

        results = []

        # -------------------------------
        # Process Resumes
        # -------------------------------
        for resume in resume_files:

            if resume.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(resume)
            else:
                resume_text = extract_text_from_docx(resume)

            resume_text = preprocess(resume_text)

            score = compute_similarity(jd_text, resume_text)

            results.append((resume.name, score))

        # -------------------------------
        # Ranking
        # -------------------------------
        results = sorted(results, key=lambda x: x[1], reverse=True)

        st.subheader("🏆 Top Candidates")

        for i, (name, score) in enumerate(results[:top_n]):
            st.write(f"{i+1}. {name} → {round(score*100, 2)}%")

        st.subheader("📊 All Results")

        for name, score in results:
            st.write(f"{name} → {round(score*100, 2)}%")
