import streamlit as st
from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    preprocess,
    compute_similarity
)

st.set_page_config(page_title="Resume Shortlister", layout="wide")

st.title("📄 AI Resume Shortlisting System")
st.markdown("Upload resumes or paste job description to find top candidates.")

# -----------------------------------
# JOB DESCRIPTION INPUT
# -----------------------------------
st.header("📌 Job Description Input")

jd_option = st.radio(
    "Choose how to provide Job Description:",
    ["Upload File", "Paste Text"]
)

jd_text = ""

if jd_option == "Upload File":
    jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"])
    if jd_file:
        if jd_file.name.endswith(".pdf"):
            jd_text = extract_text_from_pdf(jd_file)
        elif jd_file.name.endswith(".docx"):
            jd_text = extract_text_from_docx(jd_file)
        else:
            jd_text = jd_file.read().decode("utf-8")

else:
    jd_text = st.text_area("Paste Job Description here")

if jd_text:
    st.subheader("📄 JD Preview")
    st.write(jd_text[:500])
# -----------------------------------
# RESUME UPLOAD
# -----------------------------------
st.header("📂 Upload Resumes")

resume_files = st.file_uploader(
    "Upload multiple resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

top_n = st.slider("Select number of top candidates", 1, 10, 3)

# -----------------------------------
# ANALYSIS
# -----------------------------------
if st.button("🔍 Analyze Resumes"):

    if not jd_text.strip():
        st.warning("Please provide Job Description")
    elif not resume_files:
        st.warning("Please upload at least one resume")
    else:

        with st.spinner("Analyzing resumes..."):

            jd_text_clean = preprocess(jd_text)

            results = []
            progress = st.progress(0)

            for i, resume in enumerate(resume_files):

                if resume.name.endswith(".pdf"):
                    resume_text = extract_text_from_pdf(resume)
                else:
                    resume_text = extract_text_from_docx(resume)

                resume_text_clean = preprocess(resume_text)

                score = compute_similarity(jd_text_clean, resume_text_clean)

                results.append((resume.name, score))

                progress.progress((i + 1) / len(resume_files))

            # -----------------------------------
            # SORT RESULTS
            # -----------------------------------
            results = sorted(results, key=lambda x: x[1], reverse=True)

        # -----------------------------------
        # DISPLAY RESULTS
        # -----------------------------------
        st.success("✅ Analysis Complete!")

        st.subheader("🏆 Top Candidates")
        for i, (name, score) in enumerate(results[:top_n]):
            st.write(f"{i+1}. {name} → {round(score*100, 2)}%")

        st.subheader("📊 All Results")
        for name, score in results:
            st.write(f"{name} → {round(score*100, 2)}%")
