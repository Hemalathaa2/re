import streamlit as st
from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    preprocess,
    compute_detailed_score,
    get_embeddings_batch   # ✅ NEW
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
    jd_text = st.text_area("Paste Job Description here", height=200)

# JD Preview
if jd_text:
    st.subheader("📄 JD Preview (Full)")
    st.text_area("Job Description", jd_text, height=300)

# -----------------------------------
# RESUME UPLOAD
# -----------------------------------
st.header("📂 Upload Resumes")

resume_files = st.file_uploader(
    "Upload multiple resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

top_n = st.slider("Select number of top candidates", 1, 50, 5)

# -----------------------------------
# ANALYSIS
# -----------------------------------
if st.button("🔍 Analyze Resumes"):

    if not jd_text.strip():
        st.warning("Please provide Job Description")

    elif not resume_files:
        st.warning("Please upload at least one resume")

    else:
        with st.spinner("⚡ Processing resumes..."):

            # ✅ PREPROCESS JD
            jd_text_clean = preprocess(jd_text)

            # ✅ EXTRACT & PREPROCESS ALL RESUMES FIRST
            resume_texts = []
            resume_names = []

            progress = st.progress(0)

            for i, resume in enumerate(resume_files):
                try:
                    if resume.name.endswith(".pdf"):
                        text = extract_text_from_pdf(resume)
                    else:
                        text = extract_text_from_docx(resume)

                    clean_text = preprocess(text)

                    if len(clean_text.strip()) > 50:  # avoid empty/bad resumes
                        resume_texts.append(clean_text)
                        resume_names.append(resume.name)

                except Exception as e:
                    st.warning(f"⚠️ Error reading {resume.name}, skipped.")

                progress.progress((i + 1) / len(resume_files))

            if not resume_texts:
                st.error("❌ No valid resumes found.")
                st.stop()

            # ✅ BATCH EMBEDDINGS (FAST 🚀)
            jd_emb = get_embeddings_batch([jd_text_clean])[0]
            resume_embs = get_embeddings_batch(resume_texts)

            # -----------------------------------
            # SCORING
            # -----------------------------------
            results = []

            for i in range(len(resume_texts)):

                details = compute_detailed_score(
                    jd_text_clean,
                    resume_texts[i],
                    jd_emb,
                    resume_embs[i]
                )

                results.append({
                    "name": resume_names[i],
                    **details
                })

            # SORT
            results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        # -----------------------------------
        # DISPLAY RESULTS
        # -----------------------------------
        st.success("✅ Analysis Complete!")

        st.subheader("🏆 Top Candidates")

        for i, r in enumerate(results[:top_n]):
            st.markdown(f"### {i+1}. {r['name']}")

            # Scores
            st.progress(r['final_score'])

            st.write(f"⭐ Final Score: {round(r['final_score']*100, 2)}%")
            st.write(f"🧠 Semantic Match: {round(r['semantic_score']*100, 2)}%")
            st.write(f"🛠 Skill Match: {round(r['skill_score']*100, 2)}%")
            st.write(f"🔍 Semantic Skill Match: {round(r['semantic_skill_score']*100, 2)}%")

            # Skills
            st.write(
                f"✅ Matched Skills: {', '.join(r['matched_skills']) if r['matched_skills'] else 'None'}"
            )
            st.write(
                f"❌ Missing Skills: {', '.join(r['missing_skills']) if r['missing_skills'] else 'None'}"
            )

            st.divider()

        # -----------------------------------
        # ALL RESULTS
        # -----------------------------------
        st.subheader("📊 All Results")

        for r in results:
            st.write(f"{r['name']} → {round(r['final_score']*100, 2)}%")
