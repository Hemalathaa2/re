import streamlit as st
import requests

st.set_page_config(page_title="Resume Shortlister", layout="wide")

st.title("📄 AI Resume Shortlisting System")

jd_text = st.text_area("Paste Job Description", height=200)

resume_files = st.file_uploader(
    "Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True
)

top_n = st.slider("Top candidates", 1, 20, 5)

if st.button("Analyze"):

    if not jd_text or not resume_files:
        st.warning("Provide JD and resumes")
        st.stop()

    with st.spinner("Analyzing..."):

        files = [("files", (f.name, f, f.type)) for f in resume_files]

        response = requests.post(
            "http://localhost:8000/analyze/",
            files=files,
            data={"jd_text": jd_text}
        )

        results = response.json()["results"]

    st.success("Done")

    for i, r in enumerate(results[:top_n]):

        st.subheader(f"{i+1}. {r['name']}")
        st.progress(r["final_score"])

        st.write(f"Score: {round(r['final_score']*100,2)}%")
        st.write(f"Skills Match: {round(r['skill_score']*100,2)}%")

        st.write("Matched:", r["matched_skills"])
        st.write("Missing:", r["missing_skills"])

        if i == 0 and "llm_explanation" in r:
            st.subheader("🧠 AI Insight")
            st.info(r["llm_explanation"])
