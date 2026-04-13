import streamlit as st
import requests

st.set_page_config(page_title="AI Resume Shortlister", layout="wide")

# -------------------------------
# CUSTOM STYLING
# -------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #4CAF50;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #0e1117;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}
.score {
    font-size: 24px;
    font-weight: bold;
}
.skill-good {
    color: #4CAF50;
}
.skill-bad {
    color: #FF4B4B;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<p class="main-title">📄 AI Resume Shortlisting Dashboard</p>', unsafe_allow_html=True)
st.caption("Smart AI-powered hiring assistant with explainable insights")

# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    jd_text = st.text_area("📌 Job Description", height=200)

with col2:
    resume_files = st.file_uploader(
        "📂 Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

top_n = st.slider("🎯 Number of Top Candidates", 1, 20, 5)

# -------------------------------
# ANALYZE BUTTON
# -------------------------------
if st.button("🚀 Analyze Candidates"):

    if not jd_text or not resume_files:
        st.warning("⚠️ Please provide Job Description and Resumes")
        st.stop()

    with st.spinner("⚡ AI is analyzing resumes..."):

        files = [("files", (f.name, f, f.type)) for f in resume_files]

        response = requests.post(
            "http://localhost:8000/analyze/",
            files=files,
            data={"jd_text": jd_text}
        )

        results = response.json()["results"]

    st.success("✅ Analysis Completed!")

    # -------------------------------
    # TOP CANDIDATE HIGHLIGHT
    # -------------------------------
    top = results[0]

    st.subheader("🏆 Top Candidate")

    st.markdown(f"""
    <div class="card">
        <h3>{top['name']}</h3>
        <p class="score">Score: {round(top['final_score']*100,2)}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(top['final_score'])

    if "llm_explanation" in top:
        st.subheader("🧠 AI Recruiter Insight")
        st.info(top["llm_explanation"])

    # -------------------------------
    # ALL CANDIDATES
    # -------------------------------
    st.subheader("📊 Candidate Rankings")

    for i, r in enumerate(results[:top_n]):

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <div class="card">
                <h4>#{i+1} {r['name']}</h4>
                <p class="score">Score: {round(r['final_score']*100,2)}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(r['final_score'])

        with col2:
            st.write("🧠 Semantic:", f"{round(r['semantic_score']*100,2)}%")
            st.write("🛠 Skills:", f"{round(r['skill_score']*100,2)}%")

        # Skills display
        st.markdown("**✅ Matched Skills**")
        if r["matched_skills"]:
            st.markdown(
                " ".join([f"<span class='skill-good'>✔ {s}</span>" for s in r["matched_skills"]]),
                unsafe_allow_html=True
            )
        else:
            st.write("None")

        st.markdown("**❌ Missing Skills**")
        if r["missing_skills"]:
            st.markdown(
                " ".join([f"<span class='skill-bad'>✖ {s}</span>" for s in r["missing_skills"]]),
                unsafe_allow_html=True
            )
        else:
            st.write("None")

        st.divider()
