import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# 🔥 Replace with your actual deployed API URL
API_URL = "https://re-m8x0.onrender.com/analyze/"

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<h1 style='text-align:center;'>🚀 AI Hiring Dashboard</h1>
""", unsafe_allow_html=True)

# -------------------------------
# JD INPUT
# -------------------------------
st.markdown("### 📌 Job Description")

jd_option = st.radio("Choose input method:", ["Paste Text", "Upload File"])
jd_text = ""

if jd_option == "Paste Text":
    jd_text = st.text_area("Paste Job Description", height=120)
else:
    jd_file = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])

    if jd_file:
        if jd_file.name.endswith(".pdf"):
            jd_text = jd_file.read().decode("utf-8", errors="ignore")
        elif jd_file.name.endswith(".docx"):
            jd_text = jd_file.read().decode("utf-8", errors="ignore")
        else:
            jd_text = jd_file.read().decode("utf-8", errors="ignore")

# -------------------------------
# RESUME UPLOAD
# -------------------------------
st.markdown("### 📂 Upload Resumes")

resume_files = st.file_uploader(
    "Upload resumes",
    type=["pdf","docx"],
    accept_multiple_files=True
)

# -------------------------------
# JOB OPENINGS
# -------------------------------
job_openings = st.number_input(
    "👥 Number of openings",
    min_value=1,
    max_value=50,
    value=5
)

# -------------------------------
# ANALYSIS (API BASED)
# -------------------------------
if st.button("Analyze Candidates"):

    if not jd_text or not resume_files:
        st.warning("Provide JD and resumes")
        st.stop()

    with st.spinner("Processing..."):

        try:
            files = [("files", (f.name, f, f.type)) for f in resume_files]

            response = requests.post(
                API_URL,
                files=files,
                data={"jd_text": jd_text},
                timeout=120
            )

            if response.status_code != 200:
                st.error("API Error: Unable to process resumes")
                st.stop()

            results = response.json().get("results", [])

        except Exception as e:
            st.error(f"Connection failed: {e}")
            st.stop()

    if not results:
        st.warning("No valid resumes processed")
        st.stop()

    st.success("Analysis Complete")

    # -------------------------------
    # METRICS
    # -------------------------------
    top_score = round(results[0]["final_score"]*100, 2)
    avg_score = round(sum(r["final_score"] for r in results)/len(results)*100, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Top Score", f"{top_score:.2f}%")
    c2.metric("Avg Score", f"{avg_score:.2f}%")
    c3.metric("Candidates", len(results))

    # -------------------------------
    # BEST CANDIDATE
    # -------------------------------
    top = results[0]

    st.subheader("Best Candidate")
    st.markdown(f"""
    <div class="card">
    <b>{top['name']}</b><br>
    Score: {top['final_score']*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(top["final_score"]))

    st.write(top.get("llm_explanation", "No explanation available"))

    # -------------------------------
    # SHORTLIST
    # -------------------------------
    st.subheader("Shortlisted Candidates")

    for i, r in enumerate(results[:job_openings]):

        # Verdict
        if r["final_score"] > 0.7:
            verdict = "🟢 Strong Match"
        elif r["final_score"] > 0.4:
            verdict = "🟡 Moderate Match"
        else:
            verdict = "🔴 Low Match"

        st.markdown(f"""
        <div class="card">
        <b>#{i+1} {r['name']}</b><br>
        Score: {r['final_score']*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(r["final_score"]))

        # Skills
        st.write("Matched Skills:", ", ".join(r.get("matched_skills", [])) or "None")
        st.write("Missing Skills:", ", ".join(r.get("missing_skills", [])) or "None")

        # Scores
        st.write(f"Semantic: {r['semantic_score']*100:.2f}%")
        st.write(f"Skill: {r['skill_score']*100:.2f}%")
        st.write(f"Experience: {r.get('experience_score', 0)*100:.2f}%")

        st.write("Verdict:", verdict)

        # AI Explanation
        with st.expander("AI Explanation"):
            st.write(r.get("llm_explanation", "Not available"))

        st.divider()

    # -------------------------------
    # TABLE + CHART
    # -------------------------------
    df = pd.DataFrame(results)

    st.subheader("Comparison Table")
    st.dataframe(df, use_container_width=True)

    st.subheader("Score Chart")
    st.bar_chart(df.set_index("name")["final_score"])

    st.download_button("Download CSV", df.to_csv(index=False), "results.csv")
