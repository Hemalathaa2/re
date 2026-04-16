import streamlit as st
import requests
import time
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
API_URL = "https://re-m8x0.onrender.com"

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🚀 AI Hiring Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------
# JD INPUT
# -------------------------------
st.subheader("📌 Job Description")

jd_option = st.radio("Choose input method:", ["Paste Text", "Upload File"])
jd_text = ""

if jd_option == "Paste Text":
    jd_text = st.text_area("Paste Job Description", height=150)
else:
    jd_file = st.file_uploader("Upload Job Description", type=["txt"])
    if jd_file:
        jd_text = jd_file.read().decode("utf-8", errors="ignore")

# -------------------------------
# RESUME UPLOAD
# -------------------------------
st.subheader("📂 Upload Resumes")

files = st.file_uploader(
    "Upload resumes",
    type=["pdf", "docx"],
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
# ANALYZE BUTTON
# -------------------------------
if st.button("🚀 Analyze Candidates"):

    if not jd_text or not files:
        st.warning("Please provide JD and resumes")
        st.stop()

    # STEP 1: Submit Job
    with st.spinner("Submitting job..."):
        response = requests.post(
            f"{API_URL}/analyze/",
            files=[("files", (f.name, f, f.type)) for f in files],
            data={"jd_text": jd_text}
        )

        if response.status_code != 200:
            st.error("Failed to start analysis")
            st.stop()

        job_id = response.json()["job_id"]

    st.info("⏳ Processing started... Please wait")

    # STEP 2: Polling
    results = None

    progress_bar = st.progress(0)

    for i in range(30):  # max ~150 sec
        time.sleep(5)

        res = requests.get(f"{API_URL}/result/{job_id}")
        data = res.json().get("results", [])

        progress_bar.progress((i + 1) / 30)

        if data:
            results = data
            break

    if not results:
        st.warning("⚠️ Still processing. Try again later.")
        st.stop()

    st.success("✅ Analysis Complete!")

    # -------------------------------
    # METRICS
    # -------------------------------
    top_score = round(results[0]["final_score"] * 100, 2)
    avg_score = round(sum(r["final_score"] for r in results) / len(results) * 100, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Top Score", f"{top_score}%")
    c2.metric("Avg Score", f"{avg_score}%")
    c3.metric("Candidates", len(results))

    # -------------------------------
    # BEST CANDIDATE
    # -------------------------------
    top = results[0]

    st.subheader("🏆 Best Candidate")
    st.write(f"**{top['name']}**")
    st.progress(float(top["final_score"]))
    st.write(top.get("llm_explanation", "No explanation"))

    # -------------------------------
    # SHORTLIST
    # -------------------------------
    st.subheader("📋 Shortlisted Candidates")

    for i, r in enumerate(results[:job_openings]):
        st.markdown(f"### #{i+1} {r['name']}")
        st.progress(float(r["final_score"]))

        st.write("Matched Skills:", ", ".join(r.get("matched_skills", [])) or "None")
        st.write("Missing Skills:", ", ".join(r.get("missing_skills", [])) or "None")

        st.write(f"Score: {r['final_score']*100:.2f}%")

        with st.expander("AI Explanation"):
            st.write(r.get("llm_explanation", "Not available"))

        st.divider()

    # -------------------------------
    # TABLE + DOWNLOAD
    # -------------------------------
    df = pd.DataFrame(results)

    st.subheader("📊 Comparison Table")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "results.csv"
    )
