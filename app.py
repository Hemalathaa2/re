import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# -------------------------------
# CONFIG
# -------------------------------
MAX_FILE_SIZE_MB = 100

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>
body { background-color: #0f172a; }
.main-title {
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 15px;
}
.metric { font-size: 22px; font-weight: bold; }
.good { color: #4ade80; }
.bad { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<p class="main-title">🚀 AI Hiring Dashboard</p>', unsafe_allow_html=True)
st.caption("AI-powered resume screening with explainable insights")

# -------------------------------
# INPUT
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

# -------------------------------
# FILE SIZE FILTER
# -------------------------------
valid_files = []
if resume_files:
    for f in resume_files:
        size_mb = f.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.warning(f"⚠️ {f.name} exceeds 100MB and was skipped.")
        else:
            valid_files.append(f)

resume_files = valid_files

if resume_files:
    st.success(f"✅ {len(resume_files)} resumes uploaded")

# -------------------------------
# SHORTLIST INPUT (MAX 50)
# -------------------------------
top_n = st.number_input(
    "🎯 Enter number of candidates to shortlist (Max 50)",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

# -------------------------------
# ANALYZE
# -------------------------------
if st.button("⚡ Analyze Candidates"):

    if not jd_text or not resume_files:
        st.warning("Provide JD and resumes")
        st.stop()

    with st.spinner("🤖 AI analyzing resumes..."):

        files = [("files", (f.name, f, f.type)) for f in resume_files]

        response = requests.post(
            "http://localhost:8000/analyze/",
            files=files,
            data={"jd_text": jd_text}
        )

        results = response.json()["results"]

    st.success("✅ Analysis Completed!")

    # -------------------------------
    # METRICS
    # -------------------------------
    top_score = round(results[0]["final_score"] * 100, 2)
    avg_score = round(sum(r["final_score"] for r in results)/len(results) * 100, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Top Score", f"{top_score}%")
    c2.metric("📊 Avg Score", f"{avg_score}%")
    c3.metric("📁 Candidates", len(results))

    # -------------------------------
    # TOP CANDIDATE
    # -------------------------------
    top = results[0]

    st.subheader("🥇 Best Candidate")
    st.markdown(f"""
    <div class="card">
        <h2>{top['name']}</h2>
        <p class="metric">{round(top['final_score']*100,2)}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(top["final_score"])

    if "llm_explanation" in top:
        st.subheader("🧠 AI Insight")
        st.info(top["llm_explanation"])

    # -------------------------------
    # TABS
    # -------------------------------
    tab1, tab2 = st.tabs(["📊 Rankings", "📈 Insights"])

    with tab1:
        for i, r in enumerate(results[:top_n]):

            st.markdown(f"""
            <div class="card">
                <h4>#{i+1} {r['name']}</h4>
                <p class="metric">{round(r['final_score']*100,2)}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(r["final_score"])

            c1, c2 = st.columns(2)
            c1.write(f"🧠 Semantic: {round(r['semantic_score']*100,2)}%")
            c2.write(f"🛠 Skills: {round(r['skill_score']*100,2)}%")

            st.markdown("**✅ Matched Skills**")
            st.markdown(
                " ".join([f"<span class='good'>✔ {s}</span>" for s in r["matched_skills"]]),
                unsafe_allow_html=True
            )

            st.markdown("**❌ Missing Skills**")
            st.markdown(
                " ".join([f"<span class='bad'>✖ {s}</span>" for s in r["missing_skills"]]),
                unsafe_allow_html=True
            )

            st.divider()

    with tab2:
        df = pd.DataFrame(results)

        st.subheader("📊 Score Distribution")
        st.bar_chart(df.set_index("name")["final_score"])

        st.subheader("📈 Comparison")
        st.line_chart(df.set_index("name")[["semantic_score", "skill_score"]])

        st.download_button(
            "⬇ Download Results",
            df.to_csv(index=False),
            "results.csv"
        )
