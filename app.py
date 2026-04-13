import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>

/* Global */
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
    font-size: 14px;
}

/* Header */
.main-title {
    font-size: 32px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}

/* Smaller headings */
h2 { font-size: 18px !important; }
h3 { font-size: 16px !important; }
h4 { font-size: 14px !important; }

/* Inputs */
div[data-baseweb="input"] input {
    font-size:14px !important;
    padding:8px !important;
}

/* Buttons */
.stButton > button {
    font-size:14px;
    padding:8px 16px;
}

/* Metrics */
[data-testid="metric-container"] {
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# JD INPUT
# -------------------------------
st.markdown("### 📌 Job Description ")

jd_option = st.radio("Choose input method:", ["Paste Text", "Upload File"])

jd_text = ""

if jd_option == "Paste Text":
    jd_text = st.text_area("Paste JD", height=200)
else:
    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx", "txt"]
    )

    if jd_file:
        if jd_file.name.endswith(".pdf"):
            jd_text = extract_text_from_pdf(jd_file)
        elif jd_file.name.endswith(".docx"):
            jd_text = extract_text_from_docx(jd_file)
        else:
            jd_text = jd_file.read().decode("utf-8")

# -------------------------------
# RESUME UPLOAD
# -------------------------------
st.markdown("### 📂 Upload Resumes ")

resume_files = st.file_uploader(
    "Upload resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# -------------------------------
# JOB OPENINGS
# -------------------------------
st.markdown("### 👥 Enter Number of Job Openings")

job_openings = st.number_input(
    "Number of openings",
    min_value=1,
    max_value=50,
    value=5
)

# -------------------------------
# ANALYSIS
# -------------------------------
if st.button("Analyze Candidates"):

    if not jd_text or not resume_files:
        st.warning("Provide JD and resumes")
        st.stop()

    with st.spinner("🤖 Processing resumes..."):

        jd_clean = preprocess(jd_text)

        texts, names = [], []

        for f in resume_files:
            try:
                if f.name.endswith(".pdf"):
                    text = extract_text_from_pdf(f)
                else:
                    text = extract_text_from_docx(f)

                clean = preprocess(text)

                if len(clean) > 50:
                    texts.append(clean)
                    names.append(f.name)

            except:
                st.warning(f"⚠️ Error reading {f.name}")

        # embeddings
        jd_emb = get_embeddings_batch([jd_clean])[0]
        res_embs = get_embeddings_batch(texts)

        results = []

        for i in range(len(texts)):
            score = compute_detailed_score(
                jd_clean, texts[i], jd_emb, res_embs[i]
            )

            results.append({
                "name": names[i],
                **score
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)

        # LLM explanation
        if results:
            results[0]["llm_explanation"] = generate_explanation(
                jd_text, texts[0], results[0]
            )

    st.success("✅ Analysis Complete")

    # -------------------------------
    # METRICS
    # -------------------------------
    top_score = round(results[0]["final_score"]*100,2)
    avg_score = round(sum(r["final_score"] for r in results)/len(results)*100,2)

    c1,c2,c3 = st.columns(3)
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
        <h3>{round(top['final_score']*100,2)}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    top_score_val = float(top["final_score"])
    top_score_val = max(0.0, min(top_score_val, 1.0))
    st.progress(top_score_val)

    if "llm_explanation" in top:
        st.info(top["llm_explanation"])

    # -------------------------------
    # SHORTLIST
    # -------------------------------
    st.subheader("🎯 Shortlisted Candidates")

    for i, r in enumerate(results[:job_openings]):

        st.markdown(f"""
        <div class="card">
            <h4>#{i+1} {r['name']}</h4>
            <p>{round(r['final_score']*100,2)}%</p>
        </div>
        """, unsafe_allow_html=True)

        score_val = float(r["final_score"])
        score_val = max(0.0, min(score_val, 1.0))
        st.progress(score_val)

        st.write("✅", r["matched_skills"])
        st.write("❌", r["missing_skills"])

        st.divider()

    # -------------------------------
    # INSIGHTS
    # -------------------------------
    df = pd.DataFrame(results)

    st.subheader("📊 Insights")
    st.bar_chart(df.set_index("name")["final_score"])

    st.download_button("⬇ Download CSV", df.to_csv(), "results.csv")
