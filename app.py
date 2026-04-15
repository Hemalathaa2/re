import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# -------------------------------
# MODERN UI STYLE (TEMPLATE MATCH)
# -------------------------------
st.markdown("""
<style>

/* Global */
body {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #8b949e;
    margin-bottom: 25px;
}

/* Section card */
.section {
    background-color: #161b22;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #30363d;
    margin-bottom: 20px;
}

/* Metrics */
.metric-box {
    background: #161b22;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #30363d;
}

/* Buttons */
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #2ea043;
}

/* Inputs */
textarea, input {
    background-color: #0d1117 !important;
    color: white !important;
}

/* Progress */
div[data-testid="stProgress"] > div > div {
    background-color: #2ea043;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">AI Hiring Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Resume Screening using AI</div>', unsafe_allow_html=True)

# -------------------------------
# JD INPUT
# -------------------------------
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📌 Job Description")

    jd_option = st.radio("Choose input method:", ["Paste Text", "Upload File"])
    jd_text = ""

    if jd_option == "Paste Text":
        jd_text = st.text_area("Paste Job Description", height=120)
    else:
        jd_file = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])

        if jd_file:
            if jd_file.name.endswith(".pdf"):
                jd_text = extract_text_from_pdf(jd_file)
            elif jd_file.name.endswith(".docx"):
                jd_text = extract_text_from_docx(jd_file)
            else:
                jd_text = jd_file.read().decode("utf-8")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RESUME + JOB OPENINGS
# -------------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📂 Upload Resumes")

    resume_files = st.file_uploader(
        "Upload resumes",
        type=["pdf","docx"],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 👥 Openings")

    job_openings = st.number_input(
        "Number of openings",
        min_value=1,
        max_value=50,
        value=1
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# ANALYSIS BUTTON
# -------------------------------
analyze = st.button("🚀 Analyze Candidates")

# -------------------------------
# ANALYSIS LOGIC (UNCHANGED)
# -------------------------------
if analyze:

    if not jd_text or not resume_files:
        st.warning("Provide JD and resumes")
        st.stop()

    with st.spinner("Processing..."):

        jd_clean = preprocess(jd_text)

        texts, names = [], []

        for f in resume_files:
            try:
                text = extract_text_from_pdf(f) if f.name.endswith(".pdf") else extract_text_from_docx(f)
                clean = preprocess(text)

                if len(clean) > 50:
                    texts.append(clean)
                    names.append(f.name)

            except:
                st.warning(f"Error reading {f.name}")

        jd_emb = get_embeddings_batch([jd_clean])[0]
        res_embs = get_embeddings_batch(texts)

        results = []

        for i in range(len(texts)):
            score = compute_detailed_score(jd_clean, texts[i], jd_emb, res_embs[i])
            results.append({"name": names[i], **score})

        results.sort(key=lambda x: x["final_score"], reverse=True)

        for i in range(len(results)):
            results[i]["llm_explanation"] = generate_explanation(
                jd_text,
                texts[i],
                results[i]
            )

    st.success("Analysis Complete")

    # -------------------------------
    # METRICS
    # -------------------------------
    c1, c2, c3 = st.columns(3)

    top_score = round(results[0]["final_score"]*100,2)
    avg_score = round(sum(r["final_score"] for r in results)/len(results)*100,2)

    c1.metric("Top Score", f"{top_score}%")
    c2.metric("Avg Score", f"{avg_score}%")
    c3.metric("Candidates", len(results))

    # -------------------------------
    # BEST CANDIDATE
    # -------------------------------
    st.markdown("### 🏆 Best Candidate")
    top = results[0]

    st.info(f"{top['name']} | Score: {round(top['final_score']*100,2)}%")
    st.progress(float(top["final_score"]))

    if top.get("llm_explanation"):
        st.success(top["llm_explanation"])

    # -------------------------------
    # SHORTLIST
    # -------------------------------
    st.markdown("### 🎯 Shortlisted Candidates")

    for i, r in enumerate(results[:job_openings]):

        if r["final_score"] > 0.7:
            verdict = "🟢 Strong Match"
        elif r["final_score"] > 0.4:
            verdict = "🟡 Moderate Match"
        else:
            verdict = "🔴 Low Match"

        with st.container():
            st.markdown(f"**#{i+1} {r['name']}**")
            st.progress(float(r["final_score"]))

            st.write(f"Score: {r['final_score']*100:.2f}%")
            st.write(f"Matched Skills: {', '.join(r['matched_skills'])}")
            st.write(f"Missing Skills: {', '.join(r['missing_skills'])}")
            st.write(f"Verdict: {verdict}")

            with st.expander("AI Explanation"):
                st.write(r.get("llm_explanation", "No explanation"))

            st.divider()

    # -------------------------------
    # TABLE
    # -------------------------------
    st.markdown("### 📋 Comparison Table")

    df = pd.DataFrame(results)

    df_display = df[[
        "name","final_score","semantic_score","skill_score","experience_score"
    ]]

    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(lambda x: round(x*100,2))

    st.dataframe(df_display, use_container_width=True)

    # -------------------------------
    # CHART
    # -------------------------------
    st.markdown("### 📊 Score Chart")
    st.bar_chart(df.set_index("name")["final_score"])

    st.download_button("Download CSV", df.to_csv(), "results.csv")
