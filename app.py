import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# -------------------------------
# PREMIUM ADAPTIVE UI
# -------------------------------
st.markdown("""
<style>

/* Global */
body {
    font-family: 'Inter', sans-serif;
}

/* Remove default top spacing */
.block-container {
    padding-top: 1.5rem;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 4px;
    color: inherit;
}

.subtitle {
    font-size: 16px;
    opacity: 0.7;
    margin-bottom: 25px;
}

/* Clean spacing instead of boxes */
.section {
    margin-bottom: 25px;
}

/* Buttons (Premium Green) */
.stButton > button {
    background-color: #22c55e;
    color: black;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #16a34a;
}

/* JD Upload area (same vibe as resume) */
section[data-testid="stFileUploader"] {
    border: 2px dashed #22c55e !important;
    border-radius: 12px;
    padding: 15px;
}

/* Input fields AUTO adapt */
textarea, input {
    color: inherit !important;
    background-color: transparent !important;
    border-radius: 10px !important;
}

/* Remove ugly black focus box */
input:focus, textarea:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background-color: #22c55e;
}

/* Divider */
hr {
    opacity: 0.2;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">🚀 AI Hiring Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Resume Screening using AI</div>', unsafe_allow_html=True)

# -------------------------------
# JD INPUT
# -------------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("### 📌 Job Description")

jd_option = st.radio("Choose input method:", ["Paste Text", "Upload File"])
jd_text = ""

if jd_option == "Paste Text":
    jd_text = st.text_area(
        "Paste Job Description",
        height=120,
        placeholder="Paste the job description here..."
    )
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
# RESUME + OPENINGS (UNCHANGED)
# -------------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### 📂 Upload Resumes")

    resume_files = st.file_uploader(
        "Upload resumes",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

with col2:
    st.markdown("### 👥 Openings")

    job_openings = st.number_input(
        "Number of openings",
        min_value=1,
        max_value=50,
        value=1
    )

# -------------------------------
# ANALYSIS BUTTON
# -------------------------------
if st.button("🚀 Analyze Candidates"):

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

    # METRICS
    c1,c2,c3 = st.columns(3)

    top_score = round(results[0]["final_score"]*100,2)
    avg_score = round(sum(r["final_score"] for r in results)/len(results)*100,2)

    c1.metric("Top Score", f"{top_score}%")
    c2.metric("Avg Score", f"{avg_score}%")
    c3.metric("Candidates", len(results))

    # BEST CANDIDATE
    st.subheader("🏆 Best Candidate")
    top = results[0]

    st.write(f"**{top['name']}** — {round(top['final_score']*100,2)}%")
    st.progress(float(top["final_score"]))

    if top.get("llm_explanation"):
        st.info(top["llm_explanation"])

    # SHORTLIST
    st.subheader("🎯 Shortlisted Candidates")

    for i, r in enumerate(results[:job_openings]):

        if r["final_score"] > 0.7:
            verdict = "🟢 Strong Match"
        elif r["final_score"] > 0.4:
            verdict = "🟡 Moderate Match"
        else:
            verdict = "🔴 Low Match"

        st.write(f"### #{i+1} {r['name']}")
        st.progress(float(r["final_score"]))

        st.write(f"Score: {r['final_score']*100:.2f}%")
        st.write(f"Matched Skills: {', '.join(r['matched_skills']) or 'None'}")
        st.write(f"Missing Skills: {', '.join(r['missing_skills']) or 'None'}")
        st.write(f"Verdict: {verdict}")

        with st.expander("🧠 AI Explanation"):
            st.write(r.get("llm_explanation", "No explanation"))

        st.divider()

    # TABLE
    st.subheader("📋 Comparison")

    df = pd.DataFrame(results)

    df_display = df[[
        "name","final_score","semantic_score","skill_score","experience_score"
    ]]

    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(lambda x: round(x*100,2))

    st.dataframe(df_display, use_container_width=True)

    # CHART
    st.subheader("📊 Score Chart")
    st.bar_chart(df.set_index("name")["final_score"])

    st.download_button("Download CSV", df.to_csv(), "results.csv")
