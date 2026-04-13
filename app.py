import streamlit as st
import pandas as pd
from utils import *

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# -------------------------------
# UI STYLE (CLEAN + PROFESSIONAL)
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
    font-size: 14px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}

h2 { font-size:18px !important; }
h3 { font-size:16px !important; }
h4 { font-size:14px !important; }

div[data-baseweb="input"] input {
    font-size:14px !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("## 🚀 AI Hiring Dashboard")

# -------------------------------
# JD INPUT
# -------------------------------
st.markdown("### 📌 Job Description")

jd_option = st.radio("Choose input method:", ["Paste Text", "Upload File"])
jd_text = ""

if jd_option == "Paste Text":
    jd_text = st.text_area("Paste JD", height=200)
else:
    jd_file = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])

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
# ANALYSIS
# -------------------------------
if st.button("Analyze Candidates"):

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

        if results:
            results[0]["llm_explanation"] = generate_explanation(jd_text, texts[0], results[0])

    st.success("Analysis Complete")

    # -------------------------------
    # METRICS
    # -------------------------------
    top_score = round(results[0]["final_score"]*100,2)
    avg_score = round(sum(r["final_score"] for r in results)/len(results)*100,2)

    c1,c2,c3 = st.columns(3)
    c1.metric("Top Score", f"{top_score}%")
    c2.metric("Avg Score", f"{avg_score}%")
    c3.metric("Candidates", len(results))

    # -------------------------------
    # TOP CANDIDATE
    # -------------------------------
    top = results[0]

    st.subheader("Best Candidate")
    st.markdown(f"""
    <div class="card">
        <b>{top['name']}</b><br>
        Score: {round(top['final_score']*100,2)}%
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(top["final_score"]))

    if "llm_explanation" in top and top["llm_explanation"]:
        st.info(top["llm_explanation"])
    else:
        st.warning("No AI explanation available")

    # -------------------------------
    # SHORTLIST
    # -------------------------------
    st.subheader("Shortlisted Candidates")

    for i, r in enumerate(results[:job_openings]):

        st.markdown(f"""
        <div class="card">
        #{i+1} {r['name']} — {round(r['final_score']*100,2)}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(r["final_score"]))

        # ✅ FIXED SKILLS DISPLAY
        st.write("✅ Matched:", ", ".join(r["matched_skills"]) if r["matched_skills"] else "None")
        st.write("❌ Missing:", ", ".join(r["missing_skills"]) if r["missing_skills"] else "None")

        # ✅ SCORE BREAKDOWN
        st.write(f"📊 Semantic: {round(r['semantic_score']*100,2)}%")
        st.write(f"🧠 Skill: {round(r['skill_score']*100,2)}%")
        st.write(f"📄 Experience: {round(r['experience_score']*100,2)}%")

        st.divider()

    # -------------------------------
    # COMPARISON TABLE
    # -------------------------------
    st.subheader("📋 Comparison")

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
    st.subheader("📊 Score Chart")
    st.bar_chart(df.set_index("name")["final_score"])

    st.download_button("Download CSV", df.to_csv(), "results.csv")
