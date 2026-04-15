import streamlit as st
import pandas as pd
from utils import *
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import requests

st.set_page_config(page_title="AI Hiring Dashboard", layout="wide")

# API URL
API_URL = "http://localhost:8000/analyze/"

# -------------------------------
# CLEAN PRODUCTION UI
# -------------------------------
st.markdown("""
<style>

/* System adaptive */
html, body {
    color: inherit !important;
    background: inherit !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(150,150,150,0.2);
}

/* Buttons */
.stButton > button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    font-weight: 600;
}

/* Drag Upload */
section[data-testid="stFileUploader"] {
    border: 2px dashed #22c55e !important;
    border-radius: 14px;
    padding: 20px;
}

/* Cards */
.card {
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(150,150,150,0.2);
    margin-bottom: 12px;
}

/* Chips */
.chip {
    display: inline-block;
    padding: 5px 10px;
    margin: 3px;
    border-radius: 20px;
    font-size: 12px;
}
.match { background: rgba(34,197,94,0.2); }
.miss { background: rgba(239,68,68,0.2); }

</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("AI Hiring System")
page = st.sidebar.radio("Navigation", ["Dashboard", "Results"])

# -------------------------------
# DASHBOARD
# -------------------------------
if page == "Dashboard":

    st.title("🚀 AI Hiring Dashboard")

    jd_option = st.radio("Job Description Input", ["Paste", "Upload"])
    jd_text = ""

    if jd_option == "Paste":
        jd_text = st.text_area("Paste JD")
    else:
        jd_file = st.file_uploader("Upload JD", type=["pdf","docx","txt"])
        if jd_file:
            jd_text = extract_text_from_pdf(jd_file) if jd_file.name.endswith(".pdf") else extract_text_from_docx(jd_file)

    st.subheader("📂 Drag & Drop Resumes")
    resume_files = st.file_uploader(
        "Drop resumes here",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

    job_openings = st.number_input("Openings", 1, 50, 1)

    if st.button("Analyze"):

        if not jd_text or not resume_files:
            st.warning("Provide JD and resumes")
            st.stop()

        with st.spinner("Analyzing via API..."):

            try:
                response = requests.post(
                    API_URL,
                    files=[("files", (f.name, f, f.type)) for f in resume_files],
                    data={"jd_text": jd_text}
                )

                if response.status_code != 200:
                    st.error("API Error")
                    st.stop()

                results = response.json()["results"]

            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.stop()

        st.session_state["results"] = results
        st.session_state["job_openings"] = job_openings

        st.success("Analysis complete → Go to Results")

# -------------------------------
# RESULTS
# -------------------------------
if page == "Results":

    if "results" not in st.session_state:
        st.warning("Run analysis first")
        st.stop()

    results = st.session_state["results"]
    job_openings = st.session_state["job_openings"]

    # -------------------------------
    # TOP CARDS
    # -------------------------------
    st.subheader("🏆 Top Candidates")

    for i, r in enumerate(results[:job_openings]):

        st.markdown(f"""
        <div class="card">
        <b>#{i+1} {r['name']}</b><br>
        Score: {r['final_score']*100:.2f}%
        </div>
        """, unsafe_allow_html=True)

        # Skill chips
        chips = ""
        for s in r["matched_skills"]:
            chips += f'<span class="chip match">{s}</span>'
        for s in r["missing_skills"]:
            chips += f'<span class="chip miss">{s}</span>'

        st.markdown(chips, unsafe_allow_html=True)

    # -------------------------------
    # RADAR CHART
    # -------------------------------
    st.subheader("📊 Candidate Radar Comparison")

    selected = st.selectbox("Select Candidate", [r["name"] for r in results])

    r = next(x for x in results if x["name"] == selected)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            r["semantic_score"],
            r["skill_score"],
            r["experience_score"]
        ],
        theta=["Semantic", "Skills", "Experience"],
        fill='toself'
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # SIDE-BY-SIDE AI PANEL
    # -------------------------------
    st.subheader("🧠 AI Explanation Panel")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🥇 Top Candidate")
        st.write(results[0].get("llm_explanation", "No explanation"))

    with col2:
        st.markdown("### 🥈 Second Candidate")
        if len(results) > 1:
            st.write(results[1].get("llm_explanation", "No explanation"))

    # -------------------------------
    # PDF REPORT
    # -------------------------------
    st.subheader("📄 Download Report")

    def generate_pdf(results):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(tmp.name)
        styles = getSampleStyleSheet()

        content = []
        for r in results[:job_openings]:
            content.append(Paragraph(f"{r['name']} - Score: {r['final_score']*100:.2f}%", styles["Normal"]))

        doc.build(content)
        return tmp.name

    if st.button("Generate PDF"):
        pdf_path = generate_pdf(results)

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, "report.pdf")

    # -------------------------------
    # TABLE + CHART
    # -------------------------------
    df = pd.DataFrame(results)

    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("name")["final_score"])
