import pdfplumber
import docx
import re
import nltk
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# NLTK SETUP
# -------------------------------
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------------
# TEXT EXTRACTION
# -------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9+ ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# SKILL DATABASE (MULTI-DOMAIN)
# -------------------------------
SKILL_CATEGORIES = {
    "programming": [
        "python", "java", "c++", "javascript", "typescript", "go", "rust"
    ],
    "data_science": [
        "machine learning", "deep learning", "nlp",
        "data analysis", "pandas", "numpy", "scikit-learn"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform"
    ],
    "web_dev": [
        "react", "angular", "vue", "node", "flask", "django", "spring"
    ],
    "database": [
        "sql", "mysql", "postgresql", "mongodb", "oracle"
    ],
    "tools": [
        "git", "jira", "excel", "power bi", "tableau"
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving"
    ],
    "non_it": [
        "sales", "marketing", "accounting", "finance",
        "hr", "recruitment", "operations", "supply chain"
    ]
}

# -------------------------------
# SKILL EXTRACTION
# -------------------------------
def extract_skills(text):
    text = text.lower()
    found_skills = set()

    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if skill in text:
                found_skills.add(skill)

    return found_skills

# -------------------------------
# DYNAMIC SKILLS (AUTO DETECT)
# -------------------------------
def extract_dynamic_skills(text):
    words = text.split()
    dynamic_skills = set()

    for w in words:
        if len(w) > 3 and w.isalpha():
            dynamic_skills.add(w)

    return dynamic_skills

def get_all_skills(text):
    return extract_skills(text).union(extract_dynamic_skills(text))

# -------------------------------
# EXPERIENCE EXTRACTION
# -------------------------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())
    if matches:
        return max([int(m[0]) for m in matches])
    return 0

# -------------------------------
# EMBEDDINGS (BATCH)
# -------------------------------
def get_embeddings_batch(text_list):
    return model.encode(text_list, batch_size=16, show_progress_bar=False)

# -------------------------------
# SEMANTIC SKILL MATCH (ADVANCED)
# -------------------------------
def semantic_skill_match(jd_skills, res_skills):
    if not jd_skills or not res_skills:
        return 0

    jd_emb = model.encode(list(jd_skills))
    res_emb = model.encode(list(res_skills))

    similarity_matrix = cosine_similarity(jd_emb, res_emb)
    return similarity_matrix.max(axis=1).mean()

# -------------------------------
# FINAL SCORING (OPTIMIZED)
# -------------------------------
def compute_detailed_score(jd_text, resume_text, jd_emb, res_emb):

    # 1. Semantic similarity
    semantic_score = cosine_similarity([jd_emb], [res_emb])[0][0]

    # 2. Skills
    jd_skills = get_all_skills(jd_text)
    res_skills = get_all_skills(resume_text)

    matched_skills = jd_skills.intersection(res_skills)
    missing_skills = jd_skills - res_skills

    skill_score = len(matched_skills) / (len(jd_skills) + 1)

    # 3. Semantic skill similarity (NEW 🔥)
    semantic_skill_score = semantic_skill_match(jd_skills, res_skills)

    # 4. Experience
    jd_exp = extract_experience(jd_text)
    res_exp = extract_experience(resume_text)

    exp_score = min(res_exp / jd_exp, 1) if jd_exp > 0 else 0.5

    # 5. FINAL SCORE (BALANCED)
    final_score = (
        0.4 * semantic_score +
        0.25 * skill_score +
        0.2 * semantic_skill_score +
        0.15 * exp_score
    )

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "semantic_skill_score": semantic_skill_score,
        "experience_score": exp_score,
        "matched_skills": list(matched_skills),
        "missing_skills": list(missing_skills)
    }
