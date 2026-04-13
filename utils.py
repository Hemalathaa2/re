import pdfplumber
import docx
import re
import nltk
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# -------------------------------
# NLTK
# -------------------------------
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# -------------------------------
# MODEL
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# GROQ
# -------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
# SKILLS
# -------------------------------
SKILL_CATEGORIES = {
    "programming": ["python", "java", "c++", "javascript"],
    "data_science": ["machine learning", "deep learning", "nlp"],
    "tools": ["excel", "power bi", "tableau"],
    "non_it": ["sales", "marketing", "finance", "hr"]
}

def extract_skills(text):
    found = set()
    for skills in SKILL_CATEGORIES.values():
        for skill in skills:
            if skill in text:
                found.add(skill)
    return found

# -------------------------------
# EXPERIENCE
# -------------------------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())
    return max([int(m[0]) for m in matches]) if matches else 0

# -------------------------------
# EMBEDDINGS
# -------------------------------
def get_embeddings_batch(text_list):
    return model.encode(text_list, batch_size=16, show_progress_bar=False)

# -------------------------------
# SCORING
# -------------------------------
def compute_detailed_score(jd_text, resume_text, jd_emb, res_emb):

    semantic_score = cosine_similarity([jd_emb], [res_emb])[0][0]

    jd_skills = extract_skills(jd_text)
    res_skills = extract_skills(resume_text)

    matched = jd_skills & res_skills
    missing = jd_skills - res_skills

    skill_score = len(matched) / (len(jd_skills) + 1)

    jd_exp = extract_experience(jd_text)
    res_exp = extract_experience(resume_text)
    exp_score = min(res_exp / jd_exp, 1) if jd_exp > 0 else 0.5

    final_score = 0.5 * semantic_score + 0.3 * skill_score + 0.2 * exp_score

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "matched_skills": list(matched),
        "missing_skills": list(missing)
    }

# -------------------------------
# LLM EXPLANATION
# -------------------------------
def generate_explanation(jd_text, resume_text, result):
    try:
        prompt = f"""
You are a recruiter.

Job Description:
{jd_text[:800]}

Candidate Resume:
{resume_text[:800]}

Matched Skills: {result['matched_skills']}
Missing Skills: {result['missing_skills']}
Score: {round(result['final_score']*100,2)}%

Explain why this candidate is ranked #1 in 3 lines.
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except:
        return "LLM explanation unavailable"
