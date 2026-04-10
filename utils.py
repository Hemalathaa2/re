import pdfplumber
import docx
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords safely
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------------
# SKILL DICTIONARY (expandable)
# -----------------------------------
SKILL_MAP = {
    "machine learning": ["ml", "machine learning"],
    "deep learning": ["dl", "deep learning"],
    "python": ["python"],
    "sql": ["sql", "structured query language"],
    "nlp": ["nlp", "natural language processing"],
    "data analysis": ["data analysis", "data analytics"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch", "torch"],
    "aws": ["aws", "amazon web services"]
}


# -----------------------------------
# TEXT EXTRACTION
# -----------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


# -----------------------------------
# PREPROCESSING
# -----------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# -----------------------------------
# SKILL EXTRACTION
# -----------------------------------
def extract_skills(text):
    found_skills = set()

    for main_skill, variations in SKILL_MAP.items():
        for v in variations:
            if v in text:
                found_skills.add(main_skill)

    return found_skills


# -----------------------------------
# EMBEDDINGS
# -----------------------------------
def get_embedding(text):
    return model.encode([text])


# -----------------------------------
# HYBRID SCORING
# -----------------------------------
def compute_detailed_score(jd_text, resume_text):

    # Semantic similarity
    jd_emb = get_embedding(jd_text)
    res_emb = get_embedding(resume_text)
    semantic_score = cosine_similarity(jd_emb, res_emb)[0][0]

    # Skill matching
    jd_skills = extract_skills(jd_text)
    res_skills = extract_skills(resume_text)

    matched = jd_skills.intersection(res_skills)
    missing = jd_skills - res_skills

    if len(jd_skills) > 0:
        skill_score = len(matched) / len(jd_skills)
    else:
        skill_score = 0

    # Final score (weighted)
    final_score = (0.7 * semantic_score) + (0.3 * skill_score)

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "matched_skills": matched,
        "missing_skills": missing
    }
