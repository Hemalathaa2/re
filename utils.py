import pdfplumber
import docx
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

model = SentenceTransformer('all-MiniLM-L6-v2')

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
# AUTO SKILL EXTRACTION
# -------------------------------
def extract_keywords(text):
    words = text.split()
    keywords = set()

    for w in words:
        if len(w) > 2 and w not in stop_words:
            keywords.add(w)

    return keywords


# -------------------------------
# EXPERIENCE EXTRACTION
# -------------------------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())
    if matches:
        return max([int(m[0]) for m in matches])
    return 0


# -------------------------------
# EMBEDDINGS
# -------------------------------
def get_embedding(text):
    return model.encode([text])


# -------------------------------
# ADVANCED SCORING
# -------------------------------
def compute_advanced_score(jd_text, resume_text):

    jd_emb = get_embedding(jd_text)
    res_emb = get_embedding(resume_text)

    semantic_score = cosine_similarity(jd_emb, res_emb)[0][0]

    # Keywords
    jd_keywords = extract_keywords(jd_text)
    res_keywords = extract_keywords(resume_text)

    matched_keywords = jd_keywords.intersection(res_keywords)

    if len(jd_keywords) > 0:
        keyword_score = len(matched_keywords) / len(jd_keywords)
    else:
        keyword_score = 0

    # Experience
    jd_exp = extract_experience(jd_text)
    res_exp = extract_experience(resume_text)

    if jd_exp > 0:
        exp_score = min(res_exp / jd_exp, 1)
    else:
        exp_score = 0.5

    # Final weighted score
    final_score = (
        0.4 * semantic_score +
        0.3 * keyword_score +
        0.2 * exp_score +
        0.1 * (len(matched_keywords) / (len(res_keywords)+1))
    )

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "experience_score": exp_score,
        "matched_keywords": list(matched_keywords)[:10],
        "jd_experience": jd_exp,
        "resume_experience": res_exp
    }
