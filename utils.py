import pdfplumber
import docx
import re
import nltk
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

model = SentenceTransformer('all-MiniLM-L6-v2')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9+ ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

SKILL_SET = [
    "python","java","sql","machine learning","deep learning",
    "nlp","excel","communication","leadership"
]

def extract_skills(text):
    return {s for s in SKILL_SET if s in text}

def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*(years|yrs)', text.lower())
    return max([int(m[0]) for m in matches]) if matches else 0

def get_embeddings_batch(text_list):
    return model.encode(text_list, batch_size=16, show_progress_bar=False)

def compute_detailed_score(jd_text, resume_text, jd_emb, res_emb):
    semantic = cosine_similarity([jd_emb], [res_emb])[0][0]

    jd_sk = extract_skills(jd_text)
    res_sk = extract_skills(resume_text)

    match = jd_sk & res_sk
    skill_score = len(match)/(len(jd_sk)+1)

    jd_exp = extract_experience(jd_text)
    res_exp = extract_experience(resume_text)
    exp_score = min(res_exp/jd_exp,1) if jd_exp>0 else 0.5

    final = 0.5*semantic + 0.3*skill_score + 0.2*exp_score

    return {
        "final_score": final,
        "semantic_score": semantic,
        "skill_score": skill_score,
        "matched_skills": list(match),
        "missing_skills": list(jd_sk - res_sk)
    }

def generate_explanation(jd, res, score):
    try:
        prompt = f"Explain briefly why this candidate is best match.\nScore:{score['final_score']}"
        r = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role":"user","content":prompt}]
        )
        return r.choices[0].message.content
    except:
        return "Explanation unavailable"
