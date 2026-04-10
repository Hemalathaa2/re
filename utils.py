import pdfplumber
import docx
import re
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')


# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# -------------------------------
# Extract text from DOCX
# -------------------------------
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


# -------------------------------
# Clean text
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# -------------------------------
# Convert text → embeddings
# -------------------------------
def get_embedding(text):
    return model.encode([text])


# -------------------------------
# Compute similarity
# -------------------------------
def compute_similarity(jd_text, resume_text):
    jd_embedding = get_embedding(jd_text)
    resume_embedding = get_embedding(resume_text)

    score = cosine_similarity(jd_embedding, resume_embedding)[0][0]
    return score
