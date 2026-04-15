from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from utils import *

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API Running"}

@app.post("/analyze/")
async def analyze(files: List[UploadFile] = File(...), jd_text: str = Form(...)):

    jd_clean = preprocess(jd_text)

    texts, names = [], []

    for f in files:
        try:
            if f.filename.endswith(".pdf"):
                text = extract_text_from_pdf(f.file)
            else:
                text = extract_text_from_docx(f.file)

            clean = preprocess(text)

            if len(clean) > 50:
                texts.append(clean)
                names.append(f.filename)
        except:
            continue

    jd_emb = get_embeddings_batch([jd_clean])[0]
    res_embs = get_embeddings_batch(texts)

    results = []

    for i in range(len(texts)):
        score = compute_detailed_score(jd_clean, texts[i], jd_emb, res_embs[i])
        results.append({"name": names[i], **score})

    results.sort(key=lambda x: x["final_score"], reverse=True)

    # AI explanation (top 3 only for speed)
    for i in range(min(3, len(results))):
        results[i]["llm_explanation"] = generate_explanation(
            jd_text, texts[i], results[i]
        )

    return {"results": results}


