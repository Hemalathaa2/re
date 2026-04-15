from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from utils import *
from database import init_db, insert_result

init_db()
app = FastAPI()

@app.post("/analyze/")
async def analyze(files: List[UploadFile] = File(...), jd_text: str = Form(...)):

    jd_clean = preprocess(jd_text)

    texts, names = [], []

    for f in files:
        if f.filename.endswith(".pdf"):
            text = extract_text_from_pdf(f.file)
        else:
            text = extract_text_from_docx(f.file)

        clean = preprocess(text)

        if len(clean) > 50:
            texts.append(clean)
            names.append(f.filename)

    jd_emb = get_embeddings_batch([jd_clean])[0]
    res_embs = get_embeddings_batch(texts)

    results = []

    for i in range(len(texts)):
        score = compute_detailed_score(jd_clean, texts[i], jd_emb, res_embs[i])
        result = {"name": names[i], **score}
        insert_result(result)   # ✅ store in DB
        results.append(result)
    results.sort(key=lambda x: x["final_score"], reverse=True)

    if results:
        results[0]["llm_explanation"] = generate_explanation(
            jd_text, texts[0], results[0]
        )

    return {"results": results}
