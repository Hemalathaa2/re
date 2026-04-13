from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    preprocess,
    compute_detailed_score,
    get_embeddings_batch,
    generate_explanation
)

app = FastAPI()

@app.post("/analyze/")
async def analyze_resumes(
    files: List[UploadFile] = File(...),
    jd_text: str = Form(...)
):

    jd_clean = preprocess(jd_text)

    resume_texts = []
    resume_names = []

    for file in files:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file.file)
        else:
            text = extract_text_from_docx(file.file)

        clean = preprocess(text)

        if len(clean) > 50:
            resume_texts.append(clean)
            resume_names.append(file.filename)

    jd_emb = get_embeddings_batch([jd_clean])[0]
    res_embs = get_embeddings_batch(resume_texts)

    results = []

    for i in range(len(resume_texts)):
        score = compute_detailed_score(
            jd_clean, resume_texts[i], jd_emb, res_embs[i]
        )

        results.append({
            "name": resume_names[i],
            **score
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)

    # LLM for top candidate
    if results:
        explanation = generate_explanation(
            jd_text,
            resume_texts[0],
            results[0]
        )
        results[0]["llm_explanation"] = explanation

    return {"results": results}
