from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from typing import List
from utils import *
from database import init_db, insert_result, save_job, get_job
import uuid

app = FastAPI()

init_db()

@app.get("/")
def home():
    return {"status": "API Running"}

# -------------------------------
# BACKGROUND PROCESS
# -------------------------------
def process_resumes(job_id, files, jd_text):

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

    if not texts:
        save_job(job_id, [])
        return

    jd_emb = get_embeddings_batch([jd_clean])[0]
    res_embs = get_embeddings_batch(texts)

    results = []

    for i in range(len(texts)):
        score = compute_detailed_score(jd_clean, texts[i], jd_emb, res_embs[i])
        result = {"name": names[i], **score}
        results.append(result)
        insert_result(result)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    # LLM only top 2 (reduce time)
    for i in range(min(2, len(results))):
        results[i]["llm_explanation"] = generate_explanation(
            jd_text, texts[i], results[i]
        )

    save_job(job_id, results)

# -------------------------------
# START JOB
# -------------------------------
@app.post("/analyze/")
async def analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    jd_text: str = Form(...)
):

    job_id = str(uuid.uuid4())

    background_tasks.add_task(process_resumes, job_id, files, jd_text)

    return {"job_id": job_id}

# -------------------------------
# GET RESULT
# -------------------------------
@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = get_job(job_id)
    return {"results": result or []}
