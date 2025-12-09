# ml_service/app.py
#enviroment D:\job\ml_service\.venv\Scripts\Activate.ps1
#  uvicorn app:app --reload --host 0.0.0.0 --port 8000
'''
cd D:\job\ml_service
.\.venv\Scripts\activate


uvicorn app:app --reload --host 127.0.0.1 --port 8000
uvicorn ml_service.app:app --reload

'''

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import io
import os
import re

# Text extraction
import pdfplumber
import docx

# Embeddings + similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI(title="AI Job Recommendation Service")

# ----- Config -----
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_DEFAULT = 10
SIMILARITY_THRESHOLD = 0.45  # tune this (0-1)

# Load model once at startup
print("Loading embedding model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Job Recommendation Model Loaded")

# ----- Simple skill-list (extendable) -----
# Replace/extend this with a larger curated list or use an external taxonomy.
SKILL_LIST = [
    "python","java","c++","c","c#","javascript","react","node","express",
    "django","flask","sql","postgresql","mongodb","html","css","aws","azure",
    "docker","kubernetes","ml","machine learning","deep learning","tensorflow",
    "pytorch","nlp","data analysis","pandas","numpy","git","rest","graphql",
    "bash","linux","computer vision","cv","spark","hadoop","redis","reactjs"
]

# This will simulate jobs from your database
JOB_DATABASE = [
    {
        "id": 1,
        "title": "Python Backend Developer",
        "description": "Looking for Python, FastAPI, SQL, and REST API experience."
    },
    {
        "id": 2,
        "title": "Machine Learning Engineer",
        "description": "Skills needed: Python, ML, Deep Learning, TensorFlow."
    },
    {
        "id": 3,
        "title": "Full Stack Developer",
        "description": "React, Node.js, MongoDB, JavaScript, Express."
    },
]

# Normalize skill list for faster matching
SKILL_PATTERNS = [(skill, re.compile(r'\b' + re.escape(skill) + r'\b', re.I)) for skill in SKILL_LIST]


# ----- Pydantic Models -----
class ParseResponse(BaseModel):
    resume_text: str
    skills: List[str]


class JobItem(BaseModel):
    jobId: str
    title: Optional[str] = None
    description: str


class RecommendRequest(BaseModel):
    resume_text: Optional[str] = None
    jobs: List[JobItem]
    top_k: Optional[int] = TOP_K_DEFAULT


class RecommendItem(BaseModel):
    jobId: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem]


# ----- Utilities -----
def extract_text_from_pdf_bytes(b: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def extract_text_from_docx_bytes(b: bytes) -> str:
    f = io.BytesIO(b)
    doc = docx.Document(f)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def extract_text_from_bytes(b: bytes, filename: str) -> str:
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(b)
    if fname.endswith(".docx") or fname.endswith(".doc"):
        try:
            return extract_text_from_docx_bytes(b)
        except Exception:
            # fallback: decode raw bytes
            try:
                return b.decode("utf-8", errors="ignore")
            except:
                return ""
    # fallback to raw decode
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""


def extract_skills_from_text(text: str) -> List[str]:
    found = set()
    for skill, pat in SKILL_PATTERNS:
        if pat.search(text):
            found.add(skill.lower())
    # return sorted for consistency
    return sorted(found)


def compute_embeddings(texts: List[str]):
    # returns numpy array (n, d)
    if len(texts) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def recommend_jobs(user_skills: list[str]):
    # Convert user skills into a single text string
    user_text = " ".join(user_skills)

    # Compute embedding for user
    user_vec = embedding_model.encode(user_text)

    results = []

    # Compare user with every job
    for job in JOB_DATABASE:
        job_vec = embedding_model.encode(job["description"])
        score = cosine_similarity(user_vec, job_vec)

        results.append({
            "job_id": job["id"],
            "title": job["title"],
            "match_score": float(score)
        })

    # Sort by score DESC
    results.sort(key=lambda x: x["match_score"], reverse=True)

    return results

class RecommendRequest(BaseModel):
    skills: list[str]

class RecommendResponse(BaseModel):
    recommendations: list[dict]

@app.post("/recommend", response_model=RecommendResponse)
def recommend_api(payload: RecommendRequest):
    recs = recommend_jobs(payload.skills)
    return {"recommendations": recs}


# ----- Endpoints -----
@app.post("/parse", response_model=ParseResponse)
async def parse_resume(file: UploadFile = File(...)):
    """
    Upload a resume file (PDF/DOCX/TXT). Returns extracted text and detected skills.
    """
    contents = await file.read()
    text = extract_text_from_bytes(contents, file.filename)
    skills = extract_skills_from_text(text)
    return {"resume_text": text, "skills": skills}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Provide resume_text and a list of job items (jobId, description).
    Returns ranked list of jobId + similarity score (0-1).
    """
    if not req.resume_text:
        return {"recommendations": []}

    # Prepare texts: resume + job descriptions
    resume_text = req.resume_text
    jobs = req.jobs
    top_k = req.top_k or TOP_K_DEFAULT

    job_texts = [j.description for j in jobs]
    # Compute embeddings (batch): resume embedding and job embeddings
    # For efficiency, compute resume once
    resume_emb = compute_embeddings([resume_text])[0]  # shape (d,)
    job_embs = compute_embeddings(job_texts)           # shape (n_jobs, d)

    # Cosine similarity (fast)
    # util.cos_sim returns torch tensor; we use numpy to compute directly
    # but sentence-transformers has util.cos_sim - use that for stability
    scores = util.cos_sim(resume_emb, job_embs).cpu().numpy()[0]  # shape (n_jobs,)

    # pair and sort
    results = []
    for j, score in zip(jobs, scores):
        results.append({"jobId": j.jobId, "score": float(score)})

    # Filter by threshold and sort descending
    results = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]
    results.sort(key=lambda x: x["score"], reverse=True)

    # Take top_k
    recommendations = results[:top_k]

    return {"recommendations": recommendations}


# Lightweight healthcheck
@app.get("/")
def home():
    return {"message": "AI Job Portal ML Service is running"}


@app.get("/health")
async def health():
    return {"status": "ok"}

from ml.recommender import recommend_jobs

class RecommendRequest(BaseModel):
    skills: list[str]

class RecommendResponse(BaseModel):
    recommendations: list[dict]

@app.post("/recommend", response_model=RecommendResponse)
async def recommend_jobs_api(req: RecommendRequest):
    recs = recommend_jobs(req.skills)
    return {"recommendations": recs}


# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


