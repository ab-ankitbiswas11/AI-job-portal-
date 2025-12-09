from sentence_transformers import SentenceTransformer
import pickle
from db import jobs_collection, embeddings_collection

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sample_jobs = [
    {
        "title": "Machine Learning Engineer",
        "company": "Google",
        "description": "Build ML pipelines, Python, TensorFlow"
    },
    {
        "title": "Data Scientist",
        "company": "Amazon",
        "description": "Data analysis, ML models, statistics"
    },
    {
        "title": "Frontend Developer",
        "company": "Microsoft",
        "description": "React, JavaScript, UI development"
    }
]

for job in sample_jobs:
    job_id = jobs_collection.insert_one(job).inserted_id

    emb = model.encode(job["description"])
    emb_bytes = pickle.dumps(emb)

    embeddings_collection.insert_one({
        "job_id": str(job_id),
        "embedding": emb_bytes
    })

print("Jobs + Embeddings loaded!")
