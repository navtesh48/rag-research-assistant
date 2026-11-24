import pathlib, shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag.db import init
from rag.ingest import ingest_pdf
from rag.answer import ask

app = FastAPI(title='RAG Assistant')
conn = init()

class Q(BaseModel):
    question: str

@app.post('/ask')
def ask_ep(q: Q): return ask(conn, q.question)

@app.post('/ingest')
def ingest_ep(file: UploadFile = File(...)):
    p = pathlib.Path('data/pdfs') / pathlib.Path(file.filename).name
    with p.open('wb') as f: shutil.copyfileobj(file.file, f)
    return {'chunks': ingest_pdf(conn, p)}
