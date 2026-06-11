from fastapi import FastAPI
from pydantic import BaseModel

from engine import ask_question

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: QueryRequest):

    return ask_question(req.question)


@app.get("/health")
def health():
    return {"status": "ok"}
