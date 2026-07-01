from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile

from compliance import check_document
from document_parser import extract_text
from model_loader import MODEL_NAME, get_model, load_model
from schemas import ComplianceReport, ScoreReport
from scoring import score_document
from text_cleaner import parse_document


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Action Learning AI Service", version="2.0.0", lifespan=lifespan)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Compliance (Phase 1) ──────────────────────────────────────────────────────

@app.post("/check", response_model=ComplianceReport)
async def check(
    file: UploadFile = File(...),
    allowed_file_types: Optional[str] = Query(None),
    min_word_count: int = Query(0),
    max_word_count: int = Query(0),
    naming_pattern: Optional[str] = Query(None),
    required_headings: Optional[str] = Query(None),
):
    contents = await file.read()
    return check_document(
        contents=contents,
        filename=file.filename or "unknown",
        allowed_file_types=allowed_file_types,
        min_word_count=min_word_count,
        max_word_count=max_word_count,
        naming_pattern=naming_pattern,
        required_headings=required_headings,
    )


# ── Scoring (Phase 2 skeleton) ────────────────────────────────────────────────

@app.post("/score", response_model=ScoreReport)
async def score(file: UploadFile = File(...)):
    contents = await file.read()
    ext = (file.filename or "").rsplit(".", 1)[-1].upper()
    raw = extract_text(contents, ext)
    doc = parse_document(raw)
    return score_document(doc)


@app.get("/model-info")
async def model_info():
    model = get_model()
    return {
        "model": MODEL_NAME,
        "embeddingDimension": model.get_sentence_embedding_dimension(),
        "status": "loaded",
    }
