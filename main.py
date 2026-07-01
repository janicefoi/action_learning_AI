from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile

from compliance import check_document
from schemas import ComplianceReport

app = FastAPI(title="Action Learning AI Service", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


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
