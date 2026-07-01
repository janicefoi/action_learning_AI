from typing import List, Optional

from pydantic import BaseModel


class CheckResult(BaseModel):
    label: str
    passed: bool
    skipped: bool
    message: str
    detail: Optional[str] = None


class ComplianceReport(BaseModel):
    uploadId: Optional[int] = None
    fileName: str
    fileSizeBytes: int
    overallPass: bool
    fileType: CheckResult
    naming: CheckResult
    wordCount: CheckResult
    headings: CheckResult


# ── Phase 2: scoring schemas ──────────────────────────────────────────────────

class SectionScore(BaseModel):
    section: str
    present: bool
    wordCount: int
    score: float        # 0.0 – 1.0 (placeholder in Phase 2, rubric-based in Phase 3)
    feedback: str


class ScoreReport(BaseModel):
    overallScore: float
    level: str          # "excellent" | "good" | "average" | "poor"
    wordCount: int
    sections: List[SectionScore]
    embeddingsComputed: bool
