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


# ── Phase 3: scoring schemas ──────────────────────────────────────────────────

class CriterionScore(BaseModel):
    label: str
    score: float            # 0.0 – 1.0
    feedback: str
    confidence: str         # "low" | "medium" | "high"
    requiresReview: bool


class ScoreReport(BaseModel):
    overallScore: float
    level: str              # "excellent" | "good" | "average" | "poor"
    wordCount: int
    embeddingsComputed: bool
    requiresHumanReview: bool
    criteria: List[CriterionScore]
