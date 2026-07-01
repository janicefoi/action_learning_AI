from pydantic import BaseModel
from typing import Optional


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
