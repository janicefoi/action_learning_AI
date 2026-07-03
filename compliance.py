import re
from typing import Optional

from document_parser import extract_text
from schemas import CheckResult, ComplianceReport

GLOBALLY_ALLOWED = {"PDF", "DOCX"}


def check_document(
    contents: bytes,
    filename: str,
    allowed_file_types: Optional[str],
    min_word_count: int,
    max_word_count: int,
    naming_pattern: Optional[str],
    required_headings: Optional[str],
) -> ComplianceReport:
    ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else ""
    file_type_result = _check_file_type(ext, allowed_file_types)

    if not file_type_result.passed:
        return ComplianceReport(
            fileName=filename,
            fileSizeBytes=len(contents),
            overallPass=False,
            fileType=file_type_result,
            naming=_skipped("File Naming", "File rejected — naming check skipped."),
            wordCount=_skipped("Word Count", "File rejected — word count check skipped."),
            headings=_skipped("Required Sections", "File rejected — heading check skipped."),
        )

    text = ""
    parse_ok = True
    try:
        text = extract_text(contents, ext)
    except Exception:
        parse_ok = False

    naming_result = _check_naming(filename, naming_pattern)
    word_count_result = (
        _check_word_count(text, min_word_count, max_word_count)
        if parse_ok
        else _skipped("Word Count", "Could not read document content.")
    )
    headings_result = (
        _check_headings(text, required_headings)
        if parse_ok
        else _skipped("Required Sections", "Could not read document content.")
    )

    overall = (
        file_type_result.passed
        and (naming_result.skipped or naming_result.passed)
        and (word_count_result.skipped or word_count_result.passed)
        and (headings_result.skipped or headings_result.passed)
    )

    return ComplianceReport(
        fileName=filename,
        fileSizeBytes=len(contents),
        overallPass=overall,
        fileType=file_type_result,
        naming=naming_result,
        wordCount=word_count_result,
        headings=headings_result,
    )


def _check_file_type(ext: str, allowed_file_types: Optional[str]) -> CheckResult:
    if ext not in GLOBALLY_ALLOWED:
        return _fail(
            "File Type",
            "Only PDF and DOCX files are accepted.",
            f"Submitted: .{ext.lower()}" if ext else "No file extension detected.",
        )
    if allowed_file_types:
        allowed = {
            t.strip().upper().lstrip(".")
            for t in re.split(r"[,;\s]+", allowed_file_types)
            if t.strip()
        }
        if allowed and ext not in allowed:
            return _fail(
                "File Type",
                f"This assignment only accepts: {allowed_file_types}.",
                f"Submitted: .{ext.lower()}",
            )
    return _pass("File Type", f"Accepted: {ext}", None)


def _check_naming(filename: str, naming_pattern: Optional[str]) -> CheckResult:
    if not naming_pattern:
        return _skipped("File Naming", "No naming convention set for this assignment.")
    if re.fullmatch(naming_pattern, filename):
        return _pass("File Naming", "File name matches the required pattern.", None)
    return _fail(
        "File Naming",
        "File name does not match the required pattern.",
        f"Expected pattern: {naming_pattern}",
    )


def _check_word_count(text: str, min_wc: int, max_wc: int) -> CheckResult:
    if min_wc == 0 and max_wc == 0:
        return _skipped("Word Count", "No word count limit set for this assignment.")
    count = len(text.split()) if text.strip() else 0
    count_str = f"{count:,}"
    if min_wc > 0 and count < min_wc:
        return _fail(
            "Word Count",
            f"Document is too short: {count_str} words (minimum: {min_wc:,}).",
            f"Minimum required: {min_wc:,} words",
        )
    if max_wc > 0 and count > max_wc:
        return _fail(
            "Word Count",
            f"Document is too long: {count_str} words (maximum: {max_wc:,}).",
            f"Maximum allowed: {max_wc:,} words",
        )
    if min_wc > 0 and max_wc > 0:
        range_str = f"within {min_wc:,} to {max_wc:,} word limit"
    else:
        range_str = "word count accepted"
    return _pass("Word Count", f"{count_str} words — {range_str}.", None)


def _check_headings(text: str, required_headings: Optional[str]) -> CheckResult:
    if not required_headings:
        return _skipped("Required Sections", "No required sections set for this assignment.")
    required = [h.strip() for h in required_headings.split(",") if h.strip()]
    if not required:
        return _skipped("Required Sections", "No required sections set for this assignment.")
    lower = text.lower()
    missing = [h for h in required if h.lower() not in lower]
    required_str = ", ".join(required)
    if missing:
        return _fail(
            "Required Sections",
            f"Missing required section(s): {', '.join(missing)}.",
            f"Required: {required_str}",
        )
    return _pass("Required Sections", f"All required sections found: {required_str}.", None)


def _pass(label: str, message: str, detail: Optional[str]) -> CheckResult:
    return CheckResult(label=label, passed=True, skipped=False, message=message, detail=detail)


def _fail(label: str, message: str, detail: Optional[str]) -> CheckResult:
    return CheckResult(label=label, passed=False, skipped=False, message=message, detail=detail)


def _skipped(label: str, message: str) -> CheckResult:
    return CheckResult(label=label, passed=False, skipped=True, message=message, detail=None)
