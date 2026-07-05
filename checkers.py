# Compliance logic lives in compliance.py — this file is unused.
# Kept as a stub to avoid import errors if referenced elsewhere.
import re
from typing import Optional


def check_file_type(filename: str, allowed_file_types: Optional[str]) -> dict:
    if not allowed_file_types or not allowed_file_types.strip():
        return {
            "label": "File Type",
            "passed": True,
            "skipped": True,
            "message": "No file type restriction configured",
            "detail": None,
        }

    ext = ("." + filename.rsplit(".", 1)[-1]).lower() if "." in filename else ""
    allowed = [t.strip().lower() for t in allowed_file_types.split(",") if t.strip()]

    if ext in allowed:
        return {
            "label": "File Type",
            "passed": True,
            "skipped": False,
            "message": f"File type '{ext}' is accepted",
            "detail": None,
        }
    return {
        "label": "File Type",
        "passed": False,
        "skipped": False,
        "message": f"File type '{ext}' is not allowed for this assignment",
        "detail": f"Allowed: {allowed_file_types}",
    }


def check_naming(filename: str, naming_pattern: Optional[str]) -> dict:
    if not naming_pattern or not naming_pattern.strip():
        return {
            "label": "File Naming",
            "passed": True,
            "skipped": True,
            "message": "No naming pattern required",
            "detail": None,
        }

    try:
        if re.search(naming_pattern, filename, re.IGNORECASE):
            return {
                "label": "File Naming",
                "passed": True,
                "skipped": False,
                "message": "Filename matches the required pattern",
                "detail": None,
            }
        return {
            "label": "File Naming",
            "passed": False,
            "skipped": False,
            "message": "Filename does not match the required naming pattern",
            "detail": f"Pattern: {naming_pattern}",
        }
    except re.error:
        return {
            "label": "File Naming",
            "passed": False,
            "skipped": False,
            "message": "Invalid naming pattern — contact your lecturer",
            "detail": f"Pattern: {naming_pattern}",
        }


def check_word_count(
    text: str,
    min_count: Optional[int],
    max_count: Optional[int],
    extractable: bool,
) -> dict:
    if not min_count and not max_count:
        return {
            "label": "Word Count",
            "passed": True,
            "skipped": True,
            "message": "No word count restriction",
            "detail": None,
        }

    if not extractable:
        return {
            "label": "Word Count",
            "passed": True,
            "skipped": True,
            "message": "Word count check skipped — file format does not support text extraction",
            "detail": None,
        }

    count = len(text.split()) if text and text.strip() else 0

    if min_count and count < min_count:
        return {
            "label": "Word Count",
            "passed": False,
            "skipped": False,
            "message": f"Document is too short ({count} words; minimum is {min_count})",
            "detail": f"Word count: {count}",
        }
    if max_count and count > max_count:
        return {
            "label": "Word Count",
            "passed": False,
            "skipped": False,
            "message": f"Document is too long ({count} words; maximum is {max_count})",
            "detail": f"Word count: {count}",
        }

    range_str = ""
    if min_count and max_count:
        range_str = f" ({min_count}–{max_count} required)"
    elif min_count:
        range_str = f" (minimum {min_count})"
    elif max_count:
        range_str = f" (maximum {max_count})"

    return {
        "label": "Word Count",
        "passed": True,
        "skipped": False,
        "message": f"Word count {count} is within the allowed range{range_str}",
        "detail": f"Word count: {count}",
    }


def check_headings(
    text: str,
    required_headings: Optional[str],
    extractable: bool,
) -> dict:
    if not required_headings or not required_headings.strip():
        return {
            "label": "Required Headings",
            "passed": True,
            "skipped": True,
            "message": "No required headings",
            "detail": None,
        }

    if not extractable:
        return {
            "label": "Required Headings",
            "passed": True,
            "skipped": True,
            "message": "Heading check skipped — file format does not support text extraction",
            "detail": None,
        }

    headings = [h.strip() for h in required_headings.split(",") if h.strip()]
    text_lower = (text or "").lower()
    missing = [h for h in headings if h.lower() not in text_lower]

    if not missing:
        return {
            "label": "Required Headings",
            "passed": True,
            "skipped": False,
            "message": f"All {len(headings)} required heading(s) found",
            "detail": None,
        }
    return {
        "label": "Required Headings",
        "passed": False,
        "skipped": False,
        "message": f"{len(missing)} required heading(s) not found in the document",
        "detail": "Missing: " + ", ".join(missing),
    }
