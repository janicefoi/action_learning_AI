import difflib
import re
from typing import List, Optional

from document_parser import extract_text
from schemas import CheckResult, ComplianceReport
from text_cleaner import extract_sections

GLOBALLY_ALLOWED = {"PDF", "DOCX"}
_THIN_THRESHOLD = 20  # words — a section with fewer words than this is flagged as thin


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

    missing: List[str] = []
    empty: List[str] = []
    thin: List[str] = []

    # Extract all section word counts in one pass for correct boundaries
    found = [h for h in required if _heading_present(h, text)]
    missing = [h for h in required if h not in found]
    word_counts = _section_word_counts(found, text) if found else {}

    for heading in found:
        wc = word_counts.get(heading, 0)
        if wc == 0:
            empty.append(heading)
        elif wc < _THIN_THRESHOLD:
            thin.append(f"{heading} ({wc} word{'s' if wc != 1 else ''})")

    if not missing and not empty and not thin:
        return _pass(
            "Required Sections",
            f"All required sections found: {', '.join(required)}.",
            None,
        )

    summary_parts: List[str] = []
    if missing:
        summary_parts.append(f"{len(missing)} missing")
    if empty:
        summary_parts.append(f"{len(empty)} empty")
    if thin:
        summary_parts.append(f"{len(thin)} thin")

    detail_parts: List[str] = []
    if missing:
        detail_parts.append(f"Missing: {', '.join(missing)}")
    if empty:
        detail_parts.append(f"Empty (heading found, no content): {', '.join(empty)}")
    if thin:
        detail_parts.append(f"Thin (under {_THIN_THRESHOLD} words): {', '.join(thin)}")

    return _fail(
        "Required Sections",
        f"Section issues — {', '.join(summary_parts)}.",
        " | ".join(detail_parts),
    )


def _find_actual_heading(heading: str, text: str) -> Optional[str]:
    """Return the candidate line from the document that fuzzy-matches the required heading."""
    h = heading.lower()
    for candidate in _candidate_headings(text):
        c = candidate.lower()
        if c == h or c.startswith(h) or h.startswith(c):
            return candidate
        if len(c) >= 2 and difflib.SequenceMatcher(None, h, c).ratio() >= 0.75:
            return candidate
    return None


def _section_word_counts(required: List[str], text: str) -> dict:
    """
    Extract all required sections in a single pass so boundaries between them
    are correct (each section ends where the next one starts).

    Returns {required_heading: word_count}.  Headings not found in the
    document map to 0.
    """
    # Resolve each required heading to its actual form in the document
    actual_for: dict = {}  # required → actual
    for h in required:
        actual_for[h] = _find_actual_heading(h, text) or h

    # One call with all resolved headings — correct section boundaries
    all_sections = extract_sections(text, list(actual_for.values()))

    result = {}
    for req_h, actual_h in actual_for.items():
        content = all_sections.get(actual_h, "")
        result[req_h] = len(content.split()) if content.strip() else 0
    return result


def _heading_present(heading: str, text: str) -> bool:
    """
    Three-pass check (most → least strict):

    1. Exact line match  — heading fills the whole line, optional numbering
       e.g. "3. Abstract:" or "Methodology"
    2. Line-start match  — heading begins a line (catches headings that run
       into content on the same line after PDF/DOCX extraction)
    3. Fuzzy line match  — compares the required heading against every
       short line in the document using prefix containment and
       SequenceMatcher similarity (≥ 0.75).  Handles:
         • capitalisation variants  ("ABSTRACT" → "Abstract")
         • abbreviations            ("Intro"    → "Introduction")
         • trailing words           ("Method"   → "Methodology")
         • minor typos              ("Conclsion"→ "Conclusion")
    """
    escaped = re.escape(heading)
    if re.search(rf"(?im)^[ \t]*(?:\d+[\.\)]\s*)?{escaped}\s*[:\-]?\s*$", text):
        return True
    if re.search(rf"(?im)^[ \t]*(?:\d+[\.\)]\s*)?{escaped}\b", text):
        return True
    return _fuzzy_match(heading, text)


_HEADING_LINE_RE = re.compile(r"^\d+[\.\)]\s*")  # strip leading "1. " / "2) "

def _candidate_headings(text: str) -> List[str]:
    """
    Return every short line that could plausibly be a section heading.
    Strips leading numbers and trailing punctuation before returning.
    """
    candidates = []
    for raw in text.splitlines():
        stripped = raw.strip()
        clean = _HEADING_LINE_RE.sub("", stripped).rstrip(":- \t")
        if 2 <= len(clean) <= 80:
            candidates.append(clean)
    return candidates


def _fuzzy_match(heading: str, text: str) -> bool:
    h = heading.lower()
    for candidate in _candidate_headings(text):
        c = candidate.lower()
        # Prefix in either direction: "Method" ↔ "Methodology", "Intro" ↔ "Introduction"
        if c.startswith(h) or h.startswith(c):
            return True
        # Edit-distance similarity — catches typos and close variants
        if difflib.SequenceMatcher(None, h, c).ratio() >= 0.75:
            return True
    return False


def _pass(label: str, message: str, detail: Optional[str]) -> CheckResult:
    return CheckResult(label=label, passed=True, skipped=False, message=message, detail=detail)


def _fail(label: str, message: str, detail: Optional[str]) -> CheckResult:
    return CheckResult(label=label, passed=False, skipped=False, message=message, detail=detail)


def _skipped(label: str, message: str) -> CheckResult:
    return CheckResult(label=label, passed=False, skipped=True, message=message, detail=None)
