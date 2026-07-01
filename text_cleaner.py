import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SECTION_HEADINGS = ["Problem", "Action", "Reflection", "Learning"]


@dataclass
class CleanDocument:
    full_text: str
    word_count: int
    sections: Dict[str, str] = field(default_factory=dict)


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse more than two consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(text: str, headings: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Split text into sections keyed by heading.
    Recognises headings that appear alone on a line, with optional numbering
    or trailing colon/dash, e.g. "1. Problem:", "Action -", "Reflection".
    """
    if headings is None:
        headings = SECTION_HEADINGS

    pattern = re.compile(
        r"(?im)^(?:\d+[\.\)]\s*)?("
        + "|".join(re.escape(h) for h in headings)
        + r")\s*[:\-]?\s*$"
    )

    matches = list(pattern.finditer(text))
    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        # normalise to the canonical heading capitalisation
        canonical = next(
            h for h in headings if h.lower() == match.group(1).lower()
        )
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[canonical] = text[start:end].strip()

    return sections


def parse_document(raw: str, headings: Optional[List[str]] = None) -> CleanDocument:
    text = clean_text(raw)
    word_count = len(text.split()) if text.strip() else 0
    sections = extract_sections(text, headings)
    return CleanDocument(full_text=text, word_count=word_count, sections=sections)
