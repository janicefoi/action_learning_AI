"""
Phase 2 scoring skeleton.

Sections are detected, embeddings are computed to verify the model works,
and placeholder scores are returned. Phase 3 will replace the placeholder
logic with cosine-similarity scoring against reference rubric embeddings.
"""
from typing import List

from model_loader import embed
from schemas import ScoreReport, SectionScore
from text_cleaner import SECTION_HEADINGS, CleanDocument


def score_document(doc: CleanDocument, headings: List[str] = None) -> ScoreReport:
    if headings is None:
        headings = SECTION_HEADINGS

    section_scores: List[SectionScore] = []
    texts_to_embed: List[str] = []
    embed_index: List[int] = []   # which section index each text belongs to

    for i, heading in enumerate(headings):
        content = doc.sections.get(heading, "")
        present = bool(content.strip())
        wc = len(content.split()) if content.strip() else 0

        if present:
            texts_to_embed.append(content)
            embed_index.append(i)

        # Phase 2: presence-based placeholder — Phase 3 replaces with rubric similarity
        score = 0.5 if present else 0.0
        feedback = (
            "Section found. Quality scoring will be applied in Phase 3."
            if present
            else "Section not detected in document."
        )

        section_scores.append(
            SectionScore(
                section=heading,
                present=present,
                wordCount=wc,
                score=score,
                feedback=feedback,
            )
        )

    embeddings_computed = False
    if texts_to_embed:
        try:
            embed(texts_to_embed)
            embeddings_computed = True
        except Exception:
            embeddings_computed = False

    present_count = sum(1 for s in section_scores if s.present)
    overall = round(present_count / len(headings), 2) if headings else 0.0

    if overall >= 0.9:
        level = "excellent"
    elif overall >= 0.7:
        level = "good"
    elif overall >= 0.5:
        level = "average"
    else:
        level = "poor"

    return ScoreReport(
        overallScore=overall,
        level=level,
        wordCount=doc.word_count,
        sections=section_scores,
        embeddingsComputed=embeddings_computed,
    )
