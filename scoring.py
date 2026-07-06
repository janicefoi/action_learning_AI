from typing import List

from criteria_scorer import (
    score_analytical_depth,
    score_forward_orientation,
    score_logical_coherence,
    score_outcome_linkage,
)
from schemas import CriterionScore, ScoreReport
from text_cleaner import CleanDocument


def score_document(doc: CleanDocument, headings: List[str] = None) -> ScoreReport:
    criteria: List[CriterionScore] = []
    embeddings_computed = False

    try:
        criteria = [
            score_analytical_depth(doc),
            score_outcome_linkage(doc),
            score_logical_coherence(doc),
            score_forward_orientation(doc),
        ]
        embeddings_computed = True
    except Exception:
        embeddings_computed = False

    if criteria:
        overall = round(sum(c.score for c in criteria) / len(criteria), 3)
    else:
        overall = 0.0

    if overall >= 0.70:
        level = "excellent"
    elif overall >= 0.55:
        level = "good"
    elif overall >= 0.40:
        level = "average"
    else:
        level = "poor"

    requires_review = any(c.requiresReview for c in criteria)

    return ScoreReport(
        overallScore=overall,
        level=level,
        wordCount=doc.word_count,
        embeddingsComputed=embeddings_computed,
        requiresHumanReview=requires_review,
        criteria=criteria,
    )
