"""
Phase 3 — four criterion-level scorers.

Each scorer returns a CriterionScore (Pydantic model) with:
  label, score (0-1), feedback, confidence, requiresReview
"""
import re
from typing import List

import numpy as np
from sentence_transformers import util

from model_loader import embed, get_reference_embeddings
from schemas import CriterionScore
from text_cleaner import CleanDocument


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_text(doc: CleanDocument, *section_names: str) -> str:
    """
    Return text from the first matching section found.
    If none of the named sections exist, fall back to all detected section text
    combined, then to the full document — so free-form assignments still get scored.
    """
    for name in section_names:
        content = doc.sections.get(name, "").strip()
        if content:
            return content
    all_sections = " ".join(v for v in doc.sections.values() if v.strip()).strip()
    return all_sections if all_sections else doc.full_text


def _ref_sim(text: str, key: str) -> float:
    """Mean cosine similarity between text embedding and pre-computed reference embeddings."""
    if not text.strip():
        return 0.0
    emb = embed([text])[0]
    refs = get_reference_embeddings(key)
    sims = util.cos_sim(emb, refs)        # tensor (1, n_refs)
    return float(sims.mean().item())


def _lexical_bonus(text: str, keywords: List[str], max_bonus: float = 0.10) -> float:
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw in lower)
    return min(max_bonus, hits * (max_bonus / max(len(keywords), 1)))


def _confidence(score: float, word_count: int) -> str:
    if word_count < 30:
        return "low"
    if word_count < 80 or 0.30 < score < 0.50:
        return "medium"
    return "high"


def _feedback(criterion: str, score: float) -> str:
    templates = {
        "analytical_depth": [
            "Strong analytical depth. The reflection demonstrates clear causal reasoning and multi-dimensional problem analysis.",
            "Some analytical depth present. Consider exploring root causes more deeply and examining the problem from multiple angles.",
            "Analytical depth needs development. Explain WHY the problem occurred and identify underlying causes rather than describing surface events.",
        ],
        "outcome_linkage": [
            "Clear linkage between actions and outcomes. The reflection effectively connects what was done to what resulted.",
            "Partial outcome linkage. Strengthen the connection by explicitly stating how specific actions produced specific results.",
            "Outcome linkage is unclear. Make explicit connections between the actions taken and the results observed.",
        ],
        "logical_coherence": [
            "The reflection flows logically from problem through action to learning. Ideas are well connected throughout.",
            "Reasonable coherence. Consider using connective language to more explicitly link ideas across sections.",
            "The reflection would benefit from clearer logical flow. Ensure each section builds naturally on the previous one.",
        ],
        "forward_orientation": [
            "Strong forward orientation. Concrete plans and specific commitments to future change are evident.",
            "Some forward orientation present. Make learning more actionable with specific steps you will take in future.",
            "Forward orientation is limited. Identify specific things you will do differently next time and commit to concrete actions.",
        ],
    }
    bands = templates.get(criterion, ["Good.", "Average.", "Needs improvement."])
    if score >= 0.65:
        return bands[0]
    if score >= 0.40:
        return bands[1]
    return bands[2]


# ── criterion scorers ─────────────────────────────────────────────────────────

def score_analytical_depth(doc: CleanDocument) -> CriterionScore:
    """How deeply does the student analyse the problem? Prefers Problem + Reflection; falls back to full text."""
    problem = _get_text(doc, "Problem")
    reflection = _get_text(doc, "Reflection")
    text = f"{problem} {reflection}".strip() if problem != reflection else problem
    wc = len(text.split())
    sim = _ref_sim(text, "analytical_depth")

    causal_kw = [
        "because", "therefore", "as a result", "root cause", "due to",
        "consequently", "caused", "led to", "stemmed from", "underlying",
        "resulted in", "factor", "reason",
    ]
    score = min(1.0, max(0.0, sim + _lexical_bonus(text, causal_kw, 0.12)))
    conf = _confidence(score, wc)
    return CriterionScore(
        label="Analytical Depth",
        score=round(score, 3),
        feedback=_feedback("analytical_depth", score),
        confidence=conf,
        requiresReview=(conf == "low"),
    )


def score_outcome_linkage(doc: CleanDocument) -> CriterionScore:
    """Does the student explicitly link actions to outcomes? Prefers Action + Reflection; falls back to full text."""
    action = _get_text(doc, "Action")
    reflection = _get_text(doc, "Reflection")
    combined = f"{action} {reflection}".strip() if action != reflection else action
    wc = len(combined.split())

    ref_sim = _ref_sim(combined, "outcome_linkage")

    # Causal-bridge bonus: reward explicit causal language (not raw topical overlap,
    # which falsely rewards vague sections that are both about the same topic).
    causal_bridge_kw = [
        "as a result", "led to", "because of this", "this caused", "which meant",
        "consequently", "therefore", "the outcome was", "this resulted in",
        "the impact was", "directly improved", "this produced",
    ]
    bridge_bonus = _lexical_bonus(combined, causal_bridge_kw, 0.15)

    outcome_kw = [
        "result", "impact", "improved", "achieved", "outcome",
        "consequence", "effect", "produced", "measurable", "evidence",
    ]
    bonus = _lexical_bonus(combined, outcome_kw, 0.08)

    score = min(1.0, max(0.0, ref_sim + bridge_bonus + bonus))
    conf = _confidence(score, wc)
    return CriterionScore(
        label="Outcome Linkage",
        score=round(score, 3),
        feedback=_feedback("outcome_linkage", score),
        confidence=conf,
        requiresReview=(conf == "low"),
    )


def score_logical_coherence(doc: CleanDocument) -> CriterionScore:
    """Does the writing flow logically? Measured by sentence-to-sentence similarity."""
    text = doc.full_text
    wc = doc.word_count

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 20]
    seq_sim = 0.0
    if len(sentences) >= 3:
        embs = embed(sentences)
        pairs = [
            max(0.0, float(util.cos_sim(embs[i], embs[i + 1])[0][0].item()))
            for i in range(len(embs) - 1)
        ]
        seq_sim = float(np.mean(pairs)) if pairs else 0.0

    ref_sim = _ref_sim(text[:1000], "logical_coherence")

    discourse_kw = [
        "however", "furthermore", "therefore", "in addition", "as a result",
        "consequently", "moreover", "in contrast", "building on", "additionally",
        "this led", "which meant",
    ]
    bonus = _lexical_bonus(text, discourse_kw, 0.10)
    score = min(1.0, max(0.0, 0.50 * seq_sim + 0.40 * ref_sim + bonus))
    conf = _confidence(score, wc)
    return CriterionScore(
        label="Logical Coherence",
        score=round(score, 3),
        feedback=_feedback("logical_coherence", score),
        confidence=conf,
        requiresReview=(conf == "low"),
    )


def score_forward_orientation(doc: CleanDocument) -> CriterionScore:
    """Does the student commit to concrete future actions? Prefers Learning section; falls back to full text."""
    text = _get_text(doc, "Learning")
    wc = len(text.split())

    sim = _ref_sim(text, "forward_orientation")

    future_kw = [
        "will", "plan to", "next time", "going forward", "in future",
        "intend", "commit", "aim to", "shall", "i will", "i plan",
    ]
    score = min(1.0, max(0.0, sim + _lexical_bonus(text, future_kw, 0.12)))
    conf = _confidence(score, wc)
    return CriterionScore(
        label="Forward Orientation",
        score=round(score, 3),
        feedback=_feedback("forward_orientation", score),
        confidence=conf,
        requiresReview=(conf == "low"),
    )
