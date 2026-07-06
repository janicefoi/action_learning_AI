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

def _analytical_depth_feedback(score: float) -> str:
    """
    Four-band own scale for analytical depth.

      0.60+     → Analytical   — rich causal reasoning, multi-dimensional analysis
      0.40-0.59 → Developing   — some analysis; needs more causal depth
      0.20-0.39 → Descriptive  — events narrated rather than analysed
      0.00-0.19 → Minimal      — no analytical content detected
    """
    if score >= 0.60:
        return (
            "Strong analytical depth — clear causal reasoning and multi-dimensional "
            "problem analysis are evident throughout."
        )
    if score >= 0.40:
        return (
            "Developing analytical depth — some causal reasoning is present, but the "
            "analysis could go deeper. Identify root causes and examine the problem "
            "from multiple perspectives."
        )
    if score >= 0.20:
        return (
            "Mostly descriptive — events are narrated rather than analysed. "
            "Explain WHY the problem occurred, not just WHAT happened. "
            "Use language such as: 'because', 'the root cause was', 'as a result'."
        )
    return (
        "No analytical content detected. Shift from describing events to examining "
        "their causes, underlying dynamics, and implications. "
        "A strong reflection explains WHY things happened, not just WHAT."
    )


def score_analytical_depth(doc: CleanDocument) -> CriterionScore:
    """
    Analytical depth scorer — four-band own scale.

    Distinguishes reflections that ANALYSE (identify root causes, reason
    causally, examine from multiple angles) from those that merely DESCRIBE
    (narrate events without explaining why they occurred).

    Components:
      40% — semantic similarity to analytical rubric reference sentences
      40% — causal density: raw count of causal connectives; 5+ hits = full marks
      20% — high-signal analytical vocabulary bonus

    The Problem section is weighted twice because root-cause analysis lives
    there; the Reflection section is included once.
    """
    problem_raw = doc.sections.get("Problem", "").strip()
    reflection_raw = doc.sections.get("Reflection", "").strip()

    if problem_raw or reflection_raw:
        # Problem double-weighted — root-cause analysis lives there
        combined = " ".join(filter(None, [problem_raw, problem_raw, reflection_raw]))
    else:
        combined = doc.full_text

    lower = combined.lower()
    wc = len(combined.split()) if combined.strip() else 0

    # Component 1: semantic similarity to analytical rubric sentences (40%)
    ref_sim = _ref_sim(combined, "analytical_depth") if combined.strip() else 0.0

    # Component 2: causal density — actual occurrence count (not just presence);
    # 5+ causal connectives across the text = full marks on this component (40%)
    causal_connectives = [
        "because", "therefore", "as a result", "root cause", "due to",
        "consequently", "led to", "stemmed from", "underlying", "resulted in",
        "caused by", "attributed to", "hence", "thus", "the reason",
        "driven by", "explains why", "which means", "a consequence of",
        "this is why", "which caused",
    ]
    causal_hits = sum(lower.count(kw) for kw in causal_connectives)
    causal_density = min(1.0, causal_hits / 5.0)

    # Component 3: high-signal depth vocabulary bonus (20%)
    depth_markers = [
        "root cause", "underlying", "systemic", "structural", "multiple factors",
        "interconnected", "assumption", "from multiple", "multi-dimensional",
        "compounded", "evidence", "demonstrates", "reveals that", "suggests that",
        "indicates that", "on closer examination", "beyond the surface",
        "deeper analysis", "critical factor", "at its core",
    ]
    depth_bonus = _lexical_bonus(lower, depth_markers, 0.20)

    score = min(1.0, max(0.0, 0.40 * ref_sim + 0.40 * causal_density + depth_bonus))
    conf = _confidence(score, wc)

    return CriterionScore(
        label="Analytical Depth",
        score=round(score, 3),
        feedback=_analytical_depth_feedback(score),
        confidence=conf,
        requiresReview=(conf == "low"),
    )


def _outcome_linkage_feedback(score: float) -> str:
    """
    Four-band own scale for outcome linkage.

      0.65+     → Linked   — clear action-outcome connections with explicit attribution
      0.45-0.64 → Partial  — some connections made but not fully explicit
      0.25-0.44 → Implied  — outcomes mentioned but not linked to specific actions
      0.00-0.24 → Absent   — no connection between actions and results detected
    """
    if score >= 0.65:
        return (
            "Clear outcome linkage — the reflection explicitly connects actions to "
            "their results with measurable or observable evidence."
        )
    if score >= 0.45:
        return (
            "Partial outcome linkage — some connections between actions and results "
            "are made. Strengthen these by stating specifically what changed as a "
            "direct result of each action taken."
        )
    if score >= 0.25:
        return (
            "Outcomes are implied but not explicitly linked to actions. State clearly: "
            "'Because I did X, Y happened.' Include observable or measurable evidence "
            "of the change produced."
        )
    return (
        "No clear connection between actions and results. Explicitly state what "
        "changed as a result of the actions taken, and why those outcomes occurred."
    )


def score_outcome_linkage(doc: CleanDocument) -> CriterionScore:
    """
    Outcome linkage scorer — four-band own scale.

    Does the student explicitly connect what they DID (Action) to what
    RESULTED (Reflection)? Rewards explicit causal attribution, measurable
    outcomes, and direct action-result phrasing.

    Components:
      35% — semantic similarity to outcome-linkage rubric reference sentences
      45% — bridge-phrase density: actual occurrence count; 3+ hits = full marks
      20% — outcome evidence vocabulary bonus
    """
    action_raw = doc.sections.get("Action", "").strip()
    reflection_raw = doc.sections.get("Reflection", "").strip()

    if action_raw or reflection_raw:
        combined = " ".join(filter(None, [action_raw, reflection_raw]))
    else:
        combined = doc.full_text

    lower = combined.lower()
    wc = len(combined.split()) if combined.strip() else 0

    # Component 1: semantic similarity to outcome-linkage rubric sentences (35%)
    ref_sim = _ref_sim(combined, "outcome_linkage") if combined.strip() else 0.0

    # Component 2: bridge-phrase density — phrases that explicitly connect an action
    # to an outcome; 3+ occurrences = full marks on this component (45%)
    bridge_phrases = [
        "as a result", "led to", "resulted in", "consequently", "therefore",
        "the outcome was", "the result was", "this caused", "which meant",
        "because of this", "this produced", "the impact was",
        "following this", "after which", "thanks to", "by doing this",
        "this led", "which led", "this change", "this meant",
        "the effect was", "measurably improved", "worked because",
    ]
    bridge_hits = sum(lower.count(bp) for bp in bridge_phrases)
    bridge_density = min(1.0, bridge_hits / 3.0)

    # Component 3: outcome evidence vocabulary bonus (20%)
    outcome_markers = [
        "measurable", "evidence", "improved", "increased", "decreased",
        "reduced", "achieved", "delivered", "observed", "confirmed",
        "data", "feedback", "survey", "metric", "concrete",
        "visible", "demonstrated", "tangible", "dropped", "quantified",
    ]
    outcome_bonus = _lexical_bonus(lower, outcome_markers, 0.20)

    score = min(1.0, max(0.0, 0.35 * ref_sim + 0.45 * bridge_density + outcome_bonus))
    conf = _confidence(score, wc)

    return CriterionScore(
        label="Outcome Linkage",
        score=round(score, 3),
        feedback=_outcome_linkage_feedback(score),
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
