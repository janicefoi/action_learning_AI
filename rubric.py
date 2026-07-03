"""
Reference sentences for each scoring criterion.
These are pre-embedded at startup and used for cosine similarity scoring.
Each list represents high-quality exemplar responses for that criterion.
"""
from typing import Dict, List

ANALYTICAL_DEPTH = [
    "The root cause of the problem was a fundamental misalignment between team expectations and project requirements that compounded over time",
    "Analyzing the situation revealed multiple interconnected factors: unclear accountability, insufficient stakeholder communication, and flawed assumptions",
    "I identified that the surface symptoms masked a deeper structural issue in how decisions were being made across the team",
    "The problem stemmed from untested assumptions that led to cascading errors throughout the project lifecycle",
    "By examining the problem from multiple angles I uncovered systemic patterns that were not visible from a single perspective",
    "The underlying cause was organizational rather than individual, reflecting a broader gap in team dynamics and shared understanding",
]

OUTCOME_LINKAGE = [
    "As a direct result of the intervention, team communication improved significantly and conflict decreased measurably within two weeks",
    "The action I took led to concrete improvements in performance, specifically in response time and stakeholder satisfaction",
    "By addressing the root cause directly we achieved measurable improvements in delivery quality and team cohesion",
    "The change in approach produced clear outcomes: better collaboration, clearer accountability, and faster decision-making",
    "Implementing the new strategy led directly to the positive outcomes observed in the subsequent project phase",
    "The results of the action were immediately visible in improved team dynamics and higher quality deliverables",
]

LOGICAL_COHERENCE = [
    "Building on the analysis of the problem, the action taken was a logical and structured response to the identified root cause",
    "The sequence of events follows a clear narrative: the problem created a need, the action addressed it, and the reflection captures the learning",
    "Each section connects directly to the previous one, forming a coherent account of the experience from start to finish",
    "The reasoning flows naturally from identifying the issue to taking deliberate action to reflecting on the resulting outcome",
    "This reflection presents a well-structured argument where each element supports and builds upon the previous one",
]

FORWARD_ORIENTATION = [
    "In future projects I will establish clear communication protocols from the outset to prevent similar issues arising",
    "The key learning is to consult all stakeholders before making decisions that affect the entire team",
    "Going forward I plan to apply a more structured reflection practice after each major project milestone",
    "I am committed to developing my conflict resolution skills and will seek mentorship opportunities to strengthen this capability",
    "Next time I face a similar challenge I will seek diverse perspectives earlier and validate assumptions more rigorously",
    "I will implement regular team check-ins to identify problems before they escalate, based on what I learned here",
]

ALL_REFERENCES: Dict[str, List[str]] = {
    "analytical_depth": ANALYTICAL_DEPTH,
    "outcome_linkage": OUTCOME_LINKAGE,
    "logical_coherence": LOGICAL_COHERENCE,
    "forward_orientation": FORWARD_ORIENTATION,
}
