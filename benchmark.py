"""
Model selection benchmark for the Action Learning AI scoring service.

Run:  python benchmark.py

Evaluates all-MiniLM-L6-v2 against two contrasting sample reflections
(high quality vs low quality) to verify the model discriminates well
across the four scoring criteria used in the thesis rubric.
"""
from model_loader import load_model
from scoring import score_document
from text_cleaner import parse_document

HIGH_QUALITY = """
Problem
Over the course of this action-learning set, I identified a deep-rooted communication
failure within our project team. The root cause was not simply poor communication habits
but a structural misalignment: different team members operated under conflicting assumptions
about decision-making authority. Because no one explicitly owns decisions, everyone assumes
someone else does, leading to chronic inaction on critical issues. I analysed stakeholder
interviews, reviewed project artefacts, and examined meeting minutes to identify this
pattern, which corroborated research on accountability gaps in matrix-structured teams.

Action
I proposed and facilitated a structured decision-rights workshop based on the RACI framework.
I prepared the session materials, coordinated with the team lead to secure buy-in from senior
stakeholders, and ran two two-hour sessions. I deliberately drew out dissenting voices to
surface hidden assumptions. As a direct result, the team agreed on explicit ownership for the
six most contentious decision categories and updated the project charter to reflect this.

Reflection
The outcome was measurable: in the three weeks following the workshop, the number of
escalations dropped by more than half, and team members reported feeling clearer about their
roles in post-session surveys. The intervention worked because it addressed the systemic cause
rather than the surface symptoms. I realise I initially underestimated resistance from middle
managers who feared losing informal authority, and I had to adapt my facilitation approach
mid-session to address this dynamic. This required me to draw on emotional intelligence skills
I had not previously exercised at this level.

Learning
Going forward, I will begin every project engagement with a decision-rights mapping exercise
rather than waiting until conflict surfaces. I will also invest in developing my facilitation
skills by attending a certified workshop facilitation programme in the next quarter. The broader
lesson is that organisational problems are rarely individual failures; they are almost always
systemic, and addressing root causes demands structural intervention rather than behavioural nudges.
"""

LOW_QUALITY = """
Problem
We had some problems in the team. People were not communicating well and things were not working.
It was difficult and everyone was stressed. The project was not going well.

Action
I talked to my team and we had a meeting. We discussed the problems and tried to fix things.
I also sent some emails. We had another meeting later.

Reflection
The meeting went ok. Some things got better. I think the team was happier after. Not everything
was fixed but it was a bit better than before.

Learning
I learned that communication is important. Next time I will try to communicate better.
"""


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score:.3f}"


def run() -> None:
    print("Loading model and pre-computing reference embeddings…")
    load_model()
    print()

    docs = [
        ("HIGH-QUALITY reflection", HIGH_QUALITY),
        ("LOW-QUALITY reflection",  LOW_QUALITY),
    ]

    results = []
    for label, text in docs:
        doc = parse_document(text)
        report = score_document(doc)
        results.append((label, report))

    header = f"{'Criterion':<22}  {'HIGH':>22}   {'LOW':>22}   {'Delta':>7}"
    print(header)
    print("-" * len(header))

    criteria_labels = [c.label for c in results[0][1].criteria]
    for i, crit_label in enumerate(criteria_labels):
        high_score = results[0][1].criteria[i].score
        low_score  = results[1][1].criteria[i].score
        delta      = high_score - low_score
        print(f"{crit_label:<22}  {_bar(high_score)}   {_bar(low_score)}   {delta:+.3f}")

    print()
    for label, report in results:
        marker = "✓" if report.overallScore > 0.50 else "✗"
        print(f"{marker} {label}")
        print(f"  Overall: {report.overallScore:.3f}  level={report.level}  "
              f"requiresHumanReview={report.requiresHumanReview}")
        for c in report.criteria:
            flag = " [FLAG]" if c.requiresReview else ""
            print(f"  {c.label:<22} {c.score:.3f}  conf={c.confidence}{flag}")
        print()

    high_overall = results[0][1].overallScore
    low_overall  = results[1][1].overallScore
    separation   = high_overall - low_overall
    print(f"Score separation (high - low): {separation:+.3f}")
    print(f"Model: all-MiniLM-L6-v2  |  embedding dim: 384  |  param count: ~22M")
    print()
    print("Rationale for model choice:")
    print("  all-MiniLM-L6-v2 is a distilled MiniLM encoder fine-tuned on 1B+ sentence pairs")
    print("  (MS MARCO, NLI, paraphrase corpora). It achieves near BERT-large accuracy on")
    print("  semantic similarity benchmarks at 5x the inference speed and 1/6 the parameter")
    print("  count, making it well-suited for synchronous per-document scoring in this service.")
    print("  The 384-dimensional embeddings provide sufficient representational capacity for")
    print("  nuanced reflection quality distinctions while keeping latency low.")


if __name__ == "__main__":
    run()
