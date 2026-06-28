"""
reasoning.py — Redrob Hackathon
==================================
Generates per-candidate 1–2 sentence reasoning strings from actual profile
facts for the submission CSV.

Rules (from submission_spec.md Stage 4 evaluation)
----------------------------------------------------
- 1–2 sentences maximum
- Must cite specific facts from the candidate record (not templates)
- Must connect to the JD requirement being satisfied or unmet
- Must vary per candidate — identical strings across rows are penalised
- Must be honest about gaps (don't oversell a weak candidate)
- No hallucination — every fact mentioned must exist in the data

Architecture
------------
Each reasoning string is assembled from fact-extraction functions that pull
real values from the candidate record and the scorer results, then composed
into a natural sentence via a small set of structure templates.

The key variation driver is which facts are most salient for THAT candidate:
  - For a strong candidate: lead with their best signal + JD connection
  - For a mid-tier candidate: note the strength + flag the key caveat
  - For a bottom-of-100 candidate: explain why they're included despite limits
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from scorers.skills   import SkillScoreResult
from scorers.career   import CareerScoreResult
from scorers.experience import ExperienceScoreResult
from signals          import SignalResult

# ─────────────────────────────────────────────────────────────────────────────
# Fact extractors — pull clean, human-readable facts from data + scorer results
# ─────────────────────────────────────────────────────────────────────────────

def _top_skills(skill_result: SkillScoreResult, n: int = 3) -> str:
    """Return top-n matched skills as a readable string."""
    skills = (skill_result.retrieval_skills_detail or [])
    names  = [s["skill"] for s in skills[:n]]
    if not names:
        names = skill_result.must_have_matched[:n]
    if not names:
        return "limited relevant skills"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _yoe_phrase(exp_result: ExperienceScoreResult) -> str:
    """Return a YOE phrase like '7.2 years' or '6 years (stated)'."""
    yoe = exp_result.effective_yoe
    stated = exp_result.stated_yoe
    if abs(yoe - stated) > 0.5:
        return f"{yoe:.1f} years (blended from career record)"
    return f"{yoe:.1f} years"


def _career_highlight(career_result: CareerScoreResult) -> str:
    """Return the strongest single career evidence sentence."""
    if career_result.retrieval_evidence:
        # Take first evidence item, clean it up
        ev = career_result.retrieval_evidence[0]
        # Format: "NLP Engineer @ Uber (27mo): search, ranking, embeddings"
        return ev
    if career_result.product_months >= 24:
        return f"{career_result.product_months // 12} years at product companies"
    return ""


def _notice_phrase(signal_result: SignalResult) -> str:
    nd = signal_result.notice_period_days
    if nd <= 15:
        return "immediately available"
    elif nd <= 30:
        return f"{nd}-day notice"
    elif nd <= 60:
        return f"{nd}-day notice (buyout possible)"
    else:
        return f"{nd}-day notice period"


def _location_phrase(candidate: dict) -> str:
    loc = candidate["profile"].get("location", "")
    relocate = candidate["redrob_signals"].get("willing_to_relocate", False)
    city = loc.split(",")[0].strip().lower()
    preferred = {"pune", "noida"}
    acceptable = {"hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
                  "gurgaon", "gurugram"}
    if city in preferred:
        return f"based in {loc.split(',')[0]}"
    elif city in acceptable and relocate:
        return f"based in {loc.split(',')[0]}, willing to relocate"
    elif relocate:
        return f"willing to relocate from {loc.split(',')[0]}"
    else:
        return f"based in {loc.split(',')[0]}, not open to relocation"


def _missing_skills_phrase(skill_result: SkillScoreResult, max_n: int = 2) -> str:
    missing = skill_result.must_have_missing[:max_n]
    if not missing:
        return ""
    if len(missing) == 1:
        return f"no evidence of {missing[0]}"
    return f"gaps in {' and '.join(missing)}"


def _activity_phrase(signal_result: SignalResult) -> str:
    days = signal_result.days_since_active
    if days <= 7:
        return "active this week"
    elif days <= 14:
        return "active recently"
    elif days <= 30:
        return "active in past month"
    elif days <= 60:
        return "active ~2 months ago"
    elif days <= 90:
        return "last active ~3 months ago"
    else:
        return f"last active {days} days ago"


def _response_phrase(signal_result: SignalResult) -> str:
    rr = signal_result.responsiveness_score
    if rr >= 0.80:
        return "highly responsive to recruiters"
    elif rr >= 0.55:
        return "reasonably responsive"
    elif rr >= 0.35:
        return "low recruiter engagement"
    else:
        return "poor recruiter responsiveness"


def _github_phrase(candidate: dict) -> str:
    g = candidate["redrob_signals"].get("github_activity_score", -1)
    if g == -1:
        return ""
    elif g >= 70:
        return f"strong open-source activity (GitHub score {g:.0f})"
    elif g >= 40:
        return f"active GitHub presence (score {g:.0f})"
    else:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tier classifier — determines the tone and structure of the reasoning string
# ─────────────────────────────────────────────────────────────────────────────

def _classify_tier(final_score: float) -> str:
    """Classify candidate into reasoning tier based on final combined score."""
    if final_score >= 0.70:
        return "strong"
    elif final_score >= 0.50:
        return "good"
    elif final_score >= 0.35:
        return "mid"
    elif final_score >= 0.20:
        return "weak"
    else:
        return "bottom"


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning builder — assembles the string from extracted facts
# ─────────────────────────────────────────────────────────────────────────────

def build_reasoning(
    candidate: dict[str, Any],
    final_score: float,
    skill_result: SkillScoreResult,
    career_result: CareerScoreResult,
    exp_result: ExperienceScoreResult,
    signal_result: SignalResult,
) -> str:
    """
    Build a 1–2 sentence reasoning string from real profile facts.
    Varies structure based on the candidate's tier and most salient signals.
    """
    tier     = _classify_tier(final_score)
    profile  = candidate.get("profile", {})
    title    = profile.get("current_title", "?")
    company  = profile.get("current_company", "?")

    yoe      = _yoe_phrase(exp_result)
    notice   = _notice_phrase(signal_result)
    location = _location_phrase(candidate)
    top_sk   = _top_skills(skill_result)
    career_h = _career_highlight(career_result)
    missing  = _missing_skills_phrase(skill_result)
    activity = _activity_phrase(signal_result)
    response = _response_phrase(signal_result)
    github   = _github_phrase(candidate)

    # ── STRONG (≥0.70): lead with best signal, end with logistics ──────────
    if tier == "strong":
        s1_parts = [f"{title} at {company} with {yoe} of experience"]
        if career_h:
            s1_parts.append(f"including {career_h}")
        if top_sk and top_sk != "limited relevant skills":
            s1_parts.append(f"expert in {top_sk}")
        s1 = "; ".join(s1_parts) + "."

        s2_parts = [location.capitalize(), notice]
        if github:
            s2_parts.append(github)
        if signal_result.responsiveness_score >= 0.70:
            s2_parts.append("highly responsive")
        s2 = ", ".join(s2_parts) + "."
        return f"{s1} {s2}"

    # ── GOOD (0.50–0.70): strength + one caveat ────────────────────────────
    elif tier == "good":
        strength = ""
        if career_result.retrieval_score >= 0.50 and career_h:
            strength = f"career evidence of retrieval/ranking work ({career_h})"
        elif skill_result.must_have_score >= 0.50:
            strength = f"solid JD skill coverage including {top_sk}"
        elif exp_result.yoe_label == "ideal":
            strength = f"{yoe} of experience in the JD sweet spot"
        else:
            strength = f"relevant background as {title}"

        caveat = ""
        if career_result.is_pure_consulting:
            caveat = "entire career in consulting/services"
        elif exp_result.yoe_label in ("over", "slightly over"):
            caveat = f"at {yoe} may be over-experienced for the scope"
        elif missing:
            caveat = missing
        elif signal_result.notice_period_days > 60:
            caveat = f"{signal_result.notice_period_days}-day notice period"
        elif signal_result.responsiveness_score < 0.40:
            caveat = "low recruiter engagement historically"

        s1 = f"{title} with {strength}"
        if caveat:
            s1 += f"; note {caveat}."
        else:
            s1 += "."

        s2 = f"{location.capitalize()}, {notice}, {activity}."
        return f"{s1} {s2}"

    # ── MID (0.35–0.50): honest about trade-offs ───────────────────────────
    elif tier == "mid":
        best = ""
        if skill_result.must_have_score >= 0.30:
            best = f"partial JD skill match ({top_sk})"
        elif career_result.product_months >= 12:
            best = f"{career_result.product_months // 12}yr product-company background"
        elif exp_result.yoe_label in ("ideal", "acceptable"):
            best = f"{yoe} experience in the acceptable range"
        else:
            best = f"{yoe} experience as {title}"

        gaps = []
        if skill_result.must_have_score < 0.30:
            gaps.append("weak JD skill coverage")
        if career_result.is_pure_consulting:
            gaps.append("pure consulting background")
        if missing:
            gaps.append(missing)
        if exp_result.yoe_label == "hard floor":
            gaps.append("below minimum experience threshold")

        gap_str = "; ".join(gaps[:2]) if gaps else "limited direct retrieval/ranking evidence"
        s1 = f"Included for {best}, though {gap_str}."
        s2 = f"{location.capitalize()}, {notice}."
        return f"{s1} {s2}"

    # ── WEAK (0.20–0.35): borderline, explain why ranked ───────────────────
    elif tier == "weak":
        reason = ""
        if signal_result.multiplier >= 0.70:
            reason = "strong availability and responsiveness signals"
        elif exp_result.yoe_label in ("ideal", "acceptable"):
            reason = f"{yoe} experience fits the JD band"
        elif career_result.product_months >= 24:
            reason = f"{career_result.product_months // 12}yr product-company history"
        else:
            reason = "marginal fit across all dimensions"

        limit = ""
        if skill_result.must_have_score < 0.15:
            limit = "minimal relevant skill evidence"
        elif career_result.retrieval_score < 0.10:
            limit = "no clear retrieval/ranking career evidence"
        elif career_result.is_pure_consulting:
            limit = "entire career in IT services"
        else:
            limit = "limited direct JD alignment"

        s1 = f"Borderline inclusion: {reason}, but {limit}."
        s2 = f"{location.capitalize()}, {notice}."
        return f"{s1} {s2}"

    # ── BOTTOM (<0.20): tail of the top-100, very honest ───────────────────
    else:
        s1 = (
            f"Ranked at the tail of top-100 due to limited JD alignment "
            f"({title}, {yoe}); included as a potential outlier given "
            f"{activity} and {notice}."
        )
        return s1


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningResult:
    candidate_id: str
    reasoning: str
    tier: str
    char_count: int


def generate_reasoning(
    candidate: dict[str, Any],
    final_score: float,
    skill_result: SkillScoreResult,
    career_result: CareerScoreResult,
    exp_result: ExperienceScoreResult,
    signal_result: SignalResult,
) -> ReasoningResult:
    """Generate and return a ReasoningResult for one candidate."""
    text = build_reasoning(
        candidate, final_score,
        skill_result, career_result, exp_result, signal_result,
    )
    return ReasoningResult(
        candidate_id=candidate.get("candidate_id", "?"),
        reasoning=text,
        tier=_classify_tier(final_score),
        char_count=len(text),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (run: python reasoning.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pathlib
    from scorers.skills    import SkillScorer
    from scorers.career    import CareerScorer
    from scorers.experience import ExperienceScorer
    from signals           import SignalScorer
    import config

    sample_path = pathlib.Path("/mnt/user-data/uploads/sample_candidates.json")
    with open(sample_path) as f:
        candidates = json.load(f)

    sk_scorer  = SkillScorer()
    ca_scorer  = CareerScorer()
    ex_scorer  = ExperienceScorer()
    sig_scorer = SignalScorer()

    rows = []
    for c in candidates:
        sk  = sk_scorer.score(c)
        ca  = ca_scorer.score(c)
        ex  = ex_scorer.score(c)
        sg  = sig_scorer.score(c)

        base = (
            config.WEIGHTS["skills"]     * sk.final_score
            + config.WEIGHTS["career"]   * ca.final_score
            + config.WEIGHTS["experience"] * ex.final_score
            + config.WEIGHTS["location"] * 0.70  # placeholder
        )
        final = round(base * sg.multiplier, 6)
        r = generate_reasoning(c, final, sk, ca, ex, sg)
        rows.append((final, c, r))

    rows.sort(key=lambda x: x[0], reverse=True)

    print("=== REASONING STRINGS (top 20) ===\n")
    for rank, (score, c, r) in enumerate(rows[:20], 1):
        print(f"Rank {rank:>2} | {c['candidate_id']} | score={score:.3f} | tier={r.tier} | chars={r.char_count}")
        print(f"  {r.reasoning}")
        print()

    print("=== VARIATION CHECK ===")
    texts = [r.reasoning for _, _, r in rows[:20]]
    unique = len(set(texts))
    print(f"  Unique strings in top 20: {unique}/20 {'✓' if unique == 20 else '⚠ DUPLICATES FOUND'}")

    print()
    print("=== TIER DISTRIBUTION ===")
    from collections import Counter
    tiers = Counter(r.tier for _, _, r in rows)
    for tier in ["strong", "good", "mid", "weak", "bottom"]:
        count = tiers.get(tier, 0)
        print(f"  {tier:8}: {count:3d}  {'█' * count}")
