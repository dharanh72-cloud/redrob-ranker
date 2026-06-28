"""
scorers/skills.py — Redrob Hackathon
======================================
Scores how well a candidate's skills match the JD for Senior AI Engineer.

Design philosophy
-----------------
A raw keyword match is useless — keyword stuffers will win it.
This scorer builds a *trusted skill score* for each skill by combining:

  1. Relevance   — is this skill in the JD must-have / nice-to-have list?
  2. Proficiency — what level does the candidate claim? (expert → beginner)
  3. Duration    — how many months have they actually used it?
  4. Assessment  — did the platform verify it? (assessment score 0-100)
  5. Endorsements — how many peers confirmed it?

The final score is the weighted combination of must-have coverage,
nice-to-have bonus, retrieval-core depth, and a trust modifier that
down-weights any skill whose evidence is thin.

Score range: 0.0 – 1.0
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass, field
from typing import Any

# Allow running from repo root OR scorers/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Constants (skill-scorer-specific, derived from config)
# ─────────────────────────────────────────────────────────────────────────────

# Normalised must-have / nice-to-have sets for fast O(1) lookup
_MUST_HAVE: set[str] = {s.lower() for s in config.MUST_HAVE_SKILLS}
_NICE_TO_HAVE: set[str] = {s.lower() for s in config.NICE_TO_HAVE_SKILLS}
_RETRIEVAL_CORE: set[str] = {s.lower() for s in config.RETRIEVAL_CORE_SKILLS}

# Sub-component weights inside the skill scorer (must sum to 1.0)
_W_MUST_HAVE      = 0.50   # coverage of must-have skills
_W_RETRIEVAL_DEPTH = 0.25  # depth of retrieval-core skills (duration + level)
_W_NICE_TO_HAVE   = 0.15   # bonus from nice-to-have skills
_W_TRUST_PENALTY  = 0.10   # trust modifier from assessments / endorsements

assert abs(_W_MUST_HAVE + _W_RETRIEVAL_DEPTH + _W_NICE_TO_HAVE + _W_TRUST_PENALTY - 1.0) < 1e-9

# Duration soft cap — beyond this months, extra time gives diminishing returns
_DURATION_SOFT_CAP = 60    # months

# Endorsement soft cap (matches config)
_ENDORSEMENT_CAP = config.ENDORSEMENT_MAX_SCORE   # 50


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillScoreResult:
    candidate_id: str
    final_score: float = 0.0

    # Sub-scores (0.0 – 1.0 each, before weighting)
    must_have_score: float = 0.0
    retrieval_depth_score: float = 0.0
    nice_to_have_score: float = 0.0
    trust_modifier: float = 1.0      # multiplier, not additive

    # Diagnostic detail
    must_have_matched: list[str] = field(default_factory=list)
    must_have_missing: list[str] = field(default_factory=list)
    nice_to_have_matched: list[str] = field(default_factory=list)
    retrieval_skills_detail: list[dict] = field(default_factory=list)
    trust_flags: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<SkillScore {self.candidate_id} "
            f"final={self.final_score:.3f} "
            f"must={self.must_have_score:.2f} "
            f"depth={self.retrieval_depth_score:.2f} "
            f"nice={self.nice_to_have_score:.2f} "
            f"trust={self.trust_modifier:.2f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase and strip for consistent matching."""
    return text.lower().strip()


def _proficiency_weight(proficiency: str) -> float:
    """Map proficiency string to a 0–1 weight from config."""
    return config.PROFICIENCY_WEIGHTS.get(_normalise(proficiency), 0.0)


def _duration_score(duration_months: int | None) -> float:
    """
    Convert months of usage to a 0–1 score with a logarithmic curve.

    - 0 months  → 0.0
    - 6 months  → 0.35
    - 12 months → 0.55
    - 24 months → 0.75
    - 48 months → 0.92
    - 60+ months → ~1.0  (soft cap)
    """
    if not duration_months or duration_months <= 0:
        return 0.0
    capped = min(duration_months, _DURATION_SOFT_CAP)
    return round(math.log1p(capped) / math.log1p(_DURATION_SOFT_CAP), 4)


def _assessment_trust(skill_name: str, assessments: dict[str, float]) -> float | None:
    """
    Return trust multiplier (0.0–1.0) from platform assessment score, or None
    if no assessment exists for this skill.

    Uses the piecewise curve defined in config.ASSESSMENT_TRUST_CURVE.
    """
    # Case-insensitive lookup
    key = next(
        (k for k in assessments if _normalise(k) == _normalise(skill_name)),
        None,
    )
    if key is None:
        return None

    score = assessments[key]
    curve = config.ASSESSMENT_TRUST_CURVE  # list of (threshold, trust_value)

    trust = curve[0][1]  # default: lowest tier
    for threshold, trust_value in curve:
        if score >= threshold:
            trust = trust_value
    return trust


def _endorsement_score(endorsements: int) -> float:
    """
    Convert raw endorsement count to a 0–1 score (capped + normalised).

    Diminishing returns above _ENDORSEMENT_CAP.
    0 endorsements → 0.0
    10             → 0.20
    25             → 0.50
    50+            → 1.0
    """
    if not endorsements or endorsements <= 0:
        return 0.0
    return round(min(endorsements, _ENDORSEMENT_CAP) / _ENDORSEMENT_CAP, 4)


def _fuzzy_match(skill_name: str, keyword_set: set[str]) -> bool:
    """
    Return True if skill_name matches any keyword in keyword_set.
    Uses substring matching to handle variations:
      'Sentence Transformers' matches 'sentence transformers'
      'Pinecone DB' matches 'pinecone'
    """
    norm = _normalise(skill_name)
    if norm in keyword_set:
        return True
    # Substring: skill contains keyword OR keyword contains skill
    for kw in keyword_set:
        if kw in norm or norm in kw:
            return True
    return False


def _compute_skill_trust(
    skill: dict,
    assessments: dict[str, float],
) -> tuple[float, list[str]]:
    """
    Compute a per-skill trust multiplier (0.0–1.0) using:
      - Platform assessment score (if available)
      - Endorsements (social proof)
      - Duration vs proficiency sanity

    Returns (trust_multiplier, list_of_flag_strings).
    """
    name = skill.get("name", "")
    proficiency = _normalise(skill.get("proficiency", ""))
    duration = skill.get("duration_months") or 0
    endorsements = skill.get("endorsements", 0) or 0

    flags: list[str] = []

    # 1. Assessment trust (most reliable signal)
    assessment_trust = _assessment_trust(name, assessments)

    if assessment_trust is not None:
        # Assessment exists — use it as the primary trust signal
        trust = assessment_trust
        if assessment_trust < 0.40:
            flags.append(
                f"'{name}': assessment score below 40 — low trust on {proficiency} claim"
            )
    else:
        # No assessment — fall back to endorsements + duration sanity
        endorse_score = _endorsement_score(endorsements)
        dur_ok = duration >= config.ADVANCED_MIN_DURATION_MONTHS

        if proficiency == "expert":
            # Expert with no assessment: need strong endorsements + adequate duration
            if endorsements >= 20 and dur_ok:
                trust = 0.75
            elif endorsements >= 10 or dur_ok:
                trust = 0.55
            else:
                trust = 0.35
                flags.append(
                    f"'{name}': unverified expert claim, low endorsements ({endorsements}), "
                    f"short duration ({duration}mo)"
                )
        elif proficiency == "advanced":
            if endorsements >= 10 or dur_ok:
                trust = 0.70
            else:
                trust = 0.50
        else:
            # intermediate / beginner — lower bar, endorsements help
            trust = 0.60 + 0.20 * min(endorse_score, 1.0)

    return round(min(max(trust, 0.0), 1.0), 4), flags


# ─────────────────────────────────────────────────────────────────────────────
# Sub-scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_must_have(
    skills: list[dict],
    assessments: dict[str, float],
    result: SkillScoreResult,
) -> float:
    """
    Score coverage + quality of must-have JD skills.

    For each must-have skill found:
        raw_contribution = proficiency_weight × duration_score × trust_multiplier

    Final score = sum(contributions) / len(MUST_HAVE_SKILLS)
    Scaled so a perfect candidate (all must-haves, expert, long duration, verified)
    → 1.0.
    """
    # Build a lookup of candidate's skills: normalised_name → skill dict
    candidate_skills: dict[str, dict] = {}
    for s in skills:
        norm = _normalise(s.get("name", ""))
        # keep the highest proficiency if duplicate skill names
        if norm not in candidate_skills or (
            _proficiency_weight(s.get("proficiency", "")) >
            _proficiency_weight(candidate_skills[norm].get("proficiency", ""))
        ):
            candidate_skills[norm] = s

    contributions: list[float] = []
    matched: list[str] = []
    missing: list[str] = []

    for must_skill in config.MUST_HAVE_SKILLS:
        # Try exact match first, then fuzzy
        norm_must = _normalise(must_skill)
        matched_skill: dict | None = candidate_skills.get(norm_must)

        if matched_skill is None:
            # Fuzzy: check if any candidate skill name contains / is contained by
            for cname, cs in candidate_skills.items():
                if norm_must in cname or cname in norm_must:
                    matched_skill = cs
                    break

        if matched_skill is None:
            missing.append(must_skill)
            contributions.append(0.0)
            continue

        matched.append(must_skill)

        prof_w = _proficiency_weight(matched_skill.get("proficiency", ""))
        dur_s = _duration_score(matched_skill.get("duration_months"))
        trust, trust_flags = _compute_skill_trust(matched_skill, assessments)
        result.trust_flags.extend(trust_flags)

        # Combine: proficiency sets the ceiling, duration adds depth, trust verifies
        contribution = prof_w * (0.60 + 0.40 * dur_s) * trust
        contributions.append(contribution)

    result.must_have_matched = matched
    result.must_have_missing = missing

    if not contributions:
        return 0.0

    # Normalise: perfect would mean all must-haves matched at 1.0 contribution
    raw = sum(contributions) / len(config.MUST_HAVE_SKILLS)
    return round(min(raw, 1.0), 4)


def _score_retrieval_depth(
    skills: list[dict],
    assessments: dict[str, float],
    result: SkillScoreResult,
) -> float:
    """
    Score depth of retrieval/search/ranking-specific skills.

    This rewards candidates who have *deep, sustained* experience with
    embeddings, vector DBs, ranking systems — not just surface-level mentions.

    Metric: weighted average of (duration_score × trust) for retrieval-core
    skills the candidate holds, normalised by the number found.
    """
    retrieval_details: list[dict] = []

    for s in skills:
        name = s.get("name", "")
        if not _fuzzy_match(name, _RETRIEVAL_CORE):
            continue

        prof_w = _proficiency_weight(s.get("proficiency", ""))
        dur_s = _duration_score(s.get("duration_months"))
        trust, _ = _compute_skill_trust(s, assessments)

        depth = prof_w * dur_s * trust
        retrieval_details.append({
            "skill": name,
            "proficiency": s.get("proficiency"),
            "duration_months": s.get("duration_months"),
            "trust": trust,
            "depth_score": round(depth, 4),
        })

    result.retrieval_skills_detail = sorted(
        retrieval_details, key=lambda x: x["depth_score"], reverse=True
    )

    if not retrieval_details:
        return 0.0

    # Top-3 retrieval skills get full weight (breadth matters too)
    top_scores = sorted([d["depth_score"] for d in retrieval_details], reverse=True)[:3]

    # Breadth bonus: having 3+ core retrieval skills is better than 1 deep one
    breadth_factor = min(len(retrieval_details) / 3.0, 1.0)

    depth_avg = sum(top_scores) / 3.0  # always divide by 3 (missing = 0)
    final = depth_avg * (0.70 + 0.30 * breadth_factor)

    return round(min(final, 1.0), 4)


def _score_nice_to_have(
    skills: list[dict],
    assessments: dict[str, float],
    result: SkillScoreResult,
) -> float:
    """
    Bonus score for nice-to-have JD skills.

    Each nice-to-have contributes a smaller amount; the total is capped at 1.0.
    Diminishing returns: first 3 nice-to-haves count fully, rest at half weight.
    """
    matched: list[str] = []
    contributions: list[float] = []

    for s in skills:
        name = s.get("name", "")
        if not _fuzzy_match(name, _NICE_TO_HAVE):
            continue

        matched.append(name)
        prof_w = _proficiency_weight(s.get("proficiency", ""))
        trust, _ = _compute_skill_trust(s, assessments)
        contributions.append(prof_w * trust)

    result.nice_to_have_matched = matched

    if not contributions:
        return 0.0

    # Sort descending — best nice-to-haves count most
    contributions.sort(reverse=True)

    # Full weight for top-3, half weight for rest
    weighted = 0.0
    for i, c in enumerate(contributions):
        weight = 1.0 if i < 3 else 0.5
        weighted += c * weight

    # Normalise against a ceiling of ~4 full nice-to-have skills
    normalised = weighted / 4.0
    return round(min(normalised, 1.0), 4)


def _compute_trust_modifier(
    skills: list[dict],
    assessments: dict[str, float],
) -> tuple[float, list[str]]:
    """
    Compute an overall trust modifier for the candidate based on:
    - How many JD-relevant skills have verified assessment scores
    - Assessment score quality on those verified skills
    - Whether unverified high-proficiency claims are backed by duration/endorsements

    Returns (modifier: 0.0–1.0, flags: list[str])

    A modifier of 1.0 means "fully trusted".
    A modifier of 0.3 means "most claims are unverified and thin".
    """
    flags: list[str] = []
    jd_skills_found = [
        s for s in skills
        if _fuzzy_match(s.get("name", ""), _MUST_HAVE | _NICE_TO_HAVE | _RETRIEVAL_CORE)
    ]

    if not jd_skills_found:
        return 0.20, ["No JD-relevant skills found at all"]

    # Fraction of JD-relevant skills that have assessment scores
    assessed_count = sum(
        1 for s in jd_skills_found
        if _assessment_trust(s.get("name", ""), assessments) is not None
    )
    assessment_coverage = assessed_count / len(jd_skills_found)

    # Average trust across all JD-relevant skills
    trust_scores = []
    for s in jd_skills_found:
        t, f = _compute_skill_trust(s, assessments)
        trust_scores.append(t)
        flags.extend(f)

    avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

    # Keyword stuffer detection: high proficiency claims, near-zero duration average
    prof_claims = [s for s in jd_skills_found
                   if _normalise(s.get("proficiency", "")) in ("expert", "advanced")]
    if prof_claims:
        avg_duration = sum(s.get("duration_months") or 0 for s in prof_claims) / len(prof_claims)
        if avg_duration < 6 and len(prof_claims) >= 3:
            flags.append(
                f"Keyword stuffer signal: {len(prof_claims)} expert/advanced JD skills "
                f"with avg duration {avg_duration:.1f}mo — very thin evidence"
            )
            avg_trust *= 0.50   # hard penalty on modifier

    # Combine: assessment coverage boosts the modifier
    modifier = avg_trust * (0.70 + 0.30 * assessment_coverage)
    return round(min(max(modifier, 0.10), 1.0), 4), flags


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer class
# ─────────────────────────────────────────────────────────────────────────────

class SkillScorer:
    """
    Score a candidate's skill match against the Senior AI Engineer JD.

    Usage
    -----
        scorer = SkillScorer()
        result = scorer.score(candidate_dict)
        print(result.final_score)   # 0.0 – 1.0
    """

    def score(self, candidate: dict[str, Any]) -> SkillScoreResult:
        cid = candidate.get("candidate_id", "UNKNOWN")
        result = SkillScoreResult(candidate_id=cid)

        skills: list[dict] = candidate.get("skills", [])
        assessments: dict[str, float] = (
            candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
        )

        if not skills:
            result.trust_flags.append("No skills listed at all")
            return result  # score stays 0.0

        # ── 1. Must-have coverage ─────────────────────────────────────────────
        must_score = _score_must_have(skills, assessments, result)
        result.must_have_score = must_score

        # ── 2. Retrieval-core depth ───────────────────────────────────────────
        depth_score = _score_retrieval_depth(skills, assessments, result)
        result.retrieval_depth_score = depth_score

        # ── 3. Nice-to-have bonus ─────────────────────────────────────────────
        nice_score = _score_nice_to_have(skills, assessments, result)
        result.nice_to_have_score = nice_score

        # ── 4. Trust modifier ─────────────────────────────────────────────────
        trust_mod, trust_flags = _compute_trust_modifier(skills, assessments)
        result.trust_modifier = trust_mod
        result.trust_flags.extend(trust_flags)

        # ── 5. Weighted combination ───────────────────────────────────────────
        raw = (
            _W_MUST_HAVE      * must_score
            + _W_RETRIEVAL_DEPTH * depth_score
            + _W_NICE_TO_HAVE   * nice_score
            + _W_TRUST_PENALTY  * trust_mod
        )

        # Apply trust as a global multiplier for keyword-stuffer protection
        # Trust < 0.4 → score gets crushed even if keyword coverage is high
        trust_floor = 0.40
        if trust_mod < trust_floor:
            # Scale raw score down proportionally
            raw = raw * (trust_mod / trust_floor)

        result.final_score = round(min(max(raw, config.SCORE_MIN), config.SCORE_MAX), 6)
        return result

    def score_batch(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, SkillScoreResult]:
        """Score a list of candidates. Returns dict of candidate_id → result."""
        return {
            c.get("candidate_id", f"idx_{i}"): self.score(c)
            for i, c in enumerate(candidates)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (run: python scorers/skills.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pathlib

    sample_path = pathlib.Path("/mnt/user-data/uploads/sample_candidates.json")
    with open(sample_path) as f:
        candidates = json.load(f)

    scorer = SkillScorer()
    results = scorer.score_batch(candidates)

    # Sort by score descending
    ranked = sorted(results.values(), key=lambda r: r.final_score, reverse=True)

    print(f"{'Rank':<5} {'CID':<15} {'Score':>6}  {'Must':>5} {'Depth':>5} {'Nice':>5} {'Trust':>5}  Matched must-haves")
    print("─" * 110)
    for i, r in enumerate(ranked[:20], 1):
        matched_str = ", ".join(r.must_have_matched[:4])
        if len(r.must_have_matched) > 4:
            matched_str += f" (+{len(r.must_have_matched)-4})"
        print(
            f"{i:<5} {r.candidate_id:<15} {r.final_score:>6.3f}  "
            f"{r.must_have_score:>5.2f} {r.retrieval_depth_score:>5.2f} "
            f"{r.nice_to_have_score:>5.2f} {r.trust_modifier:>5.2f}  "
            f"{matched_str}"
        )

    print()
    print("=== TOP CANDIDATE DETAIL (rank 1) ===")
    top = ranked[0]
    print(f"Candidate   : {top.candidate_id}")
    print(f"Final score : {top.final_score:.4f}")
    print(f"Must-have matched  : {top.must_have_matched}")
    print(f"Must-have missing  : {top.must_have_missing}")
    print(f"Nice-to-have matched: {top.nice_to_have_matched}")
    print(f"Retrieval depth detail:")
    for d in top.retrieval_skills_detail:
        print(f"  {d['skill']:35} prof={d['proficiency']:12} dur={d['duration_months']}mo  trust={d['trust']:.2f}  depth={d['depth_score']:.3f}")
    if top.trust_flags:
        print(f"Trust flags : {top.trust_flags}")

    print()
    print("=== SCORE DISTRIBUTION ===")
    buckets = {"0.00–0.10": 0, "0.10–0.20": 0, "0.20–0.30": 0, "0.30–0.40": 0,
               "0.40–0.50": 0, "0.50–0.60": 0, "0.60–0.70": 0, "0.70+": 0}
    for r in results.values():
        s = r.final_score
        if s < 0.10:   buckets["0.00–0.10"] += 1
        elif s < 0.20: buckets["0.10–0.20"] += 1
        elif s < 0.30: buckets["0.20–0.30"] += 1
        elif s < 0.40: buckets["0.30–0.40"] += 1
        elif s < 0.50: buckets["0.40–0.50"] += 1
        elif s < 0.60: buckets["0.50–0.60"] += 1
        elif s < 0.70: buckets["0.60–0.70"] += 1
        else:          buckets["0.70+"] += 1
    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"  {bucket}: {count:3d}  {bar}")
