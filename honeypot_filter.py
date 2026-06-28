"""
honeypot_filter.py — Redrob Hackathon
======================================
Detects trap / honeypot candidates that should be excluded from the top-100.

The dataset contains ~80 honeypots with "subtly impossible profiles" plus
keyword-stuffers and behavioral twins.  A candidate accumulates a flag_count;
if it reaches HONEYPOT_SCORE_THRESHOLD (from config) the candidate is excluded
(final score forced to 0.0).

Each check is isolated in its own function so you can unit-test or disable
individual rules without touching the rest of the pipeline.

Usage
-----
    from honeypot_filter import HoneypotFilter
    hf = HoneypotFilter()
    result = hf.evaluate(candidate_dict)
    # result.is_honeypot  → bool
    # result.flags        → list[str]  (human-readable explanations)
    # result.flag_count   → int
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import config

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HoneypotResult:
    candidate_id: str
    flag_count: int = 0
    flags: list[str] = field(default_factory=list)

    @property
    def is_honeypot(self) -> bool:
        return self.flag_count >= config.HONEYPOT_SCORE_THRESHOLD

    def add_flag(self, reason: str, weight: int = 1) -> None:
        self.flags.append(reason)
        self.flag_count += weight

    def __repr__(self) -> str:
        status = "HONEYPOT" if self.is_honeypot else "ok"
        return f"<HoneypotResult {self.candidate_id} [{status}] flags={self.flag_count}>"


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> date | None:
    """Parse ISO date string (YYYY-MM-DD) to date object. Returns None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _months_between(start: date, end: date) -> int:
    """Approximate months between two dates."""
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def _normalize(text: str) -> str:
    return text.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# Individual check functions
# ─────────────────────────────────────────────────────────────────────────────

def check_timeline_vs_yoe(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 1 — Career duration months vs stated years_of_experience.

    A candidate claiming 10 YOE whose career_history only sums to 24 months is
    a honeypot.  We allow slack for gaps (freelance, self-employment, etc.).
    """
    yoe: float = candidate["profile"].get("years_of_experience", 0) or 0
    career: list[dict] = candidate.get("career_history", [])

    stated_months = yoe * 12
    actual_months = sum(ch.get("duration_months", 0) or 0 for ch in career)

    if stated_months == 0:
        return  # can't judge

    ratio = actual_months / stated_months if stated_months else 0

    # Career months WAY higher than stated YOE (e.g. 200 months career, 3 YOE stated)
    if ratio > config.TIMELINE_SLACK_RATIO:
        result.add_flag(
            f"Career months ({actual_months}) is {ratio:.1f}× stated YOE months "
            f"({stated_months:.0f}) — implausible inflation",
            weight=2,
        )

    # Career months WAY lower than stated YOE (e.g. 10 YOE but only 12 months total)
    elif actual_months > 0 and ratio < config.TIMELINE_FLOOR_RATIO:
        result.add_flag(
            f"Career months ({actual_months}) is only {ratio:.1%} of stated YOE months "
            f"({stated_months:.0f}) — suspiciously low",
            weight=2,
        )


def check_job_before_graduation(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 2 — Started a full-time job before graduating (beyond internship slack).

    A job starting 3+ years before the stated graduation year is impossible
    for a conventional career path.
    """
    edu_list: list[dict] = candidate.get("education", [])
    career: list[dict] = candidate.get("career_history", [])

    if not edu_list or not career:
        return

    # Use the latest graduation year as the reference
    grad_years = [e.get("end_year") for e in edu_list if e.get("end_year")]
    if not grad_years:
        return
    latest_grad = max(grad_years)

    for ch in career:
        start_date = _parse_date(ch.get("start_date"))
        if not start_date:
            continue
        start_year = start_date.year

        # Allow config.EDUCATION_JOB_OVERLAP_SLACK_YRS of overlap (internships)
        gap = latest_grad - start_year - config.EDUCATION_JOB_OVERLAP_SLACK_YRS
        if gap > 2:
            result.add_flag(
                f"Job at '{ch.get('company', '?')}' started {start_year}, but "
                f"graduation was {latest_grad} — {gap}yr gap is implausible",
                weight=2,
            )
            break  # one flag per candidate is enough for this check


def check_overlapping_jobs(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 3 — Two full-time roles that overlap by more than 3 months.

    Minor overlaps happen during transitions.  Major overlaps (>3 months of
    two concurrent full-time roles at different companies) are a data integrity
    red flag.
    """
    career: list[dict] = candidate.get("career_history", [])
    today = config.SIGNAL_REFERENCE_DATE

    intervals: list[tuple[date, date, str]] = []
    for ch in career:
        start = _parse_date(ch.get("start_date"))
        end_raw = ch.get("end_date")
        end = _parse_date(end_raw) if end_raw else today
        company = ch.get("company", "?")
        if start and end and start < end:
            intervals.append((start, end, company))

    intervals.sort(key=lambda x: x[0])

    for i in range(len(intervals) - 1):
        s1, e1, c1 = intervals[i]
        s2, e2, c2 = intervals[i + 1]
        if c1 == c2:
            continue  # same company (promotion / internal transfer) is fine
        overlap_months = _months_between(s2, min(e1, e2)) if s2 < e1 else 0
        if overlap_months > 3:
            result.add_flag(
                f"Overlapping full-time roles: '{c1}' and '{c2}' overlap "
                f"by {overlap_months} months",
                weight=1,
            )
            break


def check_future_dates(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 4 — Start/end dates in the future (beyond today).
    """
    today = config.SIGNAL_REFERENCE_DATE
    career: list[dict] = candidate.get("career_history", [])

    for ch in career:
        start = _parse_date(ch.get("start_date"))
        end = _parse_date(ch.get("end_date"))

        if start and start > today:
            result.add_flag(
                f"Job at '{ch.get('company', '?')}' has a future start date: {start}",
                weight=2,
            )
        if end and end > today:
            result.add_flag(
                f"Job at '{ch.get('company', '?')}' has a future end date: {end}",
                weight=1,
            )


def check_expert_skill_duration(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 5 — Claims 'expert' or 'advanced' proficiency in a skill with very
    little usage time (duration_months).

    Keyword stuffers often list 10 expert skills without any duration context,
    or with implausibly short durations.
    """
    skills: list[dict] = candidate.get("skills", [])

    for s in skills:
        proficiency = _normalize(s.get("proficiency", ""))
        duration = s.get("duration_months")  # can be None / missing
        name = s.get("name", "?")

        if duration is None:
            # Missing duration for expert skill is itself a mild signal
            # but not a hard flag — only add if multiple are missing
            continue

        if proficiency == "expert" and duration < config.EXPERT_MIN_DURATION_MONTHS:
            result.add_flag(
                f"Claims 'expert' in '{name}' but only {duration} months usage "
                f"(min expected: {config.EXPERT_MIN_DURATION_MONTHS}mo)",
                weight=1,
            )

        elif proficiency == "advanced" and duration < config.ADVANCED_MIN_DURATION_MONTHS:
            result.add_flag(
                f"Claims 'advanced' in '{name}' but only {duration} months usage "
                f"(min expected: {config.ADVANCED_MIN_DURATION_MONTHS}mo)",
                weight=1,
            )


def check_too_many_expert_skills(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 6 — Implausibly high number of 'expert' skills.

    Legitimate candidates typically have 1–5 expert-level skills.  Having 15+
    expert skills is a keyword-stuffer signal.
    """
    skills: list[dict] = candidate.get("skills", [])
    expert_count = sum(1 for s in skills if _normalize(s.get("proficiency", "")) == "expert")

    if expert_count > config.MAX_EXPERT_SKILLS:
        result.add_flag(
            f"{expert_count} skills claimed as 'expert' — implausibly high "
            f"(threshold: {config.MAX_EXPERT_SKILLS})",
            weight=1,
        )


def check_skill_vs_career_mismatch(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 7 — Skills list is full of AI/ML keywords but career history shows
    zero evidence of technical work (e.g. a Marketing Manager with 10 AI skills).

    This is the classic keyword-stuffer pattern.
    """
    skills: list[dict] = candidate.get("skills", [])
    career: list[dict] = candidate.get("career_history", [])
    current_title: str = _normalize(candidate["profile"].get("current_title", ""))

    # Count AI/ML skill claims
    ai_skill_names = {_normalize(s["name"]) for s in skills}
    ai_keywords = {_normalize(k) for k in config.MUST_HAVE_SKILLS + config.NICE_TO_HAVE_SKILLS}
    ai_skill_count = len(ai_skill_names & ai_keywords)

    if ai_skill_count < 3:
        return  # not enough AI skills to matter

    # Check if career descriptions mention any technical / AI work
    all_descriptions = " ".join(
        _normalize(ch.get("description", "")) for ch in career
    )
    technical_evidence = any(
        kw in all_descriptions
        for kw in ["model", "algorithm", "machine learning", "python", "data",
                   "neural", "training", "inference", "api", "deploy", "code",
                   "engineer", "develop", "implement", "build", "system"]
    )

    # Check for non-technical current title
    non_technical_titles = [
        "marketing", "sales", "operations", "accountant", "finance",
        "hr", "recruiter", "support", "manager", "analyst",
        "civil", "mechanical", "electrical", "procurement",
    ]
    is_non_technical = any(t in current_title for t in non_technical_titles)

    if ai_skill_count >= 5 and is_non_technical and not technical_evidence:
        result.add_flag(
            f"Lists {ai_skill_count} AI/ML skills but current title is "
            f"'{candidate['profile'].get('current_title', '?')}' with no "
            f"technical evidence in career descriptions — keyword stuffer",
            weight=2,
        )
    elif ai_skill_count >= 8 and not technical_evidence:
        result.add_flag(
            f"Lists {ai_skill_count} AI/ML skills but career history has no "
            f"technical keywords — likely keyword stuffer",
            weight=1,
        )


def check_salary_plausibility(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 8 — Salary range is implausible.

    Checks:
    - min > max (corrupted data)
    - Salary wildly below market for stated YOE (e.g. 1 LPA for 8 YOE)
    - Salary wildly above market (e.g. 500 LPA)
    - min/max ratio too extreme (e.g. min=1, max=200 — not a real expectation)
    """
    signals: dict = candidate.get("redrob_signals", {})
    sal: dict = signals.get("expected_salary_range_inr_lpa", {})
    sal_min: float = sal.get("min", 0) or 0
    sal_max: float = sal.get("max", 0) or 0
    yoe: float = candidate["profile"].get("years_of_experience", 0) or 0

    if sal_min <= 0 or sal_max <= 0:
        return  # missing data — don't penalise

    # min > max — corrupted
    if sal_min > sal_max:
        result.add_flag(
            f"Salary min ({sal_min} LPA) > max ({sal_max} LPA) — corrupted data",
            weight=2,
        )
        return

    # Implausibly low
    if sal_max < config.SALARY_MIN_PLAUSIBLE_LPA:
        result.add_flag(
            f"Max salary {sal_max} LPA is implausibly low for any professional",
            weight=1,
        )

    # Implausibly high
    if sal_min > config.SALARY_MAX_PLAUSIBLE_LPA:
        result.add_flag(
            f"Min salary {sal_min} LPA is implausibly high ({config.SALARY_MAX_PLAUSIBLE_LPA} LPA cap)",
            weight=1,
        )

    # Suspiciously low for their YOE
    if yoe >= 6 and sal_max < 10:
        result.add_flag(
            f"Max salary {sal_max} LPA is unrealistically low for {yoe:.1f} YOE",
            weight=1,
        )

    # Range too extreme (min/max ratio)
    if sal_max > 0:
        ratio = sal_min / sal_max
        if ratio < config.SALARY_MIN_RANGE_RATIO and sal_max > 30:
            result.add_flag(
                f"Salary range {sal_min}–{sal_max} LPA has suspicious spread "
                f"(ratio {ratio:.2f}) — possibly fabricated",
                weight=1,
            )


def check_education_year_sanity(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 9 — Education years are internally inconsistent.

    Checks:
    - end_year < start_year
    - Degree duration implausibly long (>10 years for a bachelor's)
    - Graduation in the future
    """
    edu_list: list[dict] = candidate.get("education", [])
    current_year = config.SIGNAL_REFERENCE_DATE.year

    for e in edu_list:
        start_yr = e.get("start_year")
        end_yr = e.get("end_year")
        degree = e.get("degree", "?")
        inst = e.get("institution", "?")

        if not start_yr or not end_yr:
            continue

        # End before start
        if end_yr < start_yr:
            result.add_flag(
                f"Education at '{inst}' ends ({end_yr}) before it starts ({start_yr})",
                weight=2,
            )

        # Implausibly long degree (>10 years)
        duration_yrs = end_yr - start_yr
        if duration_yrs > 10:
            result.add_flag(
                f"'{degree}' at '{inst}' spans {duration_yrs} years — implausible",
                weight=1,
            )

        # Graduation in the future (more than 1 year ahead — could be ongoing)
        if end_yr > current_year + 1:
            result.add_flag(
                f"Education at '{inst}' ends in {end_yr} — far future graduation",
                weight=1,
            )


def check_profile_completeness_vs_claims(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 10 — Very low profile completeness but rich skill/experience claims.

    A genuine 8-YOE AI engineer with 15 skills should have a complete profile.
    A completeness score of <30 with many expert-level claims is suspicious.
    """
    signals: dict = candidate.get("redrob_signals", {})
    completeness: float = signals.get("profile_completeness_score", 100) or 100
    skills: list[dict] = candidate.get("skills", [])
    yoe: float = candidate["profile"].get("years_of_experience", 0) or 0

    expert_or_advanced = sum(
        1 for s in skills
        if _normalize(s.get("proficiency", "")) in ("expert", "advanced")
    )

    if completeness < 30 and expert_or_advanced >= 5 and yoe >= 5:
        result.add_flag(
            f"Profile completeness is {completeness:.0f}% but claims "
            f"{expert_or_advanced} expert/advanced skills at {yoe:.1f} YOE — "
            f"inconsistent with a legitimate high-experience profile",
            weight=1,
        )


def check_assessment_vs_proficiency(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 11 — Platform assessment scores contradict claimed skill proficiency.

    If a candidate claims 'expert' in Python and scores 10/100 on the Python
    assessment, that's a strong stuffer signal.
    """
    signals: dict = candidate.get("redrob_signals", {})
    assessments: dict[str, float] = signals.get("skill_assessment_scores", {}) or {}
    skills: list[dict] = candidate.get("skills", [])

    if not assessments:
        return

    # Build a proficiency lookup: normalised skill name → proficiency
    claimed: dict[str, str] = {
        _normalize(s["name"]): _normalize(s.get("proficiency", ""))
        for s in skills
    }

    contradictions = 0
    contradiction_details: list[str] = []

    for skill_name, score in assessments.items():
        claimed_prof = claimed.get(_normalize(skill_name))
        if not claimed_prof:
            continue

        # Expert claiming <30 on assessment
        if claimed_prof == "expert" and score < 30:
            contradictions += 1
            contradiction_details.append(f"{skill_name}: claims expert, scored {score:.0f}/100")

        # Advanced claiming <20 on assessment
        elif claimed_prof == "advanced" and score < 20:
            contradictions += 1
            contradiction_details.append(f"{skill_name}: claims advanced, scored {score:.0f}/100")

    if contradictions >= 2:
        result.add_flag(
            f"Assessment scores contradict proficiency claims on {contradictions} skill(s): "
            + "; ".join(contradiction_details),
            weight=2,
        )
    elif contradictions == 1:
        result.add_flag(
            f"Assessment score contradicts proficiency claim: {contradiction_details[0]}",
            weight=1,
        )


def check_verification_signals(candidate: dict, result: HoneypotResult) -> None:
    """
    Flag 12 — Zero platform verifications combined with strong experience claims.

    Honeypot profiles often have all three verification flags set to False.
    A real 8-YOE candidate would typically have at least email verified.
    """
    signals: dict = candidate.get("redrob_signals", {})
    verified_email: bool = signals.get("verified_email", False)
    verified_phone: bool = signals.get("verified_phone", False)
    linkedin: bool = signals.get("linkedin_connected", False)
    yoe: float = candidate["profile"].get("years_of_experience", 0) or 0

    if not verified_email and not verified_phone and not linkedin and yoe >= 5:
        result.add_flag(
            f"No verifications (email/phone/LinkedIn all False) for a {yoe:.1f} YOE candidate "
            f"— likely fabricated profile",
            weight=1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main filter class
# ─────────────────────────────────────────────────────────────────────────────

class HoneypotFilter:
    """
    Run all honeypot checks on a candidate record and return a HoneypotResult.

    Example
    -------
        hf = HoneypotFilter()
        result = hf.evaluate(candidate)
        if result.is_honeypot:
            print(f"Excluded {result.candidate_id}: {result.flags}")
    """

    # Registry of all checks in priority order
    CHECKS = [
        check_timeline_vs_yoe,            # most reliable structural check
        check_job_before_graduation,       # hard impossible
        check_overlapping_jobs,            # hard impossible
        check_future_dates,               # hard impossible
        check_expert_skill_duration,       # skill depth mismatch
        check_too_many_expert_skills,      # keyword stuffer
        check_skill_vs_career_mismatch,    # stuffer pattern
        check_salary_plausibility,         # data sanity
        check_education_year_sanity,       # data sanity
        check_profile_completeness_vs_claims,  # consistency
        check_assessment_vs_proficiency,   # platform trust signal
        check_verification_signals,        # identity sanity
    ]

    def evaluate(self, candidate: dict[str, Any]) -> HoneypotResult:
        """Run all checks and return a HoneypotResult."""
        cid = candidate.get("candidate_id", "UNKNOWN")
        result = HoneypotResult(candidate_id=cid)

        for check_fn in self.CHECKS:
            try:
                check_fn(candidate, result)
            except Exception as exc:  # never let a buggy check crash the pipeline
                result.flags.append(f"[check error in {check_fn.__name__}]: {exc}")

        return result

    def is_honeypot(self, candidate: dict[str, Any]) -> bool:
        """Convenience shortcut — returns True/False only."""
        return self.evaluate(candidate).is_honeypot

    def batch_evaluate(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, HoneypotResult]:
        """
        Evaluate a batch of candidates.
        Returns dict of candidate_id → HoneypotResult.
        """
        return {
            c.get("candidate_id", f"idx_{i}"): self.evaluate(c)
            for i, c in enumerate(candidates)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run: python honeypot_filter.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import pathlib

    sample_path = pathlib.Path("../uploads/sample_candidates.json")
    if not sample_path.exists():
        sample_path = pathlib.Path("/mnt/user-data/uploads/sample_candidates.json")

    with open(sample_path) as f:
        candidates = json.load(f)

    hf = HoneypotFilter()
    honeypots = []
    clean = []

    for c in candidates:
        res = hf.evaluate(c)
        if res.is_honeypot:
            honeypots.append(res)
        else:
            clean.append(res)

    print(f"Total candidates : {len(candidates)}")
    print(f"Honeypots flagged: {len(honeypots)}  ({len(honeypots)/len(candidates)*100:.1f}%)")
    print(f"Clean candidates : {len(clean)}")
    print()

    if honeypots:
        print("=== FLAGGED CANDIDATES ===")
        for r in honeypots:
            print(f"\n{r.candidate_id}  [flags={r.flag_count}]")
            for flag in r.flags:
                print(f"  • {flag}")

    print("\n=== FLAG DISTRIBUTION (all candidates) ===")
    from collections import Counter
    flag_counts = Counter(hf.evaluate(c).flag_count for c in candidates)
    for count in sorted(flag_counts):
        bar = "█" * flag_counts[count]
        print(f"  {count} flags: {flag_counts[count]:3d} candidates  {bar}")
