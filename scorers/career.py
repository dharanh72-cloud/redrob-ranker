"""
scorers/career.py — Redrob Hackathon
======================================
Scores the substance of a candidate's career history against the JD for
Senior AI Engineer (retrieval / ranking / search systems).

Four sub-components
-------------------
1. Company type      (35%) — product companies vs pure consulting
2. Retrieval systems (35%) — evidence of shipped search/ranking/recommendation
3. Title fit         (20%) — current + historical titles alignment
4. Career trajectory (10%) — tenure stability, no pure management drift

Score range: 0.0 – 1.0

Design principles
-----------------
- Read descriptions, not just titles. A "Software Engineer" who shipped a
  ranking system beats a "Senior ML Engineer" at a services firm who did
  PowerPoint decks.
- Consulting history is not an automatic disqualifier — only a pure consulting
  career (every single role) is capped. One stint at TCS amid product-company
  experience is fine.
- The JD explicitly warns about "title-chasers" — frequent short stints for
  promotions. We penalise average tenure < 18 months.
- Recent experience matters more than old experience. Jobs in the last 3 years
  get higher description analysis weight.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Sub-component weights (must sum to 1.0)
_W_COMPANY     = 0.35
_W_RETRIEVAL   = 0.35
_W_TITLE       = 0.20
_W_TRAJECTORY  = 0.10

assert abs(_W_COMPANY + _W_RETRIEVAL + _W_TITLE + _W_TRAJECTORY - 1.0) < 1e-9

# Normalised sets from config
_CONSULTING_FIRMS  = {f.lower() for f in config.CONSULTING_FIRMS}
_CONSULTING_IND    = {i.lower() for i in config.CONSULTING_INDUSTRIES}
_PRODUCT_IND       = {i.lower() for i in config.PRODUCT_INDUSTRIES}
_RETRIEVAL_KW      = {k.lower() for k in config.RETRIEVAL_SYSTEM_KEYWORDS}
_PRODUCTION_KW     = {k.lower() for k in config.PRODUCTION_KEYWORDS}
_POS_TITLES        = {t.lower() for t in config.POSITIVE_TITLE_KEYWORDS}
_NEG_TITLES        = {t.lower() for t in config.NEGATIVE_TITLE_KEYWORDS}

# Reference date for recency calculations
_TODAY = config.SIGNAL_REFERENCE_DATE

# Known product companies that appear in Indian tech ecosystem (extend as needed)
_KNOWN_PRODUCT_COS: set[str] = {
    # Indian unicorns / product cos
    "swiggy", "zomato", "ola", "uber", "flipkart", "myntra", "amazon",
    "meesho", "razorpay", "cred", "phonepe", "paytm", "groww", "zerodha",
    "freshworks", "zoho", "browserstack", "postman", "chargebee",
    "mad street den", "uniphore", "observe.ai", "sarvam", "krutrim",
    "sarvam ai", "glean", "darwinbox", "springworks", "leadsquared",
    "cleartax", "legaldesk", "slice", "niyo", "jupiter", "fi money",
    "dunzo", "blinkit", "zepto", "nykaa", "boat", "noise",
    # Global product cos with India offices
    "google", "microsoft", "meta", "apple", "netflix", "linkedin",
    "salesforce", "adobe", "atlassian", "stripe", "twilio",
    "nvidia", "qualcomm", "samsung", "sony",
    # Common fictional product-cos in synthetic datasets
    "pied piper", "hooli", "initech", "globex", "stark industries",
    "wayne enterprises",
}

# Companies that are clearly non-product (add to consulting check)
_EXTRA_CONSULTING: set[str] = {
    "dunder mifflin", "acme corp",  # fictional non-tech placeholders
}


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CareerScoreResult:
    candidate_id: str
    final_score: float = 0.0

    company_score: float = 0.0
    retrieval_score: float = 0.0
    title_score: float = 0.0
    trajectory_score: float = 0.0

    is_pure_consulting: bool = False
    product_months: int = 0
    consulting_months: int = 0
    retrieval_evidence: list[str] = field(default_factory=list)
    production_evidence: list[str] = field(default_factory=list)
    title_signals: list[str] = field(default_factory=list)
    trajectory_flags: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<CareerScore {self.candidate_id} "
            f"final={self.final_score:.3f} "
            f"co={self.company_score:.2f} "
            f"ret={self.retrieval_score:.2f} "
            f"title={self.title_score:.2f} "
            f"traj={self.trajectory_score:.2f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return text.lower().strip()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _months_since(d: date) -> int:
    """Months between a past date and today."""
    return max(0, (_TODAY.year - d.year) * 12 + (_TODAY.month - d.month))


def _recency_weight(start_date_str: str | None) -> float:
    """
    Weight a career role by how recent it is.
    Last 3 years  → 1.0
    3–6 years ago → 0.70
    6–10 years ago→ 0.45
    10+ years ago → 0.25
    """
    d = _parse_date(start_date_str)
    if not d:
        return 0.50  # unknown — neutral
    months_ago = _months_since(d)
    if months_ago <= 36:
        return 1.00
    elif months_ago <= 72:
        return 0.70
    elif months_ago <= 120:
        return 0.45
    else:
        return 0.25


def _is_consulting_company(company: str, industry: str) -> bool:
    """Return True if this company/industry looks like a consulting / services firm."""
    co = _norm(company)
    ind = _norm(industry)
    if co in _CONSULTING_FIRMS:
        return True
    if co in _EXTRA_CONSULTING:
        return True
    if ind in _CONSULTING_IND:
        return True
    return False


def _is_product_company(company: str, industry: str, size: str) -> bool:
    """Return True if this role looks like a genuine product company."""
    co = _norm(company)
    ind = _norm(industry)
    if co in _KNOWN_PRODUCT_COS:
        return True
    if ind in _PRODUCT_IND:
        return True
    # Mid-size companies (11–5000 employees) in tech are often product cos
    product_sizes = {"11-50", "51-200", "201-500", "501-1000", "1001-5000"}
    if size in product_sizes and ind not in _CONSULTING_IND:
        return True
    return False


def _count_retrieval_keywords(text: str) -> tuple[int, int, list[str]]:
    """
    Count how many retrieval and production keywords appear in a description.
    Returns (retrieval_count, production_count, matched_keywords).
    """
    t = _norm(text)
    ret_matches = [kw for kw in _RETRIEVAL_KW if kw in t]
    prod_matches = [kw for kw in _PRODUCTION_KW if kw in t]
    return len(ret_matches), len(prod_matches), ret_matches + prod_matches


def _description_retrieval_score(description: str) -> float:
    """
    Score a single role description for evidence of retrieval/ranking work.

    0   retrieval keywords → 0.0
    1–2 keywords          → 0.25
    3–4 keywords          → 0.55
    5–6 keywords          → 0.75
    7+  keywords + production evidence → 1.0
    """
    ret_count, prod_count, _ = _count_retrieval_keywords(description)
    if ret_count == 0:
        return 0.0
    base = min(ret_count / 7.0, 1.0)          # linear up to 7 keywords
    prod_boost = min(prod_count * 0.05, 0.20)  # production evidence adds up to 0.20
    return round(min(base + prod_boost, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_company_type(
    career: list[dict],
    result: CareerScoreResult,
) -> float:
    """
    Score the proportion of career spent at product vs consulting companies,
    weighted by recency and duration.

    Pure consulting (every role is consulting/services) → score capped at 0.15
    Mixed (some product, some consulting)              → proportional
    Fully product-company background                  → 1.0
    """
    if not career:
        return 0.0

    product_weighted   = 0.0
    consulting_weighted = 0.0
    total_weighted     = 0.0

    for ch in career:
        duration  = ch.get("duration_months") or 0
        company   = ch.get("company", "")
        industry  = ch.get("industry", "")
        size      = ch.get("company_size", "")
        recency   = _recency_weight(ch.get("start_date"))
        weight    = duration * recency

        total_weighted += weight

        if _is_consulting_company(company, industry):
            consulting_weighted += weight
            result.consulting_months += duration
        elif _is_product_company(company, industry, size):
            product_weighted += weight
            result.product_months += duration
        else:
            # Unknown — split half/half (benefit of doubt)
            product_weighted   += weight * 0.5
            consulting_weighted += weight * 0.5

    if total_weighted == 0:
        return 0.0

    product_ratio = product_weighted / total_weighted
    consulting_ratio = consulting_weighted / total_weighted

    # Pure consulting: entire career at services firms
    if consulting_ratio >= 0.90:
        result.is_pure_consulting = True
        return 0.10  # hard cap

    # Mostly consulting (>60%): significant penalty
    if consulting_ratio > 0.60:
        return round(0.10 + 0.30 * product_ratio, 4)

    # Mixed or mostly product
    score = 0.40 + 0.60 * product_ratio
    return round(min(score, 1.0), 4)


def _score_retrieval_systems(
    career: list[dict],
    result: CareerScoreResult,
) -> float:
    """
    Score evidence of having shipped retrieval / ranking / search systems.

    Reads every role description and checks for retrieval + production keywords.
    Recent roles get more weight. The best single role caps at 0.80 —
    having evidence across multiple roles pushes toward 1.0.
    """
    if not career:
        return 0.0

    role_scores: list[float] = []
    best_evidence: list[str] = []

    for ch in career:
        desc      = ch.get("description", "") or ""
        recency   = _recency_weight(ch.get("start_date"))
        title     = _norm(ch.get("title", ""))

        # Base description score
        desc_score = _description_retrieval_score(desc)

        # Title bonus: NLP/Search/ML engineer titles signal the right kind of work
        title_bonus = 0.0
        for pos in _POS_TITLES:
            if pos in title:
                title_bonus = 0.15
                break

        # Recency-weighted role score
        role_score = min((desc_score + title_bonus) * recency, 1.0)
        role_scores.append(role_score)

        # Collect evidence for diagnostics
        _, _, matched_kws = _count_retrieval_keywords(desc)
        if matched_kws:
            snippet = (
                f"{ch.get('title', '?')} @ {ch.get('company', '?')} "
                f"({ch.get('duration_months', 0)}mo): {', '.join(matched_kws[:5])}"
            )
            # Split into retrieval vs production evidence
            ret_kws = [k for k in matched_kws if k in _RETRIEVAL_KW]
            prod_kws = [k for k in matched_kws if k in _PRODUCTION_KW]
            if ret_kws:
                result.retrieval_evidence.append(snippet)
            if prod_kws:
                result.production_evidence.append(
                    f"{ch.get('company', '?')}: {', '.join(prod_kws[:3])}"
                )
            if role_score > 0.40:
                best_evidence.extend(ret_kws[:3])

    if not role_scores:
        return 0.0

    role_scores_sorted = sorted(role_scores, reverse=True)

    # Best single role gets base credit (max 0.80)
    best = min(role_scores_sorted[0], 0.80)

    # Second and third roles provide diminishing bonus (breadth shows pattern, not fluke)
    breadth_bonus = 0.0
    if len(role_scores_sorted) >= 2:
        breadth_bonus += role_scores_sorted[1] * 0.15
    if len(role_scores_sorted) >= 3:
        breadth_bonus += role_scores_sorted[2] * 0.08

    return round(min(best + breadth_bonus, 1.0), 4)


def _score_title_fit(
    profile: dict,
    career: list[dict],
    result: CareerScoreResult,
) -> float:
    """
    Score how well current and historical titles align with the JD role.

    Checks:
    - Current title: is it an AI/ML/Search/NLP engineering role?
    - Career arc: has the candidate trended toward technical AI roles over time?
    - Negative titles: entirely non-technical history with no AI pivot?
    """
    current_title = _norm(profile.get("current_title", ""))

    signals: list[str] = []
    score = 0.0

    # ── Current title ─────────────────────────────────────────────────────────
    current_positive = any(pos in current_title for pos in _POS_TITLES)
    current_negative = any(neg in current_title for neg in _NEG_TITLES)

    if current_positive:
        score += 0.50
        signals.append(f"Current title '{profile.get('current_title')}' is a strong fit")
    elif current_negative:
        score += 0.00
        signals.append(f"Current title '{profile.get('current_title')}' is a mismatch")
    else:
        # Neutral title (Backend Engineer, Software Engineer, etc.)
        score += 0.20
        signals.append(f"Current title '{profile.get('current_title')}' is neutral")

    # ── Career arc: titles over time ─────────────────────────────────────────
    positive_title_months = 0
    negative_title_months = 0
    total_months = 0

    for ch in career:
        t = _norm(ch.get("title", ""))
        dur = ch.get("duration_months") or 0
        total_months += dur

        if any(pos in t for pos in _POS_TITLES):
            positive_title_months += dur
        elif any(neg in t for neg in _NEG_TITLES):
            negative_title_months += dur

    if total_months > 0:
        positive_ratio = positive_title_months / total_months
        negative_ratio = negative_title_months / total_months

        if positive_ratio >= 0.50:
            score += 0.40
            signals.append(
                f"{positive_ratio:.0%} of career in AI/ML/Search engineering titles"
            )
        elif positive_ratio >= 0.25:
            score += 0.25
            signals.append(
                f"{positive_ratio:.0%} of career in relevant titles (partial fit)"
            )
        elif negative_ratio >= 0.70:
            score += 0.00
            signals.append(
                f"{negative_ratio:.0%} of career in non-technical titles"
            )
        else:
            score += 0.10
            signals.append("Career mix of neutral and non-technical titles")

    # ── Trajectory: most recent role should trend up toward AI ───────────────
    # If the last 2 roles are better titles than earlier ones, that's a positive arc
    if len(career) >= 2:
        recent_two = sorted(career, key=lambda c: c.get("start_date") or "", reverse=True)[:2]
        recent_positive = sum(
            1 for ch in recent_two
            if any(pos in _norm(ch.get("title", "")) for pos in _POS_TITLES)
        )
        if recent_positive >= 1:
            score += 0.10
            signals.append("Recent roles show AI/ML trajectory")

    result.title_signals = signals
    return round(min(score, 1.0), 4)


def _score_trajectory(
    career: list[dict],
    result: CareerScoreResult,
) -> float:
    """
    Score career trajectory quality: tenure stability, no management drift,
    no title-chasing (short stints at many companies for promotions).

    Returns a 0–1 score. Starts at 1.0 and applies penalties.
    """
    if not career:
        return 0.50  # no data — neutral

    score = 1.0
    flags: list[str] = []

    # ── Average tenure ────────────────────────────────────────────────────────
    durations = [ch.get("duration_months") or 0 for ch in career]
    avg_tenure = sum(durations) / len(durations) if durations else 0

    if avg_tenure < config.MIN_AVG_TENURE_MONTHS:
        penalty = min((config.MIN_AVG_TENURE_MONTHS - avg_tenure) / config.MIN_AVG_TENURE_MONTHS, 0.40)
        score -= penalty
        flags.append(
            f"Avg tenure {avg_tenure:.0f}mo < {config.MIN_AVG_TENURE_MONTHS}mo threshold "
            f"— possible title-chaser (penalty: {penalty:.2f})"
        )

    # ── Job switch frequency in last 5 years ─────────────────────────────────
    cutoff = date(_TODAY.year - 5, _TODAY.month, _TODAY.day)
    recent_jobs = [
        ch for ch in career
        if _parse_date(ch.get("start_date")) and _parse_date(ch.get("start_date")) >= cutoff  # type: ignore[operator]
    ]
    if len(recent_jobs) > config.MAX_JOB_SWITCHES_5YRS:
        excess = len(recent_jobs) - config.MAX_JOB_SWITCHES_5YRS
        penalty = min(excess * 0.08, 0.25)
        score -= penalty
        flags.append(
            f"{len(recent_jobs)} jobs in last 5 years (threshold: {config.MAX_JOB_SWITCHES_5YRS}) "
            f"— job hopper signal (penalty: {penalty:.2f})"
        )

    # ── Pure management drift ─────────────────────────────────────────────────
    # If the most recent N months are all management titles with no coding evidence
    recent_sorted = sorted(career, key=lambda c: c.get("start_date") or "", reverse=True)
    management_titles = {"manager", "director", "vp ", "head of", "chief", "cto", "ceo"}
    coding_titles     = {"engineer", "scientist", "developer", "architect", "researcher", "analyst"}

    recent_mgmt_months = 0
    for ch in recent_sorted:
        t = _norm(ch.get("title", ""))
        is_mgmt   = any(m in t for m in management_titles)
        is_coding = any(c in t for c in coding_titles)
        dur = ch.get("duration_months") or 0

        if is_mgmt and not is_coding:
            recent_mgmt_months += dur
        else:
            break  # stop at first non-management role from the top

    if recent_mgmt_months > config.MAX_MANAGEMENT_GAP_MONTHS:
        penalty = min((recent_mgmt_months - config.MAX_MANAGEMENT_GAP_MONTHS) / 36, 0.30)
        score -= penalty
        flags.append(
            f"Most recent {recent_mgmt_months}mo in pure management (no coding evidence) "
            f"— role requires hands-on code (penalty: {penalty:.2f})"
        )

    result.trajectory_flags = flags
    return round(max(min(score, 1.0), 0.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer class
# ─────────────────────────────────────────────────────────────────────────────

class CareerScorer:
    """
    Score a candidate's career history against the Senior AI Engineer JD.

    Usage
    -----
        scorer = CareerScorer()
        result = scorer.score(candidate_dict)
        print(result.final_score)   # 0.0 – 1.0
    """

    def score(self, candidate: dict[str, Any]) -> CareerScoreResult:
        cid     = candidate.get("candidate_id", "UNKNOWN")
        profile = candidate.get("profile", {})
        career  = candidate.get("career_history", [])

        result = CareerScoreResult(candidate_id=cid)

        if not career:
            result.trajectory_flags.append("No career history found")
            return result

        # ── 1. Company type ───────────────────────────────────────────────────
        co_score = _score_company_type(career, result)
        result.company_score = co_score

        # ── 2. Retrieval systems evidence ─────────────────────────────────────
        ret_score = _score_retrieval_systems(career, result)
        result.retrieval_score = ret_score

        # ── 3. Title fit ──────────────────────────────────────────────────────
        title_score = _score_title_fit(profile, career, result)
        result.title_score = title_score

        # ── 4. Career trajectory ──────────────────────────────────────────────
        traj_score = _score_trajectory(career, result)
        result.trajectory_score = traj_score

        # ── 5. Weighted combination ───────────────────────────────────────────
        raw = (
            _W_COMPANY    * co_score
            + _W_RETRIEVAL  * ret_score
            + _W_TITLE      * title_score
            + _W_TRAJECTORY * traj_score
        )

        # Hard cap for pure consulting careers (regardless of other scores)
        if result.is_pure_consulting:
            raw = min(raw, config.DISQUALIFIED_CAP + 0.10)

        result.final_score = round(
            min(max(raw, config.SCORE_MIN), config.SCORE_MAX), 6
        )
        return result

    def score_batch(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, CareerScoreResult]:
        return {
            c.get("candidate_id", f"idx_{i}"): self.score(c)
            for i, c in enumerate(candidates)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (run: python scorers/career.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pathlib

    sample_path = pathlib.Path("/mnt/user-data/uploads/sample_candidates.json")
    with open(sample_path) as f:
        candidates = json.load(f)

    scorer = CareerScorer()
    results = scorer.score_batch(candidates)
    ranked  = sorted(results.values(), key=lambda r: r.final_score, reverse=True)

    print(f"{'Rank':<5} {'CID':<15} {'Score':>6}  {'Co':>5} {'Ret':>5} {'Title':>6} {'Traj':>5}  {'ConsultMo':>9} {'ProdMo':>6}  Title")
    print("─" * 120)
    for i, r in enumerate(ranked[:20], 1):
        cand = next(c for c in candidates if c["candidate_id"] == r.candidate_id)
        title = cand["profile"]["current_title"]
        pure  = " ⛔CONSULT" if r.is_pure_consulting else ""
        print(
            f"{i:<5} {r.candidate_id:<15} {r.final_score:>6.3f}  "
            f"{r.company_score:>5.2f} {r.retrieval_score:>5.2f} "
            f"{r.title_score:>6.2f} {r.trajectory_score:>5.2f}  "
            f"{r.consulting_months:>9} {r.product_months:>6}  "
            f"{title}{pure}"
        )

    print()
    print("=== TOP CANDIDATE DETAIL ===")
    top = ranked[0]
    print(f"Candidate         : {top.candidate_id}")
    print(f"Final score       : {top.final_score:.4f}")
    print(f"Product months    : {top.product_months}")
    print(f"Consulting months : {top.consulting_months}")
    print(f"Pure consulting?  : {top.is_pure_consulting}")
    print(f"Title signals     : {top.title_signals}")
    print(f"Retrieval evidence:")
    for e in top.retrieval_evidence[:4]:
        print(f"  • {e}")
    print(f"Production evidence:")
    for e in top.production_evidence[:3]:
        print(f"  • {e}")
    if top.trajectory_flags:
        print(f"Trajectory flags  : {top.trajectory_flags}")

    print()
    print("=== PURE CONSULTING CANDIDATES (capped) ===")
    pure = [r for r in results.values() if r.is_pure_consulting]
    print(f"Found {len(pure)} pure-consulting candidates in sample of {len(candidates)}")
    for r in pure[:5]:
        cand = next(c for c in candidates if c["candidate_id"] == r.candidate_id)
        print(f"  {r.candidate_id} | score={r.final_score:.3f} | {cand['profile']['current_title']}")

    print()
    print("=== SCORE DISTRIBUTION ===")
    buckets = {"0.00–0.10": 0, "0.10–0.20": 0, "0.20–0.30": 0, "0.30–0.40": 0,
               "0.40–0.50": 0, "0.50–0.60": 0, "0.60–0.70": 0, "0.70+": 0}
    for r in results.values():
        s = r.final_score
        if   s < 0.10: buckets["0.00–0.10"] += 1
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
