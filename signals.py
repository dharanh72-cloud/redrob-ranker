"""
signals.py — Redrob Hackathon
================================
Converts the 23 redrob_signals fields into a single availability multiplier
(0.0 – 1.0) that scales a candidate's base score.

The multiplier is NOT a standalone score — it's applied on top of the
skills + career + experience + location scores in rank.py.

  final_score = base_score × signal_multiplier

This means a brilliant candidate (high base) who is functionally unavailable
(low multiplier) gets correctly down-ranked. A mediocre candidate with great
availability still can't bubble up past strong-base candidates.

Four sub-signals
-----------------
1. Availability  (40%) — last_active_date, open_to_work_flag, applications_30d
2. Responsiveness(25%) — recruiter_response_rate, avg_response_time_hours,
                         interview_completion_rate
3. Notice period (25%) — notice_period_days vs JD preference (≤30 days)
4. Platform trust(10%) — github_activity_score, verified_email/phone/linkedin,
                         profile_completeness_score

Multiplier range: 0.05 – 1.10
  - Floor of 0.05 (not 0): dead candidates get severely penalised but not
    completely zeroed (honeypot_filter handles outright exclusions)
  - Ceiling of 1.10: very strong signals get a small boost above 1.0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

_TODAY = config.SIGNAL_REFERENCE_DATE
_MULTIPLIER_FLOOR   = 0.05
_MULTIPLIER_CEILING = 1.10

# Sub-component weights (must sum to 1.0)
_W_AVAILABILITY   = 0.40
_W_RESPONSIVENESS = 0.25
_W_NOTICE         = 0.25
_W_TRUST          = 0.10

assert abs(_W_AVAILABILITY + _W_RESPONSIVENESS + _W_NOTICE + _W_TRUST - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    candidate_id: str
    multiplier: float = 1.0           # the value applied to base score

    availability_score: float = 0.0
    responsiveness_score: float = 0.0
    notice_score: float = 0.0
    trust_score: float = 0.0

    days_since_active: int = 0
    notice_period_days: int = 0
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<SignalResult {self.candidate_id} "
            f"mult={self.multiplier:.3f} "
            f"avail={self.availability_score:.2f} "
            f"resp={self.responsiveness_score:.2f} "
            f"notice={self.notice_score:.2f} "
            f"trust={self.trust_score:.2f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _days_since(d: date) -> int:
    return max(0, (_TODAY - d).days)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ─────────────────────────────────────────────────────────────────────────────
# Sub-scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_availability(sig: dict, result: SignalResult) -> float:
    """
    How available is this candidate right now?

    Inputs: last_active_date, open_to_work_flag, applications_submitted_30d
    """
    score = 0.0

    # ── last_active_date (core of availability) ───────────────────────────
    last_active = _parse_date(sig.get("last_active_date"))
    if last_active:
        days = _days_since(last_active)
        result.days_since_active = days

        if days <= config.ACTIVE_DAYS_EXCELLENT:      # ≤14 days
            active_score = 1.00
        elif days <= config.ACTIVE_DAYS_GOOD:         # ≤30 days
            active_score = 0.85
        elif days <= config.ACTIVE_DAYS_ACCEPTABLE:   # ≤60 days
            active_score = 0.65
        elif days <= config.ACTIVE_DAYS_STALE:        # ≤90 days
            active_score = 0.40
        elif days <= config.ACTIVE_DAYS_DEAD:         # ≤180 days
            active_score = 0.20
        else:                                          # >180 days
            active_score = 0.05
            result.notes.append(
                f"Last active {days} days ago — functionally unavailable"
            )
    else:
        active_score = 0.30   # unknown — neutral-low
        result.notes.append("No last_active_date — availability unknown")

    score += active_score * 0.65   # last_active is 65% of availability

    # ── open_to_work_flag (20% of availability) ───────────────────────────
    otw = sig.get("open_to_work_flag", False)
    score += (0.20 if otw else 0.05)

    # ── applications_submitted_30d (15% of availability) ──────────────────
    # Actively applying = keen to move
    apps = sig.get("applications_submitted_30d", 0) or 0
    if apps >= 5:
        app_score = 1.0
    elif apps >= 2:
        app_score = 0.65
    elif apps == 1:
        app_score = 0.40
    else:
        app_score = 0.10
    score += app_score * 0.15

    return round(_clamp(score, 0.0, 1.0), 4)


def _score_responsiveness(sig: dict, result: SignalResult) -> float:
    """
    Will this candidate actually respond and show up?

    Inputs: recruiter_response_rate, avg_response_time_hours,
            interview_completion_rate
    """
    # ── recruiter_response_rate (50% of responsiveness) ───────────────────
    rr = sig.get("recruiter_response_rate", 0.0) or 0.0
    if rr >= config.RESPONSE_RATE_EXCELLENT:     # ≥0.70
        rr_score = 1.00
    elif rr >= config.RESPONSE_RATE_GOOD:        # ≥0.40
        rr_score = 0.70 + 0.30 * (rr - config.RESPONSE_RATE_GOOD) / (
            config.RESPONSE_RATE_EXCELLENT - config.RESPONSE_RATE_GOOD
        )
    elif rr >= config.RESPONSE_RATE_LOW:         # ≥0.20
        rr_score = 0.35 + 0.35 * (rr - config.RESPONSE_RATE_LOW) / (
            config.RESPONSE_RATE_GOOD - config.RESPONSE_RATE_LOW
        )
    elif rr >= config.RESPONSE_RATE_GHOST:       # ≥0.10
        rr_score = 0.10
        result.notes.append(
            f"Very low recruiter response rate ({rr:.0%}) — likely ghost candidate"
        )
    else:
        rr_score = 0.05
        result.notes.append(
            f"Near-zero response rate ({rr:.0%}) — treat as unavailable"
        )

    # ── avg_response_time_hours (25% of responsiveness) ───────────────────
    rt = sig.get("avg_response_time_hours", 48.0) or 48.0
    if rt <= config.RESPONSE_TIME_FAST_H:        # ≤12h
        rt_score = 1.00
    elif rt <= config.RESPONSE_TIME_OK_H:        # ≤48h
        rt_score = 0.75
    elif rt <= config.RESPONSE_TIME_SLOW_H:      # ≤96h
        rt_score = 0.45
    else:                                         # >96h
        rt_score = 0.20
        if rr < config.RESPONSE_RATE_LOW:
            result.notes.append(
                f"Slow response ({rt:.0f}h) + low response rate — ghost signal"
            )

    # ── interview_completion_rate (25% of responsiveness) ─────────────────
    icr = sig.get("interview_completion_rate", 0.5) or 0.5
    if icr >= config.INTERVIEW_RATE_GOOD:        # ≥0.80
        icr_score = 1.00
    elif icr >= config.INTERVIEW_RATE_LOW:       # ≥0.50
        icr_score = 0.55 + 0.45 * (icr - config.INTERVIEW_RATE_LOW) / (
            config.INTERVIEW_RATE_GOOD - config.INTERVIEW_RATE_LOW
        )
    elif icr >= config.INTERVIEW_RATE_BAD:       # ≥0.30
        icr_score = 0.20
        result.notes.append(
            f"Low interview completion rate ({icr:.0%}) — often drops out"
        )
    else:
        icr_score = 0.05
        result.notes.append(
            f"Very low interview completion ({icr:.0%}) — unreliable"
        )

    score = 0.50 * rr_score + 0.25 * rt_score + 0.25 * icr_score
    return round(_clamp(score, 0.0, 1.0), 4)


def _score_notice_period(sig: dict, result: SignalResult) -> float:
    """
    Score notice period against JD preference (≤30 days ideal, can buy out 30).

    0–30 days   → 1.00  (ideal)
    31–60 days  → 0.80  (acceptable with buyout)
    61–90 days  → 0.55  (needs negotiation)
    91–120 days → 0.30  (significant friction)
    121+ days   → 0.10  (very hard to close quickly)
    """
    notice = sig.get("notice_period_days", 60) or 60
    result.notice_period_days = notice

    if notice <= config.NOTICE_IDEAL_DAYS:        # ≤30
        score = 1.00
    elif notice <= config.NOTICE_SOFT_MAX_DAYS:   # ≤60
        # Linear from 0.80 down to 0.65 across 31–60
        t = (notice - config.NOTICE_IDEAL_DAYS) / (
            config.NOTICE_SOFT_MAX_DAYS - config.NOTICE_IDEAL_DAYS
        )
        score = 0.80 - 0.15 * t
    elif notice <= config.NOTICE_HARD_MAX_DAYS:   # ≤90
        t = (notice - config.NOTICE_SOFT_MAX_DAYS) / (
            config.NOTICE_HARD_MAX_DAYS - config.NOTICE_SOFT_MAX_DAYS
        )
        score = 0.55 - 0.25 * t
        result.notes.append(
            f"Notice period {notice}d — needs negotiation"
        )
    elif notice <= 120:
        t = (notice - config.NOTICE_HARD_MAX_DAYS) / 30
        score = 0.30 - 0.10 * t
        result.notes.append(
            f"Long notice period {notice}d — significant hiring friction"
        )
    else:
        score = 0.10
        result.notes.append(
            f"Very long notice period {notice}d — will be hard to close"
        )

    return round(_clamp(score, 0.0, 1.0), 4)


def _score_platform_trust(sig: dict, result: SignalResult) -> float:
    """
    Score platform credibility signals.

    Inputs: github_activity_score, verified_email, verified_phone,
            linkedin_connected, profile_completeness_score,
            saved_by_recruiters_30d
    """
    score = 0.0

    # ── Verification flags (40% of trust) ────────────────────────────────
    verified_email   = sig.get("verified_email", False)
    verified_phone   = sig.get("verified_phone", False)
    linkedin         = sig.get("linkedin_connected", False)
    verif_count      = sum([verified_email, verified_phone, linkedin])
    verif_score      = verif_count / 3.0
    score           += verif_score * 0.40

    # ── GitHub activity (30% of trust) ────────────────────────────────────
    github = sig.get("github_activity_score", -1)
    if github == -1:
        # No GitHub linked — mild negative for an AI Engineer role
        gh_score = 0.25
        result.notes.append("No GitHub linked — weak signal for this role")
    elif github >= config.GITHUB_GOOD_THRESHOLD:    # ≥50
        gh_score = 0.80 + 0.20 * min((github - 50) / 50, 1.0)
    elif github >= 20:
        gh_score = 0.40 + 0.40 * (github - 20) / 30
    else:
        gh_score = 0.20 + 0.20 * github / 20
    score += gh_score * 0.30

    # ── Profile completeness (20% of trust) ───────────────────────────────
    completeness = sig.get("profile_completeness_score", 0) or 0
    if completeness >= 90:
        comp_score = 1.00
    elif completeness >= config.PROFILE_COMPLETE_MIN:   # ≥60
        comp_score = 0.50 + 0.50 * (completeness - 60) / 30
    else:
        comp_score = completeness / 60 * 0.50
    score += comp_score * 0.20

    # ── Recruiter demand proxy (10% of trust) ────────────────────────────
    # saved_by_recruiters_30d: how many recruiters found this profile worth saving
    saved = sig.get("saved_by_recruiters_30d", 0) or 0
    demand_score = min(saved / 10.0, 1.0)   # cap at 10 saves = full score
    score += demand_score * 0.10

    return round(_clamp(score, 0.0, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer class
# ─────────────────────────────────────────────────────────────────────────────

class SignalScorer:
    """
    Convert redrob_signals into an availability multiplier.

    Usage
    -----
        scorer = SignalScorer()
        result = scorer.score(candidate_dict)
        final  = base_score * result.multiplier
    """

    def score(self, candidate: dict[str, Any]) -> SignalResult:
        cid = candidate.get("candidate_id", "UNKNOWN")
        result = SignalResult(candidate_id=cid)
        sig: dict = candidate.get("redrob_signals", {})

        if not sig:
            result.notes.append("No redrob_signals found — using neutral multiplier")
            result.multiplier = 0.50
            return result

        # ── Sub-scores ────────────────────────────────────────────────────
        avail  = _score_availability(sig, result)
        resp   = _score_responsiveness(sig, result)
        notice = _score_notice_period(sig, result)
        trust  = _score_platform_trust(sig, result)

        result.availability_score   = avail
        result.responsiveness_score = resp
        result.notice_score         = notice
        result.trust_score          = trust

        # ── Weighted combination → raw multiplier ─────────────────────────
        raw = (
            _W_AVAILABILITY   * avail
            + _W_RESPONSIVENESS * resp
            + _W_NOTICE         * notice
            + _W_TRUST          * trust
        )

        # ── Boost for standout availability ──────────────────────────────
        # Candidates active in last 14 days + open_to_work + low notice get a
        # small boost above 1.0 (recruiter's dream)
        boost = 0.0
        if result.days_since_active <= 14 and sig.get("open_to_work_flag"):
            boost += 0.05
        if result.notice_period_days <= 30 and raw >= 0.85:
            boost += 0.05

        result.multiplier = round(
            _clamp(raw + boost, _MULTIPLIER_FLOOR, _MULTIPLIER_CEILING), 4
        )
        return result

    def score_batch(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, SignalResult]:
        return {
            c.get("candidate_id", f"idx_{i}"): self.score(c)
            for i, c in enumerate(candidates)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (run: python signals.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pathlib

    sample_path = pathlib.Path("/mnt/user-data/uploads/sample_candidates.json")
    with open(sample_path) as f:
        candidates = json.load(f)

    scorer  = SignalScorer()
    results = scorer.score_batch(candidates)
    ranked  = sorted(results.values(), key=lambda r: r.multiplier, reverse=True)

    print(f"{'Rank':<5} {'CID':<15} {'Mult':>5}  {'Avail':>5} {'Resp':>5} {'Notice':>6} {'Trust':>5}  {'NoticeD':>7}  Notes")
    print("─" * 115)
    for i, r in enumerate(ranked, 1):
        cand  = next(c for c in candidates if c["candidate_id"] == r.candidate_id)
        title = cand["profile"]["current_title"]
        note  = r.notes[0][:50] if r.notes else ""
        print(
            f"{i:<5} {r.candidate_id:<15} {r.multiplier:>5.3f}  "
            f"{r.availability_score:>5.2f} {r.responsiveness_score:>5.2f} "
            f"{r.notice_score:>6.2f} {r.trust_score:>5.2f}  "
            f"{r.notice_period_days:>7}d  {note}"
        )

    print()
    print("=== GOLD STANDARD (CAND_0000031) ===")
    r = results["CAND_0000031"]
    print(f"  Multiplier        : {r.multiplier}")
    print(f"  Availability      : {r.availability_score}  (days_since_active={r.days_since_active})")
    print(f"  Responsiveness    : {r.responsiveness_score}")
    print(f"  Notice            : {r.notice_score}  ({r.notice_period_days}d)")
    print(f"  Trust             : {r.trust_score}")
    if r.notes:
        print(f"  Notes             : {r.notes}")
