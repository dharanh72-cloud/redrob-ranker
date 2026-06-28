"""
scorers/experience.py — Redrob Hackathon
==========================================
Scores how well a candidate's years of experience fits the JD band for
Senior AI Engineer.

JD band (from job_description.md)
----------------------------------
- Stated range  : 5–9 years
- Sweet spot    : 6–8 years  (JD says "typically see the best signal here")
- Hard floor    : <3 years   → score = 0.0  (too junior for this role)
- Soft ceiling  : >12 years  → score tapers (over-experienced; risk of being
                               bored or expecting a Staff/Principal scope)

Score curve (visualised)
------------------------
YOE   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15+
      0   0  .05 .15 .45 .75 .92 1.0 1.0 .90 .75 .60 .45 .35 .25 .15

The curve has four regions:
  1. Hard floor   : 0 – 2.9 yrs  → 0.0 – 0.05  (linear ramp, nearly zero)
  2. Ramp-up      : 3 – 5.9 yrs  → 0.05 – 0.75  (accelerating climb)
  3. Peak band    : 6 – 8.0 yrs  → 0.92 – 1.0   (ideal zone, tilt peak at 7)
  4. Taper        : 8+ yrs       → graceful decline, never hitting 0

Verification adjustments
-------------------------
The profile's `years_of_experience` is self-reported. We cross-check it
against two objective sources:
  - Sum of career_history duration_months  (actual tracked time)
  - Earliest career start date             (calendar-based estimate)

If they disagree significantly, we use a blended estimate and flag it.

Score range: 0.0 – 1.0
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Curve control points  (yoe → raw_score)
# These define a piecewise linear curve; _curve_score() interpolates between.
# ─────────────────────────────────────────────────────────────────────────────
_CURVE: list[tuple[float, float]] = [
    (0.0,  0.00),
    (2.0,  0.00),   # hard floor: <2 yrs is zero
    (3.0,  0.05),   # just above floor
    (4.0,  0.35),   # accept min — still steep ramp
    (5.0,  0.65),   # lower accept boundary (JD says 5+)
    (6.0,  0.88),   # entering ideal zone
    (7.0,  1.00),   # absolute peak (midpoint of sweet spot)
    (8.0,  0.98),   # still excellent
    (9.0,  0.88),   # top of stated range — slight taper begins
    (10.0, 0.72),
    (11.0, 0.58),
    (12.0, 0.44),
    (14.0, 0.30),
    (16.0, 0.18),
    (20.0, 0.12),   # floor for very senior — never zero (still could be great)
]

_TODAY = config.SIGNAL_REFERENCE_DATE


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperienceScoreResult:
    candidate_id: str
    final_score: float = 0.0

    # The YOE value actually used for scoring (may differ from stated)
    effective_yoe: float = 0.0
    stated_yoe: float = 0.0
    career_yoe: float = 0.0       # derived from career_history duration_months
    calendar_yoe: float = 0.0     # derived from earliest start date to today

    # How the three estimates were combined
    blend_method: str = ""
    flags: list[str] = field(default_factory=list)

    # Position label for reasoning strings
    yoe_label: str = ""           # e.g. "ideal", "under", "over", "hard floor"

    def __repr__(self) -> str:
        return (
            f"<ExperienceScore {self.candidate_id} "
            f"score={self.final_score:.3f} "
            f"eff_yoe={self.effective_yoe:.1f} "
            f"stated={self.stated_yoe:.1f} "
            f"label={self.yoe_label}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Curve interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _curve_score(yoe: float) -> float:
    """
    Interpolate the score from the piecewise linear curve.
    Clamps to [0.0, 1.0].
    """
    if yoe <= _CURVE[0][0]:
        return _CURVE[0][1]
    if yoe >= _CURVE[-1][0]:
        return _CURVE[-1][1]

    for i in range(len(_CURVE) - 1):
        x0, y0 = _CURVE[i]
        x1, y1 = _CURVE[i + 1]
        if x0 <= yoe <= x1:
            # Linear interpolation between control points
            t = (yoe - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 6)

    return 0.0   # should never reach here


def _yoe_label(yoe: float) -> str:
    """Return a human-readable label for use in reasoning strings."""
    if yoe < config.YOE_HARD_MIN:
        return "hard floor"
    elif yoe < config.YOE_ACCEPT_MIN:
        return "under"
    elif yoe < config.YOE_IDEAL_MIN:
        return "acceptable"
    elif yoe <= config.YOE_IDEAL_MAX:
        return "ideal"
    elif yoe <= config.YOE_ACCEPT_MAX:
        return "slightly over"
    else:
        return "over"


# ─────────────────────────────────────────────────────────────────────────────
# YOE estimation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yoe_from_career_months(career: list[dict]) -> float:
    """Sum duration_months across all career entries → convert to years."""
    total = sum(ch.get("duration_months") or 0 for ch in career)
    return round(total / 12, 2)


def _yoe_from_calendar(career: list[dict]) -> float:
    """
    Earliest job start date → today = calendar-based experience estimate.
    Gives a rough upper bound (doesn't account for gaps).
    """
    start_dates: list[date] = []
    for ch in career:
        d = _parse_date(ch.get("start_date"))
        if d:
            start_dates.append(d)

    if not start_dates:
        return 0.0

    earliest = min(start_dates)
    months = (_TODAY.year - earliest.year) * 12 + (_TODAY.month - earliest.month)
    return round(max(months, 0) / 12, 2)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Blending logic
# ─────────────────────────────────────────────────────────────────────────────

def _blend_yoe(
    stated: float,
    career: float,
    calendar: float,
    result: ExperienceScoreResult,
) -> float:
    """
    Combine three YOE estimates into a single effective YOE.

    Rules:
    1. If all three are within 1.5 years of each other → trust stated, use it.
    2. If stated is significantly higher than both career and calendar
       (inflated self-report) → use the lower blend.
    3. If stated is significantly lower than both (rare, modest self-reporting)
       → give benefit of doubt, nudge up slightly.
    4. If career and calendar broadly agree but stated is an outlier → use
       the average of career and calendar.
    """
    flags = result.flags

    # Guard: if career data is sparse (only 1 role), calendar is more reliable
    career_reliable = career > 0.5

    if not career_reliable and calendar <= 0:
        # Nothing to cross-check — trust stated
        result.blend_method = "stated_only"
        return stated

    # Deviation between stated and objective estimates
    reference = (career + calendar) / 2 if career_reliable else calendar
    deviation = stated - reference        # positive = stated is higher

    if abs(deviation) <= 1.5:
        # All estimates agree — trust stated
        result.blend_method = "all_agree"
        return stated

    elif deviation > 2.5:
        # Stated is notably inflated compared to actual career record
        flags.append(
            f"Stated YOE ({stated:.1f}) is {deviation:.1f}yr higher than "
            f"career-derived estimate ({reference:.1f}) — using blended value"
        )
        # Use a weighted blend that leans toward objective sources
        blended = round(0.30 * stated + 0.70 * reference, 2)
        result.blend_method = "deflated_blend"
        return blended

    elif deviation < -2.0:
        # Stated is lower than actual record (conservative self-report)
        flags.append(
            f"Stated YOE ({stated:.1f}) is {abs(deviation):.1f}yr lower than "
            f"career-derived estimate ({reference:.1f}) — giving benefit of doubt"
        )
        blended = round(0.60 * stated + 0.40 * reference, 2)
        result.blend_method = "nudged_blend"
        return blended

    else:
        # Small disagreement — slight correction toward objective sources
        blended = round(0.60 * stated + 0.40 * reference, 2)
        result.blend_method = "minor_blend"
        return blended


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer class
# ─────────────────────────────────────────────────────────────────────────────

class ExperienceScorer:
    """
    Score a candidate's years-of-experience fit against the JD band.

    Usage
    -----
        scorer = ExperienceScorer()
        result = scorer.score(candidate_dict)
        print(result.final_score)    # 0.0 – 1.0
        print(result.effective_yoe)  # YOE actually used
        print(result.yoe_label)      # "ideal" / "under" / "over" etc.
    """

    def score(self, candidate: dict[str, Any]) -> ExperienceScoreResult:
        cid    = candidate.get("candidate_id", "UNKNOWN")
        result = ExperienceScoreResult(candidate_id=cid)

        profile = candidate.get("profile", {})
        career  = candidate.get("career_history", [])

        # ── Three YOE estimates ───────────────────────────────────────────────
        stated_yoe   = float(profile.get("years_of_experience") or 0)
        career_yoe   = _yoe_from_career_months(career)
        calendar_yoe = _yoe_from_calendar(career)

        result.stated_yoe   = stated_yoe
        result.career_yoe   = career_yoe
        result.calendar_yoe = calendar_yoe

        # ── Blend into a single effective YOE ────────────────────────────────
        effective_yoe = _blend_yoe(stated_yoe, career_yoe, calendar_yoe, result)
        result.effective_yoe = effective_yoe

        # ── Apply the score curve ─────────────────────────────────────────────
        raw_score = _curve_score(effective_yoe)

        # ── Assign a human-readable label ─────────────────────────────────────
        result.yoe_label = _yoe_label(effective_yoe)

        # ── Clamp and store ───────────────────────────────────────────────────
        result.final_score = round(
            min(max(raw_score, config.SCORE_MIN), config.SCORE_MAX), 6
        )
        return result

    def score_batch(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, ExperienceScoreResult]:
        return {
            c.get("candidate_id", f"idx_{i}"): self.score(c)
            for i, c in enumerate(candidates)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (run: python scorers/experience.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pathlib

    # ── Print the full score curve first ─────────────────────────────────────
    print("=== SCORE CURVE (YOE → score) ===")
    print(f"{'YOE':>5}  {'Score':>6}  {'Label':>12}  Bar")
    print("─" * 55)
    for yoe_f in [0, 1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
                  8.5, 9, 10, 11, 12, 13, 14, 16, 20]:
        sc = _curve_score(yoe_f)
        lbl = _yoe_label(yoe_f)
        bar = "█" * int(sc * 30)
        print(f"{yoe_f:>5.1f}  {sc:>6.3f}  {lbl:>12}  {bar}")

    print()

    # ── Run on sample candidates ──────────────────────────────────────────────
    sample_path = pathlib.Path("/mnt/user-data/uploads/sample_candidates.json")
    with open(sample_path) as f:
        candidates = json.load(f)

    scorer  = ExperienceScorer()
    results = scorer.score_batch(candidates)
    ranked  = sorted(results.values(), key=lambda r: r.final_score, reverse=True)

    print("=== CANDIDATE RESULTS (top 25 by score) ===")
    print(f"{'Rank':<5} {'CID':<15} {'Score':>6}  {'Eff':>5} {'Stated':>6} {'Career':>6} {'Cal':>5}  {'Label':>12}  {'Method':>15}  Title")
    print("─" * 130)
    for i, r in enumerate(ranked[:25], 1):
        cand  = next(c for c in candidates if c["candidate_id"] == r.candidate_id)
        title = cand["profile"]["current_title"]
        flag  = " ⚠" if r.flags else ""
        print(
            f"{i:<5} {r.candidate_id:<15} {r.final_score:>6.3f}  "
            f"{r.effective_yoe:>5.1f} {r.stated_yoe:>6.1f} "
            f"{r.career_yoe:>6.1f} {r.calendar_yoe:>5.1f}  "
            f"{r.yoe_label:>12}  {r.blend_method:>15}  "
            f"{title}{flag}"
        )

    print()
    print("=== CANDIDATES WITH BLEND FLAGS ===")
    flagged = [r for r in results.values() if r.flags]
    if flagged:
        for r in flagged:
            cand = next(c for c in candidates if c["candidate_id"] == r.candidate_id)
            print(f"  {r.candidate_id} | eff={r.effective_yoe:.1f} stated={r.stated_yoe:.1f}")
            for flag in r.flags:
                print(f"    • {flag}")
    else:
        print("  None — all stated YOEs are consistent with career records.")

    print()
    print("=== SCORE DISTRIBUTION ===")
    buckets = {"0.00–0.10": 0, "0.10–0.30": 0, "0.30–0.50": 0, "0.50–0.70": 0,
               "0.70–0.90": 0, "0.90–1.00": 0}
    for r in results.values():
        s = r.final_score
        if   s < 0.10: buckets["0.00–0.10"] += 1
        elif s < 0.30: buckets["0.10–0.30"] += 1
        elif s < 0.50: buckets["0.30–0.50"] += 1
        elif s < 0.70: buckets["0.50–0.70"] += 1
        elif s < 0.90: buckets["0.70–0.90"] += 1
        else:          buckets["0.90–1.00"] += 1
    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"  {bucket}: {count:3d}  {bar}")
