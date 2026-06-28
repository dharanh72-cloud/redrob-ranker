"""
scorers/location.py — Redrob Hackathon
========================================
Scores how well a candidate's location and work preferences fit the JD.

JD logistics (from job_description.md)
----------------------------------------
- Primary offices  : Pune, Noida
- Acceptable hubs  : Hyderabad, Mumbai, Delhi NCR, Bengaluru, Gurgaon
- Work mode        : Hybrid (2–3 days onsite per week)
- Relocation       : Company will support within India; no overseas relocation
- Note             : "Remote-only candidates will not be considered unless
                     they are in Pune or Noida and the arrangement is pre-agreed"

Scoring dimensions
-------------------
1. City tier     (50%) — Pune/Noida > acceptable India hubs > rest-of-India
                          > overseas-with-relocation > overseas-locked
2. Work mode     (30%) — hybrid/onsite/flexible > remote-with-relocation
                          > remote-locked
3. Relocation    (20%) — willing_to_relocate boosts candidates outside preferred
                          cities; irrelevant if already in Pune/Noida

Score range: 0.0 – 1.0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ─────────────────────────────────────────────────────────────────────────────
# City classification tables
# ─────────────────────────────────────────────────────────────────────────────

# Tier 1 — JD primary offices (full marks)
_TIER1: set[str] = {
    "pune", "noida",
}

# Tier 2 — Acceptable Indian hubs (JD says fine with hybrid travel)
_TIER2: set[str] = {
    "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
    "gurgaon", "gurugram", "delhi ncr", "ncr",
    "new delhi",
}

# Tier 3 — Other Indian cities (need to relocate but same country = supportable)
# Anything in India not in Tier 1/2 falls here automatically.

# Known overseas countries — company won't sponsor international relocation
_OVERSEAS_COUNTRIES: set[str] = {
    "usa", "us", "united states", "uk", "united kingdom",
    "australia", "germany", "canada", "singapore", "uae",
    "dubai", "netherlands", "france", "japan", "sweden",
    "new zealand", "ireland", "switzerland",
}

# Sub-component weights
_W_CITY      = 0.50
_W_WORK_MODE = 0.30
_W_RELOCATE  = 0.20

assert abs(_W_CITY + _W_WORK_MODE + _W_RELOCATE - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocationScoreResult:
    candidate_id: str
    final_score: float = 0.0

    city_score:      float = 0.0
    work_mode_score: float = 0.0
    relocation_score: float = 0.0

    city_tier:    str = ""    # "tier1" | "tier2" | "tier3_india" | "overseas"
    is_overseas:  bool = False
    work_mode:    str = ""
    will_relocate: bool = False
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<LocationScore {self.candidate_id} "
            f"final={self.final_score:.3f} "
            f"city={self.city_score:.2f}({self.city_tier}) "
            f"mode={self.work_mode_score:.2f}({self.work_mode}) "
            f"reloc={self.relocation_score:.2f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _extract_city(location: str) -> str:
    """Pull the city portion from 'City, State' or 'City' location strings."""
    return _norm(location.split(",")[0].strip())


def _classify_city(city: str, country: str) -> str:
    """Return 'tier1', 'tier2', 'tier3_india', or 'overseas'."""
    c = _norm(country)
    if c in _OVERSEAS_COUNTRIES:
        return "overseas"
    if city in _TIER1:
        return "tier1"
    if city in _TIER2:
        return "tier2"
    return "tier3_india"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_city(
    city_tier: str,
    will_relocate: bool,
    is_overseas: bool,
    result: LocationScoreResult,
) -> float:
    """
    Score the candidate's city against JD primary offices.

    tier1 (Pune/Noida)    → 1.00  — already there
    tier2 + relocate      → 0.85  — short travel / easy relocation
    tier2 + no relocate   → 0.65  — can still do hybrid from close cities
    tier3_india + relocate→ 0.50  — company can support domestic relocation
    tier3_india + no reloc→ 0.25  — would need to commute / won't move
    overseas + relocate   → 0.20  — company doesn't sponsor international
    overseas + no relocate→ 0.05  — effectively unavailable for hybrid role
    """
    if city_tier == "tier1":
        return 1.00

    if city_tier == "tier2":
        if will_relocate:
            return 0.85
        else:
            # Can still hybrid if they travel occasionally
            result.notes.append(
                f"Tier-2 city, not willing to relocate — hybrid travel feasible but not guaranteed"
            )
            return 0.65

    if city_tier == "tier3_india":
        if will_relocate:
            result.notes.append("Indian tier-3 city but willing to relocate domestically")
            return 0.50
        else:
            result.notes.append(
                "Non-hub Indian city, unwilling to relocate — significant logistics risk"
            )
            return 0.25

    # Overseas
    if will_relocate:
        result.notes.append(
            "Overseas candidate willing to relocate — company doesn't sponsor international moves"
        )
        return 0.20
    else:
        result.notes.append(
            "Overseas candidate, not willing to relocate — incompatible with hybrid onsite requirement"
        )
        return 0.05


def _score_work_mode(
    work_mode: str,
    city_tier: str,
    will_relocate: bool,
    result: LocationScoreResult,
) -> float:
    """
    Score preferred work mode against the JD's hybrid requirement.

    hybrid    → 1.00  — perfect match
    onsite    → 0.90  — more than asked, but fine
    flexible  → 0.85  — candidate is open, can negotiate hybrid
    remote    → depends on city and relocation:
                  tier1 remote     → 0.60  (local, could negotiate)
                  remote+relocate  → 0.40  (wants remote but open to discuss)
                  remote locked    → 0.15  (won't budge — misaligned)
    """
    mode = _norm(work_mode)

    if mode == "hybrid":
        return 1.00

    if mode == "onsite":
        return 0.90

    if mode == "flexible":
        return 0.85

    if mode == "remote":
        if city_tier == "tier1":
            result.notes.append(
                "Prefers remote but is in Pune/Noida — hybrid negotiation very feasible"
            )
            return 0.60
        elif will_relocate:
            result.notes.append(
                "Prefers remote but willing to relocate — hybrid may be negotiable"
            )
            return 0.40
        else:
            result.notes.append(
                "Remote-only preference with no relocation willingness — JD is hybrid"
            )
            return 0.15

    # Unknown / missing
    return 0.50


def _score_relocation(
    city_tier: str,
    will_relocate: bool,
    is_overseas: bool,
) -> float:
    """
    Score the relocation signal as a standalone dimension.

    If already in Tier 1 → relocation is irrelevant → 1.0 (full marks)
    Otherwise willing_to_relocate is a meaningful positive signal.
    Overseas + willing → still capped (international relocation not supported).
    """
    if city_tier == "tier1":
        return 1.00   # already there — relocation irrelevant

    if is_overseas:
        # Company won't sponsor; willingness doesn't overcome the constraint
        return 0.20 if will_relocate else 0.05

    # Domestic India
    if will_relocate:
        return 1.00 if city_tier == "tier2" else 0.75
    else:
        return 0.30 if city_tier == "tier2" else 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer class
# ─────────────────────────────────────────────────────────────────────────────

class LocationScorer:
    """
    Score a candidate's location and logistics fit for the hybrid Pune/Noida role.

    Usage
    -----
        scorer = LocationScorer()
        result = scorer.score(candidate_dict)
        print(result.final_score)   # 0.0 – 1.0
        print(result.city_tier)     # "tier1" / "tier2" / "tier3_india" / "overseas"
    """

    def score(self, candidate: dict[str, Any]) -> LocationScoreResult:
        cid     = candidate.get("candidate_id", "UNKNOWN")
        result  = LocationScoreResult(candidate_id=cid)

        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        location     = profile.get("location", "") or ""
        country      = profile.get("country", "") or ""
        work_mode    = signals.get("preferred_work_mode", "") or ""
        will_relocate = signals.get("willing_to_relocate", False)

        city      = _extract_city(location)
        city_tier = _classify_city(city, country)
        is_overseas = city_tier == "overseas"

        result.city_tier    = city_tier
        result.is_overseas  = is_overseas
        result.work_mode    = work_mode
        result.will_relocate = will_relocate

        # ── Sub-scores ────────────────────────────────────────────────────
        city_s  = _score_city(city_tier, will_relocate, is_overseas, result)
        mode_s  = _score_work_mode(work_mode, city_tier, will_relocate, result)
        reloc_s = _score_relocation(city_tier, will_relocate, is_overseas)

        result.city_score       = round(city_s,  4)
        result.work_mode_score  = round(mode_s,  4)
        result.relocation_score = round(reloc_s, 4)

        # ── Weighted combination ──────────────────────────────────────────
        raw = (
            _W_CITY      * city_s
            + _W_WORK_MODE * mode_s
            + _W_RELOCATE  * reloc_s
        )

        result.final_score = round(
            max(config.SCORE_MIN, min(config.SCORE_MAX, raw)), 6
        )
        return result

    def score_batch(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, LocationScoreResult]:
        return {
            c.get("candidate_id", f"idx_{i}"): self.score(c)
            for i, c in enumerate(candidates)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test  (run: python scorers/location.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pathlib

    sample_path = pathlib.Path("/mnt/user-data/uploads/sample_candidates.json")
    with open(sample_path) as f:
        candidates = json.load(f)

    scorer  = LocationScorer()
    results = scorer.score_batch(candidates)
    ranked  = sorted(results.values(), key=lambda r: r.final_score, reverse=True)

    print(f"{'Rank':<5} {'CID':<15} {'Score':>6}  {'City':>5} {'Mode':>5} {'Reloc':>5}  {'Tier':>12}  {'WorkMode':>8}  Location / Notes")
    print("─" * 120)
    for i, r in enumerate(ranked, 1):
        cand  = next(c for c in candidates if c["candidate_id"] == r.candidate_id)
        loc   = cand["profile"]["location"]
        note  = f" [{r.notes[0][:50]}]" if r.notes else ""
        reloc_flag = "✓" if r.will_relocate else "✗"
        print(
            f"{i:<5} {r.candidate_id:<15} {r.final_score:>6.3f}  "
            f"{r.city_score:>5.2f} {r.work_mode_score:>5.2f} {r.relocation_score:>5.2f}  "
            f"{r.city_tier:>12}  {r.work_mode:>8}  "
            f"{loc} relocate={reloc_flag}{note}"
        )

    print()
    print("=== TIER BREAKDOWN ===")
    from collections import Counter
    tiers = Counter(r.city_tier for r in results.values())
    for tier, count in sorted(tiers.items()):
        bar = "█" * count
        print(f"  {tier:>15}: {count:3d}  {bar}")

    print()
    print("=== SCORE DISTRIBUTION ===")
    buckets = {"0.00–0.20": 0, "0.20–0.40": 0, "0.40–0.60": 0,
               "0.60–0.80": 0, "0.80–1.00": 0}
    for r in results.values():
        s = r.final_score
        if   s < 0.20: buckets["0.00–0.20"] += 1
        elif s < 0.40: buckets["0.20–0.40"] += 1
        elif s < 0.60: buckets["0.40–0.60"] += 1
        elif s < 0.80: buckets["0.60–0.80"] += 1
        else:          buckets["0.80–1.00"] += 1
    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"  {bucket}: {count:3d}  {bar}")

    print()
    print("=== KEY CASES ===")
    key_cases = [
        ("CAND_0000029", "Noida, onsite — should be ~1.0"),
        ("CAND_0000033", "Pune, remote, no relocation — partial penalty"),
        ("CAND_0000031", "Hyderabad, flexible, relocate — good fit"),
        ("CAND_0000008", "Noida, onsite, no relocation — should be ~1.0"),
        ("CAND_0000001", "Toronto, onsite, no relocation — should be low"),
        ("CAND_0000009", "New York, remote, no relocation — should be very low"),
    ]
    for cid, desc in key_cases:
        r = results[cid]
        print(f"  {cid}: score={r.final_score:.3f} | {desc}")
