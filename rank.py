"""
rank.py — Redrob Hackathon
============================
Main entrypoint. Loads candidates.jsonl (or .jsonl.gz), runs the full
scoring pipeline, and writes the top-100 submission CSV.

Usage
-----
    python rank.py --candidates candidates.jsonl --out submission.csv
    python rank.py --candidates candidates.jsonl.gz --out submission.csv
    python rank.py --candidates sample_candidates.json --out test.csv --sample 1000

Pipeline (in order)
--------------------
1. Load & parse candidates
2. Honeypot filter   → excluded candidates get score = 0.0 immediately
3. Skill scorer      → 0.0–1.0
4. Career scorer     → 0.0–1.0
5. Experience scorer → 0.0–1.0
6. Location scorer   → 0.0–1.0  (scorers/location.py)
7. Signal scorer     → 0.05–1.10 multiplier
8. Weighted aggregation → base_score × signal_multiplier
9. Sort → top 100 → reasoning → CSV

Compute constraints (from submission_spec.md)
----------------------------------------------
≤ 5 minutes runtime  |  ≤ 16 GB RAM  |  CPU only  |  No network calls
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, os.path.dirname(__file__))
import config
from honeypot_filter        import HoneypotFilter
from scorers.skills         import SkillScorer
from scorers.career         import CareerScorer
from scorers.experience     import ExperienceScorer
from scorers.location       import LocationScorer
from signals                import SignalScorer
from reasoning              import generate_reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result for one candidate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateRank:
    candidate_id: str
    final_score:  float
    reasoning:    str
    rank:         int = 0   # assigned after sorting


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _iter_candidates(path: str) -> Iterator[dict]:
    """Yield candidate dicts from a .jsonl or .jsonl.gz file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    opener = gzip.open if p.suffix == ".gz" else open
    mode   = "rt"

    with opener(p, mode, encoding="utf-8") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_json_array(path: str) -> list[dict]:
    """Load a JSON array file (e.g. sample_candidates.json)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_candidates(path: str, max_n: int | None = None) -> list[dict]:
    """
    Load candidates from .jsonl, .jsonl.gz, or .json (array) file.
    Optionally limit to first max_n for testing.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # JSON array (sample file)
    if suffix == ".json":
        data = _load_json_array(path)
        return data[:max_n] if max_n else data

    # JSONL / JSONL.GZ
    candidates: list[dict] = []
    for i, c in enumerate(_iter_candidates(path)):
        candidates.append(c)
        if max_n and i + 1 >= max_n:
            break
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    candidates: list[dict],
    verbose: bool = True,
) -> list[CandidateRank]:
    """
    Run the full scoring pipeline on a list of candidates.
    Returns a sorted list of CandidateRank (rank 1 = best).
    """
    t0 = time.perf_counter()

    hf_filter  = HoneypotFilter()
    sk_scorer  = SkillScorer()
    ca_scorer  = CareerScorer()
    ex_scorer  = ExperienceScorer()
    loc_scorer = LocationScorer()
    sig_scorer = SignalScorer()

    total    = len(candidates)
    excluded = 0
    results: list[tuple[float, dict, Any, Any, Any, Any]] = []

    if verbose:
        print(f"[rank.py] Processing {total:,} candidates ...")

    for i, cand in enumerate(candidates):
        # ── Progress log every 10K ────────────────────────────────────────
        if verbose and i > 0 and i % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            rate    = i / elapsed
            eta     = (total - i) / rate if rate > 0 else 0
            print(f"  {i:>7,} / {total:,}  ({elapsed:.1f}s elapsed, ETA {eta:.0f}s)")

        # ── 1. Honeypot filter ────────────────────────────────────────────
        hp = hf_filter.evaluate(cand)
        if hp.is_honeypot:
            excluded += 1
            continue   # skip entirely — don't waste time scoring

        # ── 2–6. Score all components ─────────────────────────────────────
        sk  = sk_scorer.score(cand)
        ca  = ca_scorer.score(cand)
        ex  = ex_scorer.score(cand)
        loc = loc_scorer.score(cand)
        sg  = sig_scorer.score(cand)

        # ── 7. Weighted base score ────────────────────────────────────────
        base = (
            config.WEIGHTS["skills"]       * sk.final_score
            + config.WEIGHTS["career"]     * ca.final_score
            + config.WEIGHTS["experience"] * ex.final_score
            + config.WEIGHTS["location"]   * loc.final_score
        )

        # ── 8. Apply signal multiplier ────────────────────────────────────
        final = round(base * sg.multiplier, config.SCORE_DECIMAL_PLACES)
        final = max(config.SCORE_MIN, min(config.SCORE_MAX, final))

        results.append((final, cand, sk, ca, ex, sg))

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Done. {total:,} candidates in {elapsed:.1f}s  "
              f"({excluded} honeypots excluded, {len(results):,} scored)")

    # ── 9. Sort descending by score, tie-break by candidate_id asc ───────
    results.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "")))

    # ── 10. Take top-N, generate reasoning, assign ranks ─────────────────
    if verbose:
        print(f"[rank.py] Generating reasoning for top {config.TOP_N} ...")

    ranked: list[CandidateRank] = []
    for rank_idx, (score, cand, sk, ca, ex, sg) in enumerate(results[:config.TOP_N], 1):
        r = generate_reasoning(cand, score, sk, ca, ex, sg)
        ranked.append(CandidateRank(
            candidate_id=cand["candidate_id"],
            final_score=score,
            reasoning=r.reasoning,
            rank=rank_idx,
        ))

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(ranked: list[CandidateRank], out_path: str) -> None:
    """Write submission CSV per submission_spec.md format."""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(config.CSV_COLUMNS)   # header
        for cr in ranked:
            writer.writerow([
                cr.candidate_id,
                cr.rank,
                round(cr.final_score, config.SCORE_DECIMAL_PLACES),
                cr.reasoning,
            ])
    print(f"[rank.py] Wrote {len(ranked)} rows → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redrob Hackathon ranker — produces top-100 submission CSV"
    )
    p.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl, candidates.jsonl.gz, or sample_candidates.json",
    )
    p.add_argument(
        "--out", required=True,
        help="Output CSV path (e.g. team_xyz.csv)",
    )
    p.add_argument(
        "--sample", type=int, default=None,
        help="Only process first N candidates (for testing speed)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    t_start = time.perf_counter()

    print(f"[rank.py] Loading candidates from: {args.candidates}")
    candidates = load_candidates(args.candidates, max_n=args.sample)
    print(f"[rank.py] Loaded {len(candidates):,} candidates")

    ranked = run_pipeline(candidates, verbose=not args.quiet)

    if not ranked:
        print("[rank.py] ERROR: no candidates survived the pipeline — check your input file")
        sys.exit(1)

    write_csv(ranked, args.out)

    total_time = time.perf_counter() - t_start
    print(f"[rank.py] Total runtime: {total_time:.1f}s")

    # Warn if approaching the 5-minute limit
    if total_time > config.MAX_RUNTIME_SECS:
        print(f"[rank.py] ⚠  Exceeded {config.MAX_RUNTIME_SECS}s limit! "
              f"Profile your scorers for bottlenecks.")
    else:
        remaining = config.MAX_RUNTIME_SECS - total_time
        print(f"[rank.py] ✓  {remaining:.0f}s remaining under the 5-min constraint")

    # Quick sanity print of top 10
    print("\n[rank.py] Top 10 candidates:")
    print(f"  {'Rank':<5} {'CID':<15} {'Score':>8}  Reasoning snippet")
    print("  " + "─" * 90)
    for cr in ranked[:10]:
        snippet = cr.reasoning[:70] + ("…" if len(cr.reasoning) > 70 else "")
        print(f"  {cr.rank:<5} {cr.candidate_id:<15} {cr.final_score:>8.6f}  {snippet}")


if __name__ == "__main__":
    main()