"""
4_extract_planeat.py — mine user-preference patterns from the abstracted log and
emit training samples to ``data/user_behavior_samples.json``.

Extended version of 4_extract_new.py with additional pattern type:
  (c) periodic / time-slot: apps that cluster in specific hour_bins even when
      no single preceding event reliably triggers them.

Pipeline (per plan rev 3 + periodic extension):

  Step A  — parse + sort
  Step C  — coalesce repeats (≥2 same line within 2 min)        [moved before B]
  Step B  — drop events with count < min_occurrence
  Step D  — assign context (day, hour_bin, location, movement)
  Step E  — mine candidate patterns
            (a) sequential: every event Y in (t_X − Δt, t_X) for each X
            (b) context-onset: when context changes, emit virtual onset and
                pair with all events within Δt after the change
            (c) periodic/time-slot: for each (app, hour_bin_set) pair where
                the app's usage concentrates in those bins, emit a pattern
                with trigger="Enter <hour_bin_set> time slot" and the app as gold
  Step E.1 — minimum context per (trigger, gold) group: keep only the dims
            whose value is identical across all candidates in the group
  Step E.2 — compute occurrence and confidence; filter by thresholds
  Step F  — detect shifted patterns (≥shifted_ctx_dims_min context dims match,
            different gold between halves)
  Step G  — temporal categorization (always / shifted; emergent/decaying opt-in)
  Step H  — generate samples (input_text, gold_answer, gold_reasoning)
  Step I  — splits (regular + continual)
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import accumulate
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_FMT = "%y-%m-%d %H:%M"

# Hour-of-day bins per plan rev 3.
HOUR_BINS: list[tuple[str, int, int]] = [
    ("dawn",           0,  7),
    ("early_morning",  7,  9),
    ("morning",        9, 11),
    ("lunch",         11, 13),
    ("afternoon",     13, 15),
    ("late_afternoon", 15, 18),
    ("dinner",        18, 20),
    ("evening",       20, 22),
    ("night",         22, 24),
]

HOUR_BIN_NAMES = [name for name, _, _ in HOUR_BINS]
HOUR_BIN_ORDER = {name: i for i, (name, _, _) in enumerate(HOUR_BINS)}

# Lines that signal movement/state changes — used to maintain `movement` ctx.
_MOVEMENT_PATTERNS = {
    "Begin walking":   ("walking",     "start"),
    "Stop walking":    ("walking",     "stop"),
    "Begin running":   ("running",     "start"),
    "Stop running":    ("running",     "stop"),
    "Board a vehicle": ("in_vehicle",  "start"),
    "Get off vehicle": ("in_vehicle",  "stop"),
}

# Hour-bin boundary hours (used to enumerate context-onset moments mid-gap).
_HOUR_BIN_BOUNDARIES = [0, 7, 9, 11, 13, 15, 18, 20, 22]

# Gold blocklist for SEQUENTIAL candidates only — these are passive state
# transitions, not deliberate user actions.
SEQUENTIAL_GOLD_BLOCKLIST = (
    "Begin walking", "Stop walking",
    "Begin running", "Stop running",
    "Board a vehicle", "Get off vehicle",
    "Day starts:",
)

# Only "Use X app" lines are eligible as periodic golds.
PERIODIC_GOLD_PREFIX = "Use "


def _is_blocked_sequential_gold(line: str) -> bool:
    return any(line.startswith(p) for p in SEQUENTIAL_GOLD_BLOCKLIST)


# ---------------------------------------------------------------------------
# Step A — parse + sort
# ---------------------------------------------------------------------------

def parse_log(path: str) -> list[tuple[datetime, str]]:
    """Read the abstracted log and return list of (datetime, line)."""
    with open(path) as f:
        data = json.load(f)
    out: list[tuple[datetime, str]] = []
    for s in data:
        if " | " not in s:
            continue
        t_str, line = s.split(" | ", 1)
        try:
            t = datetime.strptime(t_str, TIME_FMT)
        except ValueError:
            continue
        out.append((t, line.strip()))
    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------------------
# Step C — coalesce repeats (≥2 same line within 2 min)
# ---------------------------------------------------------------------------

def coalesce_repeats(events: list[tuple[datetime, str]],
                     gap_seconds: int = 120) -> list[tuple[datetime, str]]:
    """Collapse runs of identical lines within `gap_seconds`. Keeps the first
    occurrence of each run; later same-line events within the gap are dropped."""
    out: list[tuple[datetime, str]] = []
    last_t: dict[str, datetime] = {}
    for t, line in events:
        prev = last_t.get(line)
        if prev is not None and (t - prev).total_seconds() <= gap_seconds:
            last_t[line] = t  # extend the run window
            continue
        out.append((t, line))
        last_t[line] = t
    return out


# ---------------------------------------------------------------------------
# Step B — drop low-frequency events
# ---------------------------------------------------------------------------

def filter_frequent(events: list[tuple[datetime, str]],
                    min_count: int = 3) -> tuple[list[tuple[datetime, str]], dict[str, int]]:
    """Keep only events whose line occurs ≥ min_count times. Returns the
    filtered list and the counter for diagnostics."""
    counts = Counter(line for _, line in events)
    kept = [(t, line) for t, line in events if counts[line] >= min_count]
    return kept, dict(counts)


# ---------------------------------------------------------------------------
# Step D — context bucket
# ---------------------------------------------------------------------------

def hour_bin_of(t: datetime) -> str:
    h = t.hour
    for name, lo, hi in HOUR_BINS:
        if lo <= h < hi:
            return name
    return "dawn"


def day_of(t: datetime) -> str:
    return "weekend" if t.weekday() >= 5 else "weekday"


@dataclass(frozen=True)
class Context:
    day: str
    hour_bin: str
    location: str
    movement: str

    def as_dict(self) -> dict[str, str]:
        return {"day": self.day, "hour_bin": self.hour_bin,
                "location": self.location, "movement": self.movement}

    def render(self) -> str:
        parts = [f"{k}={v}" for k, v in self.as_dict().items() if v]
        return ", ".join(parts)


def assign_contexts(events: list[tuple[datetime, str]]) -> list[Context]:
    """Walk the event timeline once and assign a Context to each event by
    tracking the most recent location change and movement state change."""
    last_location = "unknown"
    movement_state = "stationary"
    contexts: list[Context] = []
    for t, line in events:
        for prefix, (state, kind) in _MOVEMENT_PATTERNS.items():
            if line.startswith(prefix):
                movement_state = state if kind == "start" else "stationary"
                break
        if line.startswith("Moved to "):
            last_location = line[len("Moved to "):].strip()
        contexts.append(Context(
            day=day_of(t),
            hour_bin=hour_bin_of(t),
            location=last_location,
            movement=movement_state,
        ))
    return contexts


# ---------------------------------------------------------------------------
# Step E — pattern mining
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    trigger: str
    gold: str
    ctx: Context             # full post-change context at trigger time
    t: datetime
    is_context_only: bool = False
    is_periodic: bool = False   # NEW: marks periodic/time-slot candidates
    # For context-onset candidates: which ctx dims just changed at onset_t.
    changed_dims: tuple = ()


def _enumerate_context_change_points(
    events: list[tuple[datetime, str]],
) -> list[tuple[datetime, Context, tuple[str, ...]]]:
    """Walk a merged timeline of (event times) ∪ (hour_bin boundaries) and
    emit (time, new_context, changed_dims) at every moment the 4-dim context
    changes."""
    if not events:
        return []
    log_start = events[0][0]
    log_end = events[-1][0]

    boundary_times: list[datetime] = []
    cur_day = log_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur_day <= log_end:
        for h in _HOUR_BIN_BOUNDARIES:
            t = cur_day.replace(hour=h)
            if log_start <= t <= log_end:
                boundary_times.append(t)
        cur_day += timedelta(days=1)

    merged: list[tuple[datetime, Optional[str]]] = (
        [(t, line) for t, line in events]
        + [(t, None) for t in boundary_times]
    )
    merged.sort(key=lambda x: x[0])

    DIMS = ("day", "hour_bin", "location", "movement")
    last_loc = "unknown"
    last_mov = "stationary"
    onsets: list[tuple[datetime, Context, tuple[str, ...]]] = []
    prev_ctx: Optional[Context] = None
    for t, line in merged:
        if line is not None:
            if line.startswith("Moved to "):
                last_loc = line[len("Moved to "):].strip()
            for prefix, (state, kind) in _MOVEMENT_PATTERNS.items():
                if line.startswith(prefix):
                    last_mov = state if kind == "start" else "stationary"
                    break
        cur_ctx = Context(
            day=day_of(t), hour_bin=hour_bin_of(t),
            location=last_loc, movement=last_mov,
        )
        if cur_ctx != prev_ctx:
            if prev_ctx is None:
                changed = DIMS
            else:
                changed = tuple(
                    d for d in DIMS
                    if getattr(cur_ctx, d) != getattr(prev_ctx, d)
                )
            onsets.append((t, cur_ctx, changed))
            prev_ctx = cur_ctx
    return onsets


def mine_candidates(events: list[tuple[datetime, str]],
                    contexts: list[Context],
                    delta_t_min: int,
                    events_full: Optional[list[tuple[datetime, str]]] = None,
                    periodic_concentration_min: float = 0.15,
                    periodic_combined_min_concentration: float = 0.70,
                    periodic_min_slot_occurrence: int = 3,
                    ) -> list[Candidate]:
    """Step E: emit (trigger, gold, context, time) candidates.

    (a) Sequential: for every event X at index i, every event Y in
        (t_X − Δt, t_X) becomes a candidate (Y, X). Apply
        SEQUENTIAL_GOLD_BLOCKLIST to drop X's that are passive state
        transitions.

    (b) Context-onset: whenever ANY ctx dim changes, emit a virtual onset.
        Every event in [onset_t, onset_t + Δt] becomes a gold.

    (c) Periodic/time-slot: for each app that clusters in specific hour_bins,
        emit candidates with trigger="Enter <hour_bin> time slot".
        The candidate fires at the first occurrence of the app within each
        time-slot entry (hour_bin onset) — one candidate per entry, not per
        event, to avoid double-counting with sequential/context-onset.
    """
    delta = timedelta(minutes=delta_t_min)
    n = len(events)
    candidates: list[Candidate] = []

    # ------- (a) sequential — gold blocklist applies -------
    j_lo = 0
    for i in range(n):
        t_i, line_i = events[i]
        if _is_blocked_sequential_gold(line_i):
            continue
        while j_lo < i and (t_i - events[j_lo][0]) > delta:
            j_lo += 1
        for j in range(j_lo, i):
            t_j, line_j = events[j]
            if line_j == line_i:
                continue
            candidates.append(Candidate(
                trigger=line_j, gold=line_i, ctx=contexts[i], t=t_i,
                is_context_only=False,
            ))

    # ------- (b) context-onset (no blocklist) -------
    onsets = _enumerate_context_change_points(
        events_full if events_full is not None else events
    )
    times = [t for t, _ in events]
    for onset_t, onset_ctx, changed_dims in onsets:
        changed_pairs = sorted(
            (d, getattr(onset_ctx, d)) for d in changed_dims
        )
        trigger_str = "Context changed to: " + ", ".join(
            f"{d}={v}" for d, v in changed_pairs
        )
        lo = bisect.bisect_right(times, onset_t)
        hi_t = onset_t + delta
        hi = bisect.bisect_right(times, hi_t)
        for k in range(lo, hi):
            t_k, line_k = events[k]
            candidates.append(Candidate(
                trigger=trigger_str,
                gold=line_k,
                ctx=onset_ctx,
                t=t_k,
                is_context_only=True,
                changed_dims=changed_dims,
            ))

    # ------- (c) periodic / time-slot -------
    # For each app, find hour_bins where it concentrates above a threshold.
    # Then for each entry into that hour_bin, if the app is used within the
    # time-slot window, emit one candidate.
    candidates.extend(
        _mine_periodic_candidates(events, contexts, events_full, delta_t_min,
                                  concentration_min=periodic_concentration_min,
                                  combined_min_concentration=periodic_combined_min_concentration,
                                  min_slot_occurrence=periodic_min_slot_occurrence)
    )

    return candidates


def _mine_periodic_candidates(
    events: list[tuple[datetime, str]],
    contexts: list[Context],
    events_full: Optional[list[tuple[datetime, str]]],
    delta_t_min: int,
    concentration_min: float = 0.15,
    combined_min_concentration: float = 0.70,
    min_slot_occurrence: int = 3,
) -> list[Candidate]:
    """Step E(c): mine periodic/time-slot patterns.

    For each "Use X app" line:
      1. Count occurrences per (day_type, hour_bin) pair.
      2. Find hour_bin sets where:
         - Each selected bin individually has ≥ concentration_min of total usage
         - Selected bins combined have ≥ combined_min_concentration of total usage
      3. For each qualifying (app, day_type, hour_bin_set):
         - Walk the timeline and find every entry into that hour_bin_set.
         - If the app is used during that time-slot entry, emit a candidate.

    The two-threshold rule ensures: (1) each bin in the pattern is genuinely
    time-concentrated, (2) the bins together cover most of the app's usage,
    so the app IS a time-driven app — not one that's spread across all hours.
    """
    # Only consider "Use X app" events
    app_events = [(t, line) for t, line in events if line.startswith(PERIODIC_GOLD_PREFIX)]

    if not app_events:
        return []

    # Step 1: count per (day_type, hour_bin) for each app
    app_ctx_counts: dict[str, Counter] = defaultdict(Counter)
    for t, line in app_events:
        dt = day_of(t)
        hb = hour_bin_of(t)
        app_ctx_counts[line][(dt, hb)] += 1

    # Step 2: find concentrated (app, day_type, hour_bin_set) groups
    # Rule: (1) each selected hour_bin must individually have ≥ bin_min_concentration
    # of the app's total usage, (2) the selected bins combined must have
    # ≥ combined_min_concentration of total usage.
    #
    # We try two strategies and keep whichever passes:
    #   (i)  per-day_type (weekday/weekend) — captures patterns specific to one
    #   (ii) all-days combined — captures patterns that span both day types
    periodic_specs: list[tuple[str, str, tuple[str, ...]]] = []  # (app, day_type, hour_bin_tuple)
    seen_specs: set[tuple[str, str, tuple[str, ...]]] = set()

    def _try_spec(app: str, dt: str, hb_counts: Counter, total_app: int):
        """Check concentration thresholds and add to periodic_specs if passed."""
        # (1) Select bins where individual concentration ≥ bin_min_concentration
        selected_bins: set[str] = set()
        for hb, cnt in hb_counts.items():
            if cnt / total_app >= concentration_min:
                selected_bins.add(hb)

        if not selected_bins:
            return

        # (2) Check combined concentration ≥ combined_min_concentration
        combined = sum(hb_counts[hb] for hb in selected_bins) / total_app
        if combined < combined_min_concentration:
            return

        ordered = tuple(sorted(selected_bins, key=lambda b: HOUR_BIN_ORDER[b]))
        spec = (app, dt, ordered)
        if spec not in seen_specs:
            seen_specs.add(spec)
            periodic_specs.append(spec)

    for app, ctx_counts in app_ctx_counts.items():
        total_app = sum(ctx_counts.values())
        if total_app < min_slot_occurrence:
            continue

        # Group by day_type
        by_day: dict[str, Counter] = defaultdict(Counter)
        for (dt, hb), cnt in ctx_counts.items():
            by_day[dt][hb] += cnt

        # Strategy (i): per-day_type
        for dt, hb_counts in by_day.items():
            dt_total = sum(hb_counts.values())
            if dt_total < min_slot_occurrence:
                continue
            _try_spec(app, dt, hb_counts, total_app)

        # Strategy (ii): all-days combined
        all_hb_counts: Counter = Counter()
        for hb_counts in by_day.values():
            all_hb_counts += hb_counts
        _try_spec(app, "any", all_hb_counts, total_app)

    if not periodic_specs:
        return []

    # Step 3: for each spec, walk the timeline and find time-slot entries
    # where the app is used. Use the FULL event timeline for context tracking.
    timeline = events_full if events_full is not None else events
    times = [t for t, _ in events]
    candidates: list[Candidate] = []

    for app, dt, hb_set in periodic_specs:
        hb_set_set = set(hb_set)

        # Build trigger string
        if len(hb_set) == 1:
            trigger_str = f"Enter {hb_set[0]} time slot"
        else:
            trigger_str = f"Enter {'/'.join(hb_set)} time slot"

        # Walk the full timeline to find entries into the time-slot.
        # An "entry" is when we transition into one of the target hour_bins
        # on the correct day_type.
        prev_in_slot = False
        slot_entry_time: Optional[datetime] = None
        app_used_in_slot = False
        slot_ctx: Optional[Context] = None

        for idx, (t, line) in enumerate(timeline):
            cur_dt = day_of(t)
            cur_hb = hour_bin_of(t)
            in_slot = (cur_hb in hb_set_set) and (dt == "any" or cur_dt == dt)

            if in_slot and not prev_in_slot:
                # Entering the time-slot
                slot_entry_time = t
                app_used_in_slot = False
                # Get context from the filtered events at this time
                # Use the nearest filtered event's context
                fi = bisect.bisect_right(times, t)
                if fi < len(contexts):
                    slot_ctx = contexts[fi]
                else:
                    slot_ctx = contexts[-1]

            if in_slot and not app_used_in_slot and line == app:
                # First usage of the app in this time-slot entry
                app_used_in_slot = True
                # Find the context from the filtered events
                fi = bisect.bisect_left(times, t)
                if fi < len(contexts):
                    c_ctx = contexts[fi]
                else:
                    c_ctx = contexts[-1]
                candidates.append(Candidate(
                    trigger=trigger_str,
                    gold=app,
                    ctx=c_ctx,
                    t=t,
                    is_periodic=True,
                ))

            prev_in_slot = in_slot

    return candidates


# ---------------------------------------------------------------------------
# Step E.1 — minimum context per (trigger, gold) group
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    trigger: str
    gold: str
    is_context_only: bool
    # ctx as dict; missing keys = "any"
    ctx: dict[str, str]
    is_periodic: bool = False   # NEW
    candidates: list[Candidate] = field(default_factory=list)
    # filled by E.2
    occurrence: int = 0
    confidence: float = 0.0
    # filled by F/G
    weeks_seen: list[int] = field(default_factory=list)
    pattern_id: str = ""
    pattern_category: str = ""


def minimize_context(candidates: list[Candidate]) -> list[Pattern]:
    """Step E.1 — discover context that distinguishes one (trigger, gold) from
    other (trigger, *) groups.

    Periodic candidates are handled separately: each unique
    (trigger, day_type_from_trigger, hour_bin_set, gold) becomes a pattern
    with context constrained to the remaining dims (location, movement) via
    the same dominance logic, but day and hour_bin are NOT minimized away
    because they are intrinsic to the periodic signal.
    """
    DIMS = ("day", "hour_bin", "location", "movement")
    DOMINANCE_RATIO = 0.3
    patterns: list[Pattern] = []

    # ---- Periodic candidates ----
    periodic_by_key: dict[tuple, list[Candidate]] = defaultdict(list)
    for c in candidates:
        if c.is_periodic:
            # Group by (trigger, gold) — the day_type and hour_bin are encoded
            # in the trigger string already
            periodic_by_key[(c.trigger, c.gold)].append(c)

    for (trigger, gold), pcands in periodic_by_key.items():
        # For periodic patterns, day and hour_bin are already fixed by the
        # trigger. Only minimize location and movement.
        PERIODIC_DIMS = ("location", "movement")

        # Count occurrences per (dim, value) — since all candidates share the
        # same gold, we just count how many candidates have each value.
        dim_value_counts: dict[str, Counter] = defaultdict(Counter)
        for c in pcands:
            for dim in PERIODIC_DIMS:
                dim_value_counts[dim][getattr(c.ctx, dim)] += 1

        # Keep values that appear often enough (≥ DOMINANCE_RATIO of max count).
        ctx_kept: dict[str, frozenset[str]] = {}
        for dim in PERIODIC_DIMS:
            vc = dim_value_counts[dim]
            if not vc:
                continue
            vals = set(vc.keys())
            if len(vals) == 1:
                ctx_kept[dim] = frozenset(vals)
            else:
                max_count = max(vc.values())
                threshold = max(1, int(max_count * DOMINANCE_RATIO))
                dominant_vals = frozenset(v for v, cnt in vc.items() if cnt >= threshold)
                if dominant_vals and dominant_vals != frozenset(vals):
                    ctx_kept[dim] = dominant_vals
                elif len(dominant_vals) <= 3:
                    ctx_kept[dim] = dominant_vals

        scope_cands = [
            c for c in pcands
            if all(getattr(c.ctx, dim) in V for dim, V in ctx_kept.items())
        ]
        occurrence = len(scope_cands)

        # Confidence for periodic: occurrence / total number of time-slot entries
        # We approximate by counting unique days the app appeared in the slot.
        # For simplicity, use occurrence as-is; confidence will be refined
        # against all candidates with the same trigger in E.2.
        ctx_dict: dict[str, list[str]] = {
            dim: sorted(V) for dim, V in ctx_kept.items()
        }

        p = Pattern(
            trigger=trigger, gold=gold, is_context_only=False,
            is_periodic=True, ctx=ctx_dict, candidates=scope_cands,
        )
        p.occurrence = occurrence
        # Confidence will be properly computed below
        patterns.append(p)

    # ---- Compute confidence for periodic patterns ----
    # For periodic patterns with the same trigger, confidence =
    # P(gold | trigger, ctx) against all periodic candidates with that trigger.
    periodic_by_trigger: dict[str, list[Pattern]] = defaultdict(list)
    for p in patterns:
        if p.is_periodic:
            periodic_by_trigger[p.trigger].append(p)

    # Also need to count total slot-entries for denominator.
    # We'll use a simpler approach: confidence = occurrence / total_periodic_for_trigger
    for trigger, plist in periodic_by_trigger.items():
        total = sum(p.occurrence for p in plist)
        for p in plist:
            p.confidence = p.occurrence / total if total else 0.0

    # ---- Sequential candidates ----
    seq_by_trigger: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        if not c.is_context_only and not c.is_periodic:
            seq_by_trigger[c.trigger].append(c)

    for trigger, all_cands in seq_by_trigger.items():
        dominant_set_per_dv: dict[tuple[str, str], set[str]] = {}
        seen_values_per_dim: dict[str, set[str]] = defaultdict(set)
        raw_counts_per_dv: dict[tuple[str, str], dict[str, int]] = {}
        for dim in DIMS:
            by_v_gold: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for c in all_cands:
                by_v_gold[getattr(c.ctx, dim)][c.gold] += 1
            for v, gold_counts in by_v_gold.items():
                seen_values_per_dim[dim].add(v)
                max_count = max(gold_counts.values())
                threshold = max(1, int(max_count * DOMINANCE_RATIO))
                dom_set = {g for g, cnt in gold_counts.items() if cnt >= threshold}
                dominant_set_per_dv[(dim, v)] = dom_set
                raw_counts_per_dv[(dim, v)] = dict(gold_counts)

        unique_dom_golds: set[str] = set()
        for dom_set in dominant_set_per_dv.values():
            unique_dom_golds.update(dom_set)

        all_golds_for_trigger: set[str] = {c.gold for c in all_cands}

        for G in unique_dom_golds:
            ctx_kept: dict[str, frozenset[str]] = {}
            for dim in DIMS:
                V_G = frozenset(
                    v for v in seen_values_per_dim[dim]
                    if G in dominant_set_per_dv.get((dim, v), set())
                )
                V_T = seen_values_per_dim[dim]
                if V_G != V_T and len(V_G) > 0:
                    ctx_kept[dim] = V_G

            scope_cands = [
                c for c in all_cands
                if all(getattr(c.ctx, dim) in V for dim, V in ctx_kept.items())
            ]
            gold_cands = [c for c in scope_cands if c.gold == G]
            if not gold_cands:
                continue
            occurrence = len(gold_cands)
            scope_total = len(scope_cands)
            confidence = occurrence / scope_total if scope_total else 0.0

            ctx_dict: dict[str, list[str]] = {
                dim: sorted(V) for dim, V in ctx_kept.items()
            }
            p = Pattern(
                trigger=trigger, gold=G, is_context_only=False,
                ctx=ctx_dict, candidates=gold_cands,
            )
            p.occurrence = occurrence
            p.confidence = confidence
            patterns.append(p)

        non_dom_golds = all_golds_for_trigger - unique_dom_golds
        for G in non_dom_golds:
            gold_cands = [c for c in all_cands if c.gold == G]
            total = len(all_cands)
            confidence = len(gold_cands) / total if total else 0.0
            if len(gold_cands) >= 3 and confidence >= 0.1:
                p = Pattern(
                    trigger=trigger, gold=G, is_context_only=False,
                    ctx={}, candidates=gold_cands,
                )
                p.occurrence = len(gold_cands)
                p.confidence = confidence
                patterns.append(p)

    # ---- Context-onset: ctx = UNCHANGED dims only ----
    onset_buckets: dict[tuple[str, tuple], dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        if not c.is_context_only:
            continue
        unchanged_pairs = tuple(sorted(
            (d, getattr(c.ctx, d)) for d in DIMS if d not in c.changed_dims
        ))
        onset_buckets[(c.trigger, unchanged_pairs)][c.gold].append(c)

    for (trigger, ctx_tuple), gold_to_cands in onset_buckets.items():
        denom = sum(len(cs) for cs in gold_to_cands.values())
        for gold, gcands in gold_to_cands.items():
            occurrence = len(gcands)
            confidence = occurrence / denom if denom else 0.0
            p = Pattern(
                trigger=trigger, gold=gold, is_context_only=True,
                ctx=dict(ctx_tuple), candidates=gcands,
            )
            p.occurrence = occurrence
            p.confidence = confidence
            patterns.append(p)

    return patterns


# ---------------------------------------------------------------------------
# Step E.2 — compute occurrence/confidence and filter
# ---------------------------------------------------------------------------

def compute_and_filter(patterns: list[Pattern],
                       min_occurrence: int,
                       min_confidence: float) -> list[Pattern]:
    """E.2 — filter patterns whose precomputed occurrence/confidence pass
    the thresholds."""
    return [
        p for p in patterns
        if p.occurrence >= min_occurrence and p.confidence >= min_confidence
    ]


# ---------------------------------------------------------------------------
# Step F — shifted detection
# ---------------------------------------------------------------------------

def iso_week(t: datetime) -> int:
    return t.isocalendar()[1]


def split_halves(patterns: list[Pattern],
                 week_split: int) -> None:
    """Annotate weeks_seen for each pattern."""
    for p in patterns:
        weeks = sorted({iso_week(c.t) for c in p.candidates})
        p.weeks_seen = weeks


def detect_shifted(patterns: list[Pattern],
                   week_split: int,
                   shifted_ctx_dims_min: int,
                   min_occurrence: int) -> tuple[list[Pattern], list[Pattern]]:
    """Return (shifted_old_new_pairs, remaining)."""
    by_trigger: dict[str, list[Pattern]] = defaultdict(list)
    for p in patterns:
        by_trigger[p.trigger].append(p)

    shifted_set: set[int] = set()
    shifted_results: list[Pattern] = []
    half_thresh = max(1, min_occurrence // 2)

    for trigger, plist in by_trigger.items():
        for p in plist:
            p_pre = sum(1 for c in p.candidates if iso_week(c.t) < week_split)
            p_post = sum(1 for c in p.candidates if iso_week(c.t) >= week_split)
            p._pre = p_pre  # type: ignore[attr-defined]
            p._post = p_post  # type: ignore[attr-defined]

        for p_old in plist:
            if id(p_old) in shifted_set:
                continue
            if p_old._pre < half_thresh:  # type: ignore[attr-defined]
                continue
            for p_new in plist:
                if p_new is p_old or id(p_new) in shifted_set:
                    continue
                if p_old.gold == p_new.gold:
                    continue
                if p_new._post < half_thresh:  # type: ignore[attr-defined]
                    continue
                shared = sum(1 for k in p_old.ctx
                             if k in p_new.ctx and p_old.ctx[k] == p_new.ctx[k])
                if shared < shifted_ctx_dims_min:
                    continue
                p_old.pattern_category = "shifted_old"
                p_new.pattern_category = "shifted_new"
                shifted_set.add(id(p_old))
                shifted_set.add(id(p_new))
                shifted_results.append(p_old)
                shifted_results.append(p_new)
                break

    remaining = [p for p in patterns if id(p) not in shifted_set]
    return shifted_results, remaining


# ---------------------------------------------------------------------------
# Step G — temporal categorization
# ---------------------------------------------------------------------------

def categorize(patterns: list[Pattern],
               week_split: int,
               min_occurrence: int,
               use_emergent: bool,
               use_decaying: bool) -> list[Pattern]:
    """Assign pattern_category to each non-shifted pattern."""
    half_thresh = max(1, min_occurrence // 2)
    out: list[Pattern] = []
    for p in patterns:
        if p.pattern_category.startswith("shifted"):
            out.append(p)
            continue
        pre = sum(1 for c in p.candidates if iso_week(c.t) < week_split)
        post = sum(1 for c in p.candidates if iso_week(c.t) >= week_split)
        total = pre + post
        if pre >= half_thresh and post >= half_thresh:
            p.pattern_category = "always"
            out.append(p)
        elif total >= min_occurrence:
            p.pattern_category = "transient"
            out.append(p)
        elif use_emergent and pre == 0 and post >= min_occurrence:
            p.pattern_category = "emergent"
            out.append(p)
        elif use_decaying and post == 0 and pre >= min_occurrence:
            p.pattern_category = "decaying"
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Step H — sample generation
# ---------------------------------------------------------------------------

def render_ctx(ctx) -> str:
    """Render context dims as a human-readable summary."""
    if not ctx:
        return "always"
    parts: list[str] = []
    for dim, val in ctx.items():
        if isinstance(val, (list, tuple, frozenset, set)):
            vals = list(val)
            if len(vals) == 1:
                parts.append(f"{dim}={vals[0]}")
            else:
                parts.append(f"{dim} ∈ {{{', '.join(map(str, vals))}}}")
        else:
            parts.append(f"{dim}={val}")
    return ", ".join(parts)


def slice_history(times: list[datetime], lines: list[str],
                  end_t: datetime, window_days: int) -> list[str]:
    """Return abstracted log lines (with time prefix) within the last N days
    up to (exclusive of) end_t."""
    lo = end_t - timedelta(days=window_days)
    lo_idx = bisect.bisect_left(times, lo)
    hi_idx = bisect.bisect_left(times, end_t)
    out = []
    for k in range(lo_idx, hi_idx):
        t_str = times[k].strftime(TIME_FMT)
        out.append(f"{t_str} | {lines[k]}")
    return out


def make_sample(p: Pattern, c: Candidate,
                times: list[datetime], lines: list[str],
                window_days: int) -> dict:
    """Build one training sample dict for one occurrence of pattern `p`."""
    history = slice_history(times, lines, c.t, window_days)
    history_block = "\n".join(history) if history else "(no recent activity)"

    sample_ctx_text = render_ctx(c.ctx.as_dict())
    pattern_ctx_text = render_ctx(p.ctx)

    parts = [
        f"[User Log History — last {window_days} days, oldest → newest]",
        history_block,
        "[/User Log History]",
        "",
    ]
    parts += ["[Context]", sample_ctx_text, "[/Context]", ""]
    parts += [
        f"[Trigger @ {c.t.strftime(TIME_FMT)}]",
        p.trigger,
        "[/Trigger]",
        "",
    ]
    parts.append(
        "What is user's the most probable next action considering user log history and the current context?"
    )

    input_text = "\n".join(parts)

    if p.is_periodic:
        scope = pattern_ctx_text if pattern_ctx_text != "always" else "any context"
        reasoning = (
            f"In abstracted history (4 weeks), {p.occurrence}× during "
            f"'{p.trigger}' under {scope}, the user does '{p.gold}' "
            f"(confidence {p.confidence:.2f})."
        )
    elif p.is_context_only:
        reasoning = (
            f"In abstracted history (4 weeks), {p.occurrence}× upon entering "
            f"context [{pattern_ctx_text}], the user does '{p.gold}' "
            f"(confidence {p.confidence:.2f})."
        )
    else:
        scope = pattern_ctx_text if pattern_ctx_text != "always" else "any context"
        reasoning = (
            f"In abstracted history (4 weeks), {p.occurrence}× after "
            f"'{p.trigger}' under {scope}, the user does '{p.gold}' "
            f"(confidence {p.confidence:.2f})."
        )

    return {
        "input_text": input_text,
        "gold_answer": p.gold,
        "gold_reasoning": reasoning,
        "_meta": {
            "pattern_id": p.pattern_id,
            "pattern_category": p.pattern_category,
            "occurrence": p.occurrence,
            "confidence": round(p.confidence, 3),
            "weeks_seen": p.weeks_seen,
            "trigger_time": c.t.strftime(TIME_FMT),
            "sample_ctx": sample_ctx_text,
            "pattern_ctx": pattern_ctx_text,
            "is_context_only": p.is_context_only,
            "is_periodic": p.is_periodic,
        },
    }


def strip_meta(sample: dict) -> dict:
    """Return a copy without `_meta`."""
    return {k: v for k, v in sample.items() if k != "_meta"}


# ---------------------------------------------------------------------------
# Step I — splits
# ---------------------------------------------------------------------------

def regular_split(patterns: list[Pattern],
                  times: list[datetime], lines: list[str],
                  window_days: int,
                  valid_frac: float = 0.05,
                  test_frac: float = 0.15,
                  seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Regular split with the 'prior > occurrence//2' rule."""
    rng = random.Random(seed)
    train, valid, test = [], [], []
    for p in patterns:
        cands = sorted(p.candidates, key=lambda c: c.t)
        N = len(cands)
        thresh = N // 2
        eligible = [(r, c) for r, c in enumerate(cands) if r > thresh]
        rng.shuffle(eligible)
        n_e = len(eligible)
        n_test = int(round(n_e * test_frac))
        n_valid = int(round(n_e * valid_frac))
        for k, (r, c) in enumerate(eligible):
            sample = make_sample(p, c, times, lines, window_days)
            if k < n_test:
                test.append(sample)
            elif k < n_test + n_valid:
                valid.append(sample)
            else:
                train.append(sample)
    return train, valid, test


def continual_split(patterns: list[Pattern],
                    times: list[datetime], lines: list[str],
                    window_days: int,
                    week_split: int,
                    seed: int = 42) -> dict:
    """Continual split per plan rev 2/3."""
    rng = random.Random(seed)
    p1_train, p1_test = [], []
    p2_train = []
    p2_test = {"always": [], "shifted_new": [], "emergent": [], "decaying": [], "transient": []}

    for p in patterns:
        cat = p.pattern_category
        pre = [c for c in p.candidates if iso_week(c.t) < week_split]
        post = [c for c in p.candidates if iso_week(c.t) >= week_split]

        def split_90_10(cs):
            cs2 = list(cs)
            rng.shuffle(cs2)
            n_test = max(1, int(round(len(cs2) * 0.10))) if cs2 else 0
            return cs2[n_test:], cs2[:n_test]

        if cat == "always":
            tr1, ts1 = split_90_10(pre)
            tr2, ts2 = split_90_10(post)
            p1_train += [make_sample(p, c, times, lines, window_days) for c in tr1]
            p1_test  += [make_sample(p, c, times, lines, window_days) for c in ts1]
            p2_train += [make_sample(p, c, times, lines, window_days) for c in tr2]
            p2_test["always"] += [make_sample(p, c, times, lines, window_days) for c in ts2]
        elif cat == "shifted_old":
            tr1, ts1 = split_90_10(pre)
            p1_train += [make_sample(p, c, times, lines, window_days) for c in tr1]
            p1_test  += [make_sample(p, c, times, lines, window_days) for c in ts1]
        elif cat == "shifted_new":
            tr2, ts2 = split_90_10(post)
            p2_train += [make_sample(p, c, times, lines, window_days) for c in tr2]
            p2_test["shifted_new"] += [make_sample(p, c, times, lines, window_days) for c in ts2]
        elif cat == "transient":
            tr1, ts1 = split_90_10(pre) if pre else ([], [])
            tr2, ts2 = split_90_10(post) if post else ([], [])
            p1_train += [make_sample(p, c, times, lines, window_days) for c in tr1]
            p1_test  += [make_sample(p, c, times, lines, window_days) for c in ts1]
            p2_train += [make_sample(p, c, times, lines, window_days) for c in tr2]
            p2_test.setdefault("transient", []).extend(
                [make_sample(p, c, times, lines, window_days) for c in ts2]
            )
        elif cat == "decaying":
            cs = list(pre)
            rng.shuffle(cs)
            n10 = max(1, int(round(len(cs) * 0.10))) if cs else 0
            ts1 = cs[:n10]
            ts2 = cs[n10:n10 * 2]
            tr1 = cs[n10 * 2:]
            p1_train += [make_sample(p, c, times, lines, window_days) for c in tr1]
            p1_test  += [make_sample(p, c, times, lines, window_days) for c in ts1]
            p2_test["decaying"] += [make_sample(p, c, times, lines, window_days) for c in ts2]
        elif cat == "emergent":
            tr2, ts2 = split_90_10(post)
            p2_train += [make_sample(p, c, times, lines, window_days) for c in tr2]
            p2_test["emergent"] += [make_sample(p, c, times, lines, window_days) for c in ts2]

    return {
        "phase1_train": p1_train,
        "phase1_test":  p1_test,
        "phase2_train": p2_train,
        "phase2_test":  p2_test,
    }


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="user_usage_4weeks_3_abstracted.json")
    p.add_argument("--output", default="user_behavior_samples_planeat.json")

    # ----- IMPORTANT pattern-shaping thresholds -----
    p.add_argument(
        "--min_occurrence", type=int, default=3,
        help="Minimum occurrences per pattern.",
    )
    p.add_argument(
        "--min_confidence", type=float, default=0.2,
        help="Minimum P(gold | trigger, context).",
    )
    p.add_argument(
        "--delta_t_min", type=int, default=3,
        help="Δt window in minutes.",
    )

    # ----- Periodic pattern thresholds -----
    p.add_argument(
        "--periodic_concentration_min", type=float, default=0.15,
        help="Minimum fraction of total usage that a single hour_bin must "
             "have to be selected as part of a periodic pattern.",
    )
    p.add_argument(
        "--periodic_combined_min_concentration", type=float, default=0.70,
        help="Minimum combined fraction of total usage across all selected "
             "hour_bins. Ensures the app is genuinely time-driven, not spread "
             "across all hours.",
    )
    p.add_argument(
        "--periodic_min_slot_occurrence", type=int, default=3,
        help="Minimum number of times an app must appear in a time-slot "
             "for a periodic pattern to be created.",
    )

    # ----- secondary knobs -----
    p.add_argument("--stm_window_days", type=int, default=0)
    p.add_argument("--week_split_phase", type=int, default=17)
    p.add_argument("--shifted_ctx_dims_min", type=int, default=2)
    p.add_argument("--use_emergent", action="store_true")
    p.add_argument("--use_decaying", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    in_path = args.input if os.path.isabs(args.input) else os.path.join(repo_root, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(repo_root, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- A: parse ----
    events_full = parse_log(in_path)
    print(f"[A] parsed {len(events_full)} events from {os.path.basename(in_path)}")

    # ---- C: coalesce (BEFORE B) ----
    events_full = coalesce_repeats(events_full)
    print(f"[C] {len(events_full)} events after coalesce repeats (≥2 within 2 min)")

    # ---- D: assign contexts on UNFILTERED events ----
    contexts_full = assign_contexts(events_full)
    print(f"[D] context dims tracked over full timeline ({len(events_full)} events)")

    # ---- B: drop low-frequency events ----
    counts = Counter(line for _, line in events_full)
    keep = [counts[line] >= args.min_occurrence for _, line in events_full]
    events = [e for e, k in zip(events_full, keep) if k]
    contexts = [c for c, k in zip(contexts_full, keep) if k]
    print(f"[B] {len(events)} events after dropping count<{args.min_occurrence} "
          f"({len(set(line for _, line in events))} unique lines)")

    # ---- E: candidates ----
    candidates = mine_candidates(
        events, contexts,
        delta_t_min=args.delta_t_min,
        events_full=events_full,
        periodic_concentration_min=args.periodic_concentration_min,
        periodic_combined_min_concentration=args.periodic_combined_min_concentration,
        periodic_min_slot_occurrence=args.periodic_min_slot_occurrence,
    )
    n_seq = sum(1 for c in candidates if not c.is_context_only and not c.is_periodic)
    n_onset = sum(1 for c in candidates if c.is_context_only)
    n_periodic = sum(1 for c in candidates if c.is_periodic)
    print(f"[E] {len(candidates)} candidate instances mined "
          f"(sequential={n_seq}, context-onset={n_onset}, periodic={n_periodic})")

    # ---- E.1: minimum context per (trigger, gold) ----
    patterns = minimize_context(candidates)
    n_periodic_p = sum(1 for p in patterns if p.is_periodic)
    print(f"[E.1] {len(patterns)} patterns (periodic={n_periodic_p})")

    # ---- E.2: compute occ/conf, filter ----
    patterns = compute_and_filter(patterns, args.min_occurrence, args.min_confidence)
    n_periodic_p = sum(1 for p in patterns if p.is_periodic)
    print(f"[E.2] {len(patterns)} patterns pass filters (periodic={n_periodic_p})")

    # Annotate weeks_seen
    split_halves(patterns, args.week_split_phase)

    # ---- F: shifted detection ----
    shifted, remaining = detect_shifted(
        patterns, args.week_split_phase, args.shifted_ctx_dims_min, args.min_occurrence,
    )
    print(f"[F] detected {len(shifted)//2} shifted pairs ({len(shifted)} entries)")

    # ---- G: categorize remaining ----
    remaining = categorize(
        remaining, args.week_split_phase, args.min_occurrence,
        use_emergent=args.use_emergent, use_decaying=args.use_decaying,
    )
    final_patterns: list[Pattern] = shifted + remaining

    # Assign IDs
    for i, pat in enumerate(final_patterns):
        pat.pattern_id = f"P_{i:04d}"

    cat_counts = Counter(p.pattern_category for p in final_patterns)
    print(f"[G] kept {len(final_patterns)} patterns by category: {dict(cat_counts)}")

    # ---- H + I: samples + splits ----
    times = [t for t, _ in events]
    lines_only = [line for _, line in events]

    train, valid, test = regular_split(
        final_patterns, times, lines_only, args.stm_window_days,
        seed=args.seed,
    )
    print(f"[I.regular] train={len(train)}, valid={len(valid)}, test={len(test)}")

    continual = continual_split(
        final_patterns, times, lines_only, args.stm_window_days,
        week_split=args.week_split_phase, seed=args.seed,
    )
    print(f"[I.continual] phase1_train={len(continual['phase1_train'])}, "
          f"phase1_test={len(continual['phase1_test'])}, "
          f"phase2_train={len(continual['phase2_train'])}, "
          f"phase2_test={ {k: len(v) for k,v in continual['phase2_test'].items()} }")

    # Pattern metadata for analysis
    pattern_meta = [
        {
            "pattern_id": p.pattern_id,
            "trigger": p.trigger,
            "gold": p.gold,
            "context": p.ctx,
            "is_context_only": p.is_context_only,
            "is_periodic": p.is_periodic,
            "occurrence": p.occurrence,
            "confidence": round(p.confidence, 3),
            "weeks_seen": p.weeks_seen,
            "category": p.pattern_category,
        }
        for p in final_patterns
    ]

    out = {
        "train":    [strip_meta(s) for s in train],
        "valid":    [strip_meta(s) for s in valid],
        "test":     [strip_meta(s) for s in test],
        "continual": {
            "phase1_train": [strip_meta(s) for s in continual["phase1_train"]],
            "phase1_test":  [strip_meta(s) for s in continual["phase1_test"]],
            "phase2_train": [strip_meta(s) for s in continual["phase2_train"]],
            "phase2_test":  {k: [strip_meta(s) for s in v]
                             for k, v in continual["phase2_test"].items()},
        },
        "patterns": pattern_meta,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote samples to {out_path}")

    # Top-15 gold answer distribution in train
    print("\nTop-15 train gold answers:")
    c = Counter(s["gold_answer"] for s in train)
    n_total = max(1, len(train))
    for g, n in c.most_common(15):
        print(f"  {n:4d} ({100*n/n_total:5.1f}%)  {g}")

    # Periodic pattern summary
    periodic_patterns = [p for p in final_patterns if p.is_periodic]
    if periodic_patterns:
        print(f"\n--- Periodic patterns ({len(periodic_patterns)}) ---")
        for pp in periodic_patterns:
            print(f"  {pp.pattern_id}: trigger='{pp.trigger}', gold='{pp.gold}', "
                  f"occ={pp.occurrence}, conf={pp.confidence:.2f}, "
                  f"ctx={pp.ctx}, cat={pp.pattern_category}")


if __name__ == "__main__":
    main()
