"""
4_extract.py — mine user-preference patterns from the abstracted log and
emit training samples to ``data/user_behavior_samples.json``.

Pipeline (per plan rev 3):

  Step A  — parse + sort
  Step C  — coalesce repeats (≥2 same line within 2 min)        [moved before B]
  Step B  — drop events with count < min_occurrence
  Step D  — assign context (day, hour_bin, location, movement)
  Step E  — mine candidate patterns
            (a) sequential: every event Y in (t_X − Δt, t_X) for each X
            (b) context-onset: when context changes, emit virtual onset and
                pair with all events within Δt after the change
  Step E.1 — minimum context per (trigger, gold) group: keep only the dims
            whose value is identical across all candidates in the group
  Step E.2 — compute occurrence and confidence; filter by thresholds
  Step F  — detect shifted patterns (≥shifted_ctx_dims_min context dims match,
            different gold between halves)
  Step G  — temporal categorization (always / shifted; emergent/decaying opt-in)
  Step H  — generate samples (input_text, gold_answer, gold_reasoning)
  Step I  — splits (regular + continual)

Output JSON schema:
  {
    "train": [...],
    "valid": [...],
    "test":  [...],
    "continual": {
        "phase1_train": [...], "phase1_test": [...],
        "phase2_train": [...], "phase2_test": {always:[], shifted_new:[],
                                              emergent:[], decaying:[]}
    },
    "patterns": [pattern_meta, ...]   # for analysis
  }

Only `train`/`valid`/`test` are read by the current `tasks/user_behavior.py`;
the other keys are present for future continual-learning task code.
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
# transitions, not deliberate user actions. (Applied only to sequential
# candidates; context-onset candidates keep them since they characterize the
# environment that follows a context change.)
SEQUENTIAL_GOLD_BLOCKLIST = (
    "Begin walking", "Stop walking",
    "Begin running", "Stop running",
    "Board a vehicle", "Get off vehicle",
    "Cellular network connected", "Cellular network disconnected",
    "Wi-Fi connected", "Wi-Fi disconnected",
    "Network connected", "Network disconnected",
    "Day starts:",
)


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
        # Movement state from "Begin walking" / "Stop walking" / etc.
        for prefix, (state, kind) in _MOVEMENT_PATTERNS.items():
            if line.startswith(prefix):
                movement_state = state if kind == "start" else "stationary"
                break
        # Location from "Move to Location N"
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
    # For context-onset candidates: which ctx dims just changed at onset_t.
    # Empty for sequential candidates.
    changed_dims: tuple = ()


def _enumerate_context_change_points(
    events: list[tuple[datetime, str]],
) -> list[tuple[datetime, Context, tuple[str, ...]]]:
    """Walk a merged timeline of (event times) ∪ (hour_bin boundaries) and
    emit (time, new_context, changed_dims) at every moment the 4-dim context
    changes. ``changed_dims`` is the subset of dims whose value differs from
    the previous context (so a downstream consumer can describe the trigger
    as "what just changed", and treat the unchanged dims as the surrounding
    context).

    The state evolves as follows:
      - location: updated when an event line is "Move to <Location N>"
      - movement: updated by Begin/Stop walking/running, Board/Get off vehicle
      - day, hour_bin: updated by the timestamp itself

    Hour-bin boundary times (07/09/11/13/15/18/20/22 + midnights) are
    inserted as virtual no-op events so we detect context flips that happen
    mid-gap (e.g., 'lunch' → 'afternoon' at 13:00 even with no event).
    """
    if not events:
        return []
    log_start = events[0][0]
    log_end = events[-1][0]

    # Build virtual boundary times within [log_start, log_end].
    boundary_times: list[datetime] = []
    cur_day = log_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur_day <= log_end:
        for h in _HOUR_BIN_BOUNDARIES:
            t = cur_day.replace(hour=h)
            if log_start <= t <= log_end:
                boundary_times.append(t)
        cur_day += timedelta(days=1)

    # Merge events and boundary times in chronological order.
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
            # update state from event
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
                changed = DIMS  # first onset: treat all dims as initialized
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
                    ) -> list[Candidate]:
    """Step E: emit (trigger, gold, context, time) candidates.

    (a) Sequential: for every event X at index i, every event Y in
        (t_X − Δt, t_X) becomes a candidate (Y, X). Apply
        SEQUENTIAL_GOLD_BLOCKLIST to drop X's that are passive state
        transitions (Begin/Stop walking, network connected, etc.).

    (b) Context-onset: starting from the initial log context, whenever ANY
        ctx dim changes (location/movement from events; hour_bin/day from
        time), emit a virtual onset at the change time. Every event in
        [onset_t, onset_t + Δt] becomes a gold (no blocklist applied).
    """
    delta = timedelta(minutes=delta_t_min)
    n = len(events)
    candidates: list[Candidate] = []

    # ------- (a) sequential — gold blocklist applies -------
    j_lo = 0
    for i in range(n):
        t_i, line_i = events[i]
        if _is_blocked_sequential_gold(line_i):
            continue  # skip — passive state transitions are not learnable golds
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
    # "바뀐 후 10분 이내" — strictly AFTER onset_t. Use bisect_right(onset_t)
    # so the event that caused the change (location move / movement state
    # change at exactly onset_t) is excluded from being its own gold.
    # State tracking uses the FULL (unfiltered) event timeline to capture
    # location/movement transitions even when their lines are too infrequent
    # to pass step B; gold candidates still come only from filtered events.
    onsets = _enumerate_context_change_points(
        events_full if events_full is not None else events
    )
    times = [t for t, _ in events]
    for onset_t, onset_ctx, changed_dims in onsets:
        # Trigger string: just the dims that just changed and their new values.
        # Unchanged dims are recorded as the pattern's surrounding context.
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

    Conceptual flow:

      Initial state: candidates are already micro-grouped by (trigger, full_ctx,
      gold) — each unique combination is one micro-pattern. Within any single
      micro-pattern, gold is trivially "dominant" (it's the only gold there).
      So the user's rule (1) — split when gold changes by context — is
      already satisfied just by grouping.

      What remains: for each (trigger, gold) pair, find the *minimum context*
      that distinguishes this gold from other golds appearing for the same
      trigger. That is, identify which context dim values are responsible for
      this gold being the predicted follow-up of trigger.

    For each trigger T:
      1. Compute ``dominant_gold[(dim, v)]`` = argmax_g count(T, dim=v, gold=g),
         for every (dim, v) seen in candidates with trigger=T.
      2. For every gold G that appears as a dominant gold for some (dim, v):
         - For each dim D, collect V_G_D = values of D where G dominates.
         - If V_G_D equals the full set of D values seen for T → D doesn't
           distinguish G (G dominates everywhere on this dim) → drop D.
         - Else → keep D with the values in V_G_D as the pattern's context
           constraint.
      3. The pattern is (T, kept_ctx, G). Its candidates = trigger=T, gold=G,
         AND for every kept dim D, ctx.D ∈ V_G_D. The denominator (for
         confidence) = trigger=T candidates whose ctx is in scope of kept_ctx.

    Special case: if no dim is informative for G (V_G_D = V_T_D for every dim),
    the pattern's context is empty — display as ``always``.

    Confidence and occurrence are computed inline since the kept ctx is
    multi-valued and can't be re-keyed cleanly in a separate E.2 step. The
    threshold filter (E.2) just tests the precomputed values.

    **Context-onset candidates bypass E.1**: their context IS the trigger; each
    unique (CONTEXT_ONSET, full_ctx, gold) becomes its own pattern with
    confidence = P(gold | onset, full_ctx).
    """
    DIMS = ("day", "hour_bin", "location", "movement")
    patterns: list[Pattern] = []

    # ---- Sequential candidates ----
    seq_by_trigger: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        if not c.is_context_only:
            seq_by_trigger[c.trigger].append(c)

    for trigger, all_cands in seq_by_trigger.items():
        # Step 1: per-(dim, value) dominant gold
        dominant_per_dv: dict[tuple[str, str], str] = {}
        seen_values_per_dim: dict[str, set[str]] = defaultdict(set)
        for dim in DIMS:
            by_v_gold: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for c in all_cands:
                by_v_gold[getattr(c.ctx, dim)][c.gold] += 1
            for v, gold_counts in by_v_gold.items():
                seen_values_per_dim[dim].add(v)
                dom = max(gold_counts.items(), key=lambda kv: kv[1])[0]
                dominant_per_dv[(dim, v)] = dom

        # Step 2: each unique dominant gold becomes a candidate pattern
        unique_dom_golds = {g for g in dominant_per_dv.values()}

        for G in unique_dom_golds:
            # For each dim, collect values where G dominates
            ctx_kept: dict[str, frozenset[str]] = {}
            for dim in DIMS:
                V_G = frozenset(
                    v for v in seen_values_per_dim[dim]
                    if dominant_per_dv.get((dim, v)) == G
                )
                V_T = seen_values_per_dim[dim]
                if V_G != V_T and len(V_G) > 0:
                    ctx_kept[dim] = V_G
                # else: G dominates at every value of dim (drop D), or G never
                # dominates on dim (no info to distinguish — also dropped).

            # Step 3: compute occurrence within kept_ctx scope, but confidence
            # is computed GLOBALLY for the trigger — P(gold | trigger),
            # regardless of ctx_kept. This filters patterns by how reliably
            # gold follows trigger overall (per the user's spec). Ctx_kept
            # determines which candidates are samples for this pattern, not
            # the denominator of confidence.
            scope_cands = [
                c for c in all_cands
                if all(getattr(c.ctx, dim) in V for dim, V in ctx_kept.items())
            ]
            gold_cands = [c for c in scope_cands if c.gold == G]
            if not gold_cands:
                continue
            occurrence = len(gold_cands)
            global_gold_count = sum(1 for c in all_cands if c.gold == G)
            global_total = len(all_cands)
            confidence = global_gold_count / global_total if global_total else 0.0

            # Materialize ctx as plain dict[str, list[str]] for JSON serialization
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

    # ---- Context-onset: ctx = UNCHANGED dims only (changed dims already
    # encoded in the trigger string). ----
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
    the thresholds. Occurrence and confidence are set during E.1 because
    the kept ctx may be multi-valued and the denominator must be computed
    against the in-scope candidate set (not just other patterns sharing the
    same ctx-key)."""
    return [
        p for p in patterns
        if p.occurrence >= min_occurrence and p.confidence >= min_confidence
    ]


def _compute_and_filter_legacy_unused(patterns, min_occurrence, min_confidence):
    """For each pattern, occurrence = len(candidates).

    Confidence = occurrence / sum_g occurrence(same-trigger, same-context, g).
    Drop patterns failing either threshold.
    """
    # Group patterns by (trigger, frozenset(ctx items)) to compute confidence.
    keyed: dict[tuple[str, frozenset], list[Pattern]] = defaultdict(list)
    for p in patterns:
        ctx_key = frozenset(p.ctx.items())
        keyed[(p.trigger, ctx_key)].append(p)

    out: list[Pattern] = []
    for key, plist in keyed.items():
        total = sum(len(p.candidates) for p in plist)
        for p in plist:
            occ = len(p.candidates)
            conf = occ / total if total else 0.0
            p.occurrence = occ
            p.confidence = conf
            if occ >= min_occurrence and conf >= min_confidence:
                out.append(p)
    return out


# ---------------------------------------------------------------------------
# Step F — shifted detection
# ---------------------------------------------------------------------------

def iso_week(t: datetime) -> int:
    return t.isocalendar().week


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
    """Return (shifted_old_new_pairs, remaining).

    For each (trigger, gold-old) and (trigger, gold-new) where:
      - same trigger, gold-old != gold-new
      - gold-old has ≥ min_occurrence//2 occurrences in weeks < week_split
      - gold-new has ≥ min_occurrence//2 occurrences in weeks ≥ week_split
      - their contexts share ≥ shifted_ctx_dims_min dims with same value
    classify both as shifted_old / shifted_new.
    """
    # Index patterns by trigger
    by_trigger: dict[str, list[Pattern]] = defaultdict(list)
    for p in patterns:
        by_trigger[p.trigger].append(p)

    shifted_set: set[id] = set()
    shifted_results: list[Pattern] = []
    half_thresh = max(1, min_occurrence // 2)

    for trigger, plist in by_trigger.items():
        # Compute pre/post counts per pattern
        for p in plist:
            p_pre = sum(1 for c in p.candidates if iso_week(c.t) < week_split)
            p_post = sum(1 for c in p.candidates if iso_week(c.t) >= week_split)
            p._pre = p_pre  # type: ignore[attr-defined]
            p._post = p_post  # type: ignore[attr-defined]

        # Pair old (pre-only) with new (post-only) for shifted
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
                # ctx overlap
                shared = sum(1 for k in p_old.ctx
                             if k in p_new.ctx and p_old.ctx[k] == p_new.ctx[k])
                if shared < shifted_ctx_dims_min:
                    continue
                # Mark both
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
    """Assign pattern_category to each non-shifted pattern.

    - always   : both halves have occurrence ≥ min_occurrence // 2
    - emergent : (opt-in) zero in pre, ≥ min_occurrence in post
    - decaying : (opt-in) zero in post, ≥ min_occurrence in pre
    Patterns that don't fit any kept category are dropped.
    """
    half_thresh = max(1, min_occurrence // 2)
    out: list[Pattern] = []
    for p in patterns:
        if p.pattern_category.startswith("shifted"):
            out.append(p)
            continue
        pre = sum(1 for c in p.candidates if iso_week(c.t) < week_split)
        post = sum(1 for c in p.candidates if iso_week(c.t) >= week_split)
        if pre >= half_thresh and post >= half_thresh:
            p.pattern_category = "always"
            out.append(p)
        elif use_emergent and pre == 0 and post >= min_occurrence:
            p.pattern_category = "emergent"
            out.append(p)
        elif use_decaying and post == 0 and pre >= min_occurrence:
            p.pattern_category = "decaying"
            out.append(p)
        # else: dropped
    return out


# ---------------------------------------------------------------------------
# Step H — sample generation
# ---------------------------------------------------------------------------

def render_ctx(ctx) -> str:
    """Render context dims as a human-readable summary.

    Sequential patterns: ctx values can be a list of allowed values per dim
    (e.g., {"location": ["Location 1", "Location 3"]}); render as
    "location ∈ {Location 1, Location 3}" or just the value if list of length 1.

    Context-onset patterns: ctx values are scalars (e.g., {"day": "weekday"});
    render as "day=weekday".

    Empty ctx → "always".
    """
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
    """Build one training sample dict for one occurrence of pattern `p`.

    The [Context] block uses the OCCURRENCE's actual context (single-value,
    matching what inference-time input will look like), not the pattern's
    aggregated ctx. The pattern's discriminative ctx (multi-value lists) is
    surfaced via gold_reasoning text and stored in _meta for analysis.
    """
    history = slice_history(times, lines, c.t, window_days)
    history_block = "\n".join(history) if history else "(no recent activity)"

    # Sample [Context] block uses the candidate's actual ctx (single-value,
    # matches inference distribution). Pattern's ctx is metadata only.
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

    # Auto-templated reasoning — references the PATTERN's discriminative ctx
    # (multi-value possible) so the model sees the abstract preference signal.
    if p.is_context_only:
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
            "sample_ctx": sample_ctx_text,      # actual ctx in input
            "pattern_ctx": pattern_ctx_text,    # discriminative scope
            "is_context_only": p.is_context_only,
        },
    }


def strip_meta(sample: dict) -> dict:
    """Return a copy without `_meta` (UserBehaviorTask only reads 3 fields)."""
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
    """Regular split with the 'prior > occurrence//2' rule.

    For each pattern p with occurrence N:
      - sort its candidates by time
      - for each candidate at rank r (0-indexed), it's eligible iff r > N // 2
      - eligible samples are randomly assigned to train/valid/test
    """
    rng = random.Random(seed)
    train, valid, test = [], [], []
    for p in patterns:
        cands = sorted(p.candidates, key=lambda c: c.t)
        N = len(cands)
        thresh = N // 2  # need r > thresh, i.e., at least ceil(N/2)+1 prior
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
    """Continual split per plan rev 2/3: phase boundary at `week_split`.

    Returns:
      {
        "phase1_train": [...], "phase1_test": [...],
        "phase2_train": [...],
        "phase2_test": {always: [...], shifted_new: [...],
                        emergent: [...], decaying: [...]},
      }
    """
    rng = random.Random(seed)
    p1_train, p1_test = [], []
    p2_train = []
    p2_test = {"always": [], "shifted_new": [], "emergent": [], "decaying": []}

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
        elif cat == "decaying":
            # 80/10/10: phase1_train / phase1_test / phase2_test_decaying
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
    p.add_argument("--output", default="user_behavior_samples.json")

    # ----- IMPORTANT pattern-shaping thresholds -----
    p.add_argument(
        "--min_occurrence", type=int, default=3,
        help="[important] Minimum occurrences per pattern. HIGHER → each "
             "kept pattern has more training samples (one sample per "
             "occurrence), so each pattern is easier to learn (more support). "
             "LOWER → more patterns retained, but each may be sparsely "
             "supported. Raise this to make training easier.",
    )
    p.add_argument(
        "--min_confidence", type=float, default=0.20,
        help="[important] Minimum P(gold | trigger, context). HIGHER → "
             "patterns are more deterministic (less ambiguity in what gold "
             "follows), so easier to learn. LOWER → tolerate ambiguous "
             "patterns where the same trigger+context maps to many golds. "
             "Raise this to make training easier.",
    )
    p.add_argument(
        "--delta_t_min", type=int, default=10,
        help="[important] Δt window in minutes for sequential triggering and "
             "for context-onset → first-action coupling. LARGER → more "
             "candidate (trigger, gold) pairs per event, more sample "
             "diversity per pattern, easier learning. SMALLER → tighter "
             "chains, fewer candidates, sparser training.",
    )

    # ----- secondary knobs -----
    # Note: E.1 has no tunable threshold — it just compares dominant golds
    # across each dim's values. Confidence enters at E.2 (compute_and_filter).
    # Context-onset patterns bypass E.1 entirely and keep full context.
    p.add_argument("--stm_window_days", type=int, default=0)
    p.add_argument("--week_split_phase", type=int, default=17,
                   help="ISO week boundary: weeks <split = phase 1, ≥split = phase 2.")
    p.add_argument("--shifted_ctx_dims_min", type=int, default=2,
                   help="Min ctx dims that must agree for two patterns to be "
                        "considered the 'same context' for shifted detection.")
    p.add_argument("--use_emergent", action="store_true",
                   help="Include emergent (post-only) patterns. Default off.")
    p.add_argument("--use_decaying", action="store_true",
                   help="Include decaying (pre-only) patterns. Default off.")
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
    # Context (location/movement) is tracked on the full timeline so that
    # infrequent 'Moved to Location N' events that would otherwise be dropped
    # at step B still update state. This is critical: without it, location
    # is stuck at 'unknown' / stale.
    contexts_full = assign_contexts(events_full)
    print(f"[D] context dims tracked over full timeline ({len(events_full)} events)")

    # ---- B: drop low-frequency events (sync-filter contexts too) ----
    counts = Counter(line for _, line in events_full)
    keep = [counts[line] >= args.min_occurrence for _, line in events_full]
    events = [e for e, k in zip(events_full, keep) if k]
    contexts = [c for c, k in zip(contexts_full, keep) if k]
    print(f"[B] {len(events)} events after dropping count<{args.min_occurrence} "
          f"({len(set(line for _, line in events))} unique lines)")

    # ---- E: candidates (sequential = filtered events; onset = full timeline) ----
    candidates = mine_candidates(
        events, contexts,
        delta_t_min=args.delta_t_min,
        events_full=events_full,
    )
    print(f"[E] {len(candidates)} candidate (trigger, gold) instances mined "
          f"(Δt={args.delta_t_min} min)")

    # ---- E.1: minimum context per (trigger, gold) — majority-voting ----
    patterns = minimize_context(candidates)
    print(f"[E.1] {len(patterns)} (trigger, kept_ctx, gold) patterns "
          f"(drop dim if dominant-gold same across all its values)")

    # ---- E.2: compute occ/conf, filter ----
    patterns = compute_and_filter(patterns, args.min_occurrence, args.min_confidence)
    print(f"[E.2] {len(patterns)} patterns pass occurrence≥{args.min_occurrence} "
          f"AND confidence≥{args.min_confidence}")

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

    # Category counts
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


if __name__ == "__main__":
    main()
