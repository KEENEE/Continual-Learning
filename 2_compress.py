"""
Shrink lifelog JSON (e.g. user_usage_week2.json) for LLM prompts.

Usage:
  python compress_user_usage_json.py -i user_usage_week2.json -o user_usage_week2.compact.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

# "2026-04-13 12:08:37" -> "26-04-13 12:08:37" (two bytes saved per matching field).
_DATETIME_20YY = re.compile(r"^20\d{2}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$")


# Applied as the last in-memory step before json.dump (prompts / code above still use full names).
_DATA_TYPE_ABBREV: dict[str, str] = {
    "app_usage": "app",
    "notification": "noti",
}

# Not driven by explicit user intent on the device; safe to drop for behavior modeling.
_APP_DROP_BY_TYPES = frozenset(
    {
        "STANDBY_BUCKET_CHANGED", "CONFIGURATION_CHANGE",
        "SCREEN_INTERACTIVE", "SCREEN_NON_INTERACTIVE",
        "ACTIVITY_PAUSED", "ACTIVITY_STOPPED",
        "FOREGROUND_SERVICE_START", "FOREGROUND_SERVICE_STOP",
        "KEYGUARD_SHOWN", "KEYGUARD_HIDDEN",
        "NOTIFICATION_INTERRUPTION", "NOTIFICATION_SEEN",
        "TYPE_28", "TYPE_9", "SLICE_PINNED",
        "DEVICE_SHUTDOWN", "DEVICE_START", "DEVICE_STARTUP"
    }
)

def thin_consecutive_app_usage(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    i = 0
    n = len(records)
    while i < n:
        r = records[i]
        if r.get("data_type") != "app_usage":
            out.append(r)
            i += 1
            continue
        pkg = r.get("package")
        typ = r.get("type")
        j = i
        while (
            j < n
            and records[j].get("data_type") == "app_usage"
            and records[j].get("package") == pkg
            and records[j].get("type") == typ
        ):
            j += 1
        out.append(r)
        i = j
    return out


def thin_consecutive_connection(records: list[dict]) -> list[dict]:
    """Drop a connection row if (event_kind, summary) equals the previous
    connection row's, ignoring non-connection rows in between."""
    out: list[dict] = []
    prev_key: tuple | None = None
    for r in records:
        if r.get("data_type") != "connection":
            out.append(r)
            continue
        key = (r.get("event_kind"), r.get("summary"))
        if key == prev_key:
            continue
        out.append(r)
        prev_key = key
    return out


def dedup_user_interaction_after_resume(records: list[dict]) -> list[dict]:
    """Drop ``USER_INTERACTION`` rows whose immediately preceding record (in
    sequence) is an ``ACTIVITY_RESUMED`` of the same package — the resume
    already implies that the user is acting in that app, so the bare
    interaction event is redundant.
    """
    out: list[dict] = []
    for r in records:
        if (r.get("data_type") == "app_usage"
                and r.get("type") == "USER_INTERACTION"
                and out
                and out[-1].get("data_type") == "app_usage"
                and out[-1].get("type") == "ACTIVITY_RESUMED"
                and out[-1].get("package") == r.get("package")):
            continue
        out.append(r)
    return out


def is_empty(v: Any) -> bool:
    if v is None:
        return True
    if v == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def strip_empty_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v2 = strip_empty_keys(v)
            if not is_empty(v2):
                cleaned[k] = v2
        return cleaned
    if isinstance(obj, list):
        return [strip_empty_keys(x) for x in obj]
    return obj


_GENERIC_ENTRY_CLASSES = frozenset({
    "MainActivity", "SplashActivity", "IntroActivity", "SplashSchemeActivity",
})


def drop_keys_in_app_usage(records: list[dict]) -> None:
    for r in records:
        if r.get("data_type") != "app_usage":
            continue
        if not r.get("app_name"):
            r["app_name"] = r.get("package")
        r.pop("package", None)

        typ = r.get("type")
        if isinstance(typ, str) and typ.upper() == "SHORTCUT_INVOCATION":
            r.pop("shortcutId", None)

        c = r.get("class")
        if isinstance(c, str):
            short = c.rsplit(".", 1)[-1] if "." in c else c
            if short in _GENERIC_ENTRY_CLASSES:
                r.pop("class", None)
            else:
                r["class"] = short


def drop_keys_in_notification(records: list[dict]) -> None:
    for r in records:
        if r.get("data_type") != "notification":
            continue
        r.pop("channel_id", None)
        r.pop("category", None)
        r.pop("removed_at", None)
        r.pop("seen_time", None)
        if "big_text" in r and "text" in r:
            if r.get("text") == r.get("big_text"):
                r.pop("big_text", None)
        if not r.get("app_name"):
            r["app_name"] = r.get("package")
        r.pop("package", None)


def align_notification_time_to_seen_time(records: list[dict]) -> None:
    """In-place for notifications: heal seen_time, then realign ``time``.

    seen_time is considered unusable when missing, earlier than ``time``, or
    more than a day after ``time`` (the latter typically a bulk OS cancel_all).
    When unusable:
    - if ``removed_at`` falls within ``[time, time + 1 day]``, copy it in
    - otherwise, fall back to ``time + 1s``

    After healing, ``time`` is overwritten with ``seen_time`` so the record is
    timestamped at when the notification was seen rather than when it arrived.
    ``seen_time`` and ``removed_at`` are left on the record;
    ``drop_keys_in_notification`` removes them downstream.
    """
    one_day = timedelta(days=1)
    one_sec = timedelta(seconds=1)

    for r in records:
        if r.get("data_type") != "notification":
            continue
        t_dt = _parse_event_time(r.get("time"))
        if t_dt is None:
            continue
        st_dt = _parse_event_time(r.get("seen_time"))
        rm_dt = _parse_event_time(r.get("removed_at"))
        max_allowed = t_dt + one_day

        unusable = st_dt is None or st_dt < t_dt or st_dt > max_allowed
        if unusable:
            if rm_dt is not None and t_dt <= rm_dt <= max_allowed:
                r["seen_time"] = rm_dt.strftime(_TIME_FMT)
            else:
                r["seen_time"] = (t_dt + one_sec).strftime(_TIME_FMT)

        r["time"] = r["seen_time"]


# Meters / accuracy / elevation — round to 2 dp per user request ("거리 데이터").
LOCATION_DISTANCE_KEYS = frozenset({"distance_moved", "estimation_accuracy", "altitude"})
LOCATION_COORD_KEYS = frozenset({"latitude", "longitude"})


def _round_numeric_compact(value: Any, ndigits: int) -> Any:
    """Round floats; emit int when the value is integral after rounding (smaller JSON)."""
    if not isinstance(value, (int, float)):
        return value
    r = round(float(value), ndigits)
    ir = int(r)
    if abs(r - ir) < 1e-9:
        return ir
    return r

def compress_location_numeric(records: list[dict], coord_decimals: int | None) -> None:
    """In-place: round location distance fields; optionally round lat/lon for fewer digits."""
    for r in records:
        if r.get("data_type") != "location":
            continue
        for k in LOCATION_DISTANCE_KEYS:
            if k in r:
                r[k] = _round_numeric_compact(r[k], 2)
        if coord_decimals is not None:
            for k in LOCATION_COORD_KEYS:
                if k in r:
                    r[k] = _round_numeric_compact(r[k], coord_decimals)


def repair_korean_duration(s: str) -> str:
    """
    Fix corrupted duration strings where Hangul (시간/분/초) became '?'.

    Patterns: `` after digits → 시간; single `?` token order → 분 then 초
    (two such tokens), or 분 only after an hour token if one remains.
    """
    if not isinstance(s, str) or "?" not in s:
        return s
    parts = [p for p in s.split() if p]
    if not parts:
        return s
    has_hours = any(re.match(r"^\d+\?\?$", p) for p in parts)
    ss_indices = [i for i, p in enumerate(parts) if re.match(r"^\d+\?$", p)]
    ss_count = len(ss_indices)
    out: list[str] = []
    for i, p in enumerate(parts):
        m2 = re.match(r"^(\d+)\?\?$", p)
        if m2:
            out.append(f"{m2.group(1)}시간")
            continue
        m1 = re.match(r"^(\d+)\?$", p)
        if m1:
            pos_in_ss = ss_indices.index(i)
            if ss_count >= 2:
                out.append(f"{m1.group(1)}분" if pos_in_ss == 0 else f"{m1.group(1)}초")
            else:
                out.append(f"{m1.group(1)}분" if has_hours else f"{m1.group(1)}초")
            continue
        out.append(p)
    return " ".join(out)


def duration_to_seconds(dur: str) -> int | None:
    """Parse Korean duration like '2시간 3분 5초' after optional repair."""
    if not isinstance(dur, str) or not dur.strip():
        return None
    fixed = repair_korean_duration(dur)
    total = 0
    if m := re.search(r"(\d+)\s*시간", fixed):
        total += int(m.group(1)) * 3600
    if m := re.search(r"(\d+)\s*분", fixed):
        total += int(m.group(1)) * 60
    if m := re.search(r"(\d+)\s*초", fixed):
        total += int(m.group(1))
    return total


def _format_korean_duration(total_seconds: int) -> str:
    """Inverse of ``duration_to_seconds`` for non-negative integer seconds."""
    if total_seconds < 0:
        total_seconds = 0
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}시간")
    if m:
        parts.append(f"{m}분")
    if s or not parts:
        parts.append(f"{s}초")
    return " ".join(parts)


# Sleep stage 1 — locations the user calls "home" (used to ignore sleep rows
# that happen elsewhere, e.g. naps at the office or transit).
SLEEP_HOME_LOCATION_IDS = frozenset({10, 42})


def drop_sleep_records_away_from_home(records: list[dict]) -> list[dict]:
    """Sleep stage 1: drop sleep rows when both fall-asleep AND wake boundaries
    are at non-home locations.

    A record is kept whenever at least one boundary lies at a home location
    (Location 10 or 42). The location at a boundary is the most recent location
    sample at or before that boundary; records pre-dating any location sample
    are kept (no evidence of being away).

    Trailing 짧은 수면 chains are dropped naturally by the same boundary check
    on their own timestamps — no additional plumbing required.
    """
    loc_pairs: list[tuple[datetime, int]] = []
    for r in records:
        if r.get("data_type") != "location":
            continue
        t = _parse_event_time(r.get("time"))
        if t is None:
            continue
        lid = _effective_location_id(r)
        if lid is None:
            continue
        loc_pairs.append((t, lid))
    loc_pairs.sort(key=lambda x: x[0])
    loc_times = [p[0] for p in loc_pairs]

    def state_at(t: datetime) -> int | None:
        i = bisect.bisect_right(loc_times, t) - 1
        if i < 0:
            return None
        return loc_pairs[i][1]

    out: list[dict] = []
    for r in records:
        if r.get("data_type") != "sleep":
            out.append(r)
            continue
        t_fall = _parse_event_time(
            r.get("time_fallasleep") or r.get("time_fallsleep") or r.get("time")
        )
        t_wake = _parse_event_time(r.get("time_wakeup") or r.get("time"))
        s_fall = state_at(t_fall) if t_fall else None
        s_wake = state_at(t_wake) if t_wake else None
        if s_fall is None and s_wake is None:
            out.append(r)
            continue
        at_home = (
            (s_fall is not None and s_fall in SLEEP_HOME_LOCATION_IDS)
            or (s_wake is not None and s_wake in SLEEP_HOME_LOCATION_IDS)
        )
        if at_home:
            out.append(r)
    return out


# Sleep reconstruction thresholds.
SLEEP_MIN_GAP_MINUTES = 45
SLEEP_CHAIN_GAP_MINUTES = 5


def reconstruct_sleep_records(records: list[dict]) -> list[dict]:
    """Replace all ``sleep`` records with rule-based reconstruction from raw events.

    Rules:
    - Rule 2 (screen): SCREEN OFF followed by SCREEN ON with gap ≥ ``SLEEP_MIN_GAP_MINUTES``
      → ``수면`` (fall_hour ∈ [18,24)∪[0,10)) or ``낮잠`` ([10,18)).
      fall = OFF time, wake = ON time.
    - Rule 4 (device): DEVICE_SHUTDOWN with night fall_hour followed by DEVICE_STARTUP
      with gap ≥ ``SLEEP_MIN_GAP_MINUTES`` → ``수면``.
      fall = last SCREEN_INTERACTIVE before SHUTDOWN (captures phone-idle period before
      battery death). wake = STARTUP time.
    - Rule 3 chain: after a sleep wake, if next OFF starts within
      ``SLEEP_CHAIN_GAP_MINUTES``, absorb the OFF→ON pair and update wake.
    - Rule 5 chain: after a sleep wake, if next SHUTDOWN starts within
      ``SLEEP_CHAIN_GAP_MINUTES``, absorb the SHUTDOWN→STARTUP pair and update wake.
    - Earliest trigger wins; events inside [fall, wake] are absorbed silently.

    Output sleep dicts contain time, time_fallasleep, time_wakeup, class, duration.
    Run BEFORE the ``_APP_DROP_BY_TYPES`` filter (which strips screen/device events).
    """
    sleep_min = timedelta(minutes=SLEEP_MIN_GAP_MINUTES)
    chain_max = timedelta(minutes=SLEEP_CHAIN_GAP_MINUTES)

    type_map = {
        "SCREEN_INTERACTIVE": "ON",
        "SCREEN_NON_INTERACTIVE": "OFF",
        "DEVICE_SHUTDOWN": "SHUTDOWN",
        "DEVICE_STARTUP": "STARTUP",
    }

    raw_events: list[tuple[datetime, str, int]] = []
    for orig_idx, r in enumerate(records):
        if r.get("data_type") != "app_usage":
            continue
        ek = type_map.get(r.get("type"))
        if not ek:
            continue
        t = _parse_event_time(r.get("time"))
        if t is None:
            continue
        raw_events.append((t, ek, orig_idx))

    seen_shut: set[tuple] = set()
    events: list[tuple[datetime, str, int]] = []
    for e in raw_events:
        t, ek, _ = e
        if ek == "SHUTDOWN":
            key = (t, ek)
            if key in seen_shut:
                continue
            seen_shut.add(key)
        events.append(e)
    events.sort(key=lambda x: (x[0], x[2]))
    n = len(events)

    def is_night(h: int) -> bool:
        return h >= 18 or h < 10

    def next_idx(start: int, kind: str) -> int | None:
        for j in range(start, n):
            if events[j][1] == kind:
                return j
        return None

    def last_on_before(end: int) -> datetime | None:
        for j in range(end - 1, -1, -1):
            if events[j][1] == "ON":
                return events[j][0]
        return None

    new_sleeps: list[dict] = []
    i = 0
    while i < n:
        t_i, k_i, _ = events[i]
        trigger: tuple | None = None  # (rule, fall, wake, wake_idx)

        if k_i == "OFF":
            j = next_idx(i + 1, "ON")
            if j is not None and events[j][0] - t_i >= sleep_min:
                trigger = ("rule2", t_i, events[j][0], j)

        if k_i == "SHUTDOWN" and is_night(t_i.hour):
            j = next_idx(i + 1, "STARTUP")
            if j is not None and events[j][0] - t_i >= sleep_min:
                last_on = last_on_before(i)
                fall = last_on if last_on is not None else t_i
                trigger = ("rule4", fall, events[j][0], j)

        if trigger is None:
            i += 1
            continue

        rule, fall, wake, wake_idx = trigger
        if rule == "rule4":
            cls = "수면"
        else:
            h = fall.hour
            cls = "낮잠" if 10 <= h < 18 else "수면"

        cur_wake = wake
        cur_wake_idx = wake_idx
        while True:
            r3 = None
            for k in range(cur_wake_idx + 1, n):
                ek_t, ek_ty, _ = events[k]
                if ek_t - cur_wake >= chain_max:
                    break
                if ek_ty == "OFF":
                    on_idx = next_idx(k + 1, "ON")
                    if on_idx is not None:
                        r3 = (ek_t, events[on_idx][0], on_idx)
                    break

            r5 = None
            for k in range(cur_wake_idx + 1, n):
                ek_t, ek_ty, _ = events[k]
                if ek_t - cur_wake >= chain_max:
                    break
                if ek_ty == "SHUTDOWN":
                    su_idx = next_idx(k + 1, "STARTUP")
                    if su_idx is not None:
                        r5 = (ek_t, events[su_idx][0], su_idx)
                    break

            best = None
            if r3 and r5:
                best = r3 if r3[0] <= r5[0] else r5
            elif r3:
                best = r3
            elif r5:
                best = r5

            if best is None:
                break
            cur_wake = best[1]
            cur_wake_idx = best[2]

        dur_secs = int((cur_wake - fall).total_seconds())
        new_sleeps.append({
            "data_type": "sleep",
            "time": fall.strftime(_TIME_FMT),
            "time_fallasleep": fall.strftime(_TIME_FMT),
            "time_wakeup": cur_wake.strftime(_TIME_FMT),
            "class": cls,
            "duration": _format_korean_duration(dur_secs),
        })
        i = cur_wake_idx + 1

    out = [r for r in records if r.get("data_type") != "sleep"]
    out.extend(new_sleeps)
    return out


# Short naps / noise: drop sleep rows below this parsed duration (seconds).
MIN_SLEEP_DURATION_SECONDS = 120

def drop_short_sleep_records(
    records: list[dict],
    min_seconds: int = MIN_SLEEP_DURATION_SECONDS,
) -> list[dict]:
    """Remove sleep rows with parsed duration strictly under min_seconds (default 2 min)."""
    out: list[dict] = []
    for r in records:
        if r.get("data_type") != "sleep":
            out.append(r)
            continue
        elif r.get("class") != "짧은 수면":
            out.append(r)
            continue

        dur = r.get("duration", "") #짧은 수면의 duration
        if isinstance(dur, str):
            dur = repair_korean_duration(dur)
        secs = duration_to_seconds(dur) if isinstance(dur, str) else None

        if secs < min_seconds:
            continue
        out.append(r)
    return out


_TIME_FMT = "%Y-%m-%d %H:%M:%S"
_WEEKDAY_ABBR_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def add_weekday_field(records: list[dict]) -> None:
    """In-place: set `weekday` to Mon..Sun from naive `time` (no timezone conversion)."""
    for r in records:
        t = r.get("time")
        if not isinstance(t, str):
            continue
        ts = t.strip()
        if len(ts) < 19:
            continue
        try:
            naive = datetime.strptime(ts[:19], _TIME_FMT)
        except ValueError:
            continue
        r["weekday"] = _WEEKDAY_ABBR_EN[naive.weekday()]


def _parse_event_time(t: Any) -> datetime | None:
    if not isinstance(t, str):
        return None
    try:
        return datetime.strptime(t, _TIME_FMT)
    except ValueError:
        return None


def _fix_duration_on_record(r: dict) -> None:
    d = r.get("duration")
    if isinstance(d, str):
        r["duration"] = repair_korean_duration(d)


def _effective_location_id(r: dict) -> int | None:
    lid = r.get("location_id")
    if lid is not None:
        try:
            return int(lid)
        except (TypeError, ValueError):
            pass
    lab = r.get("location_label")
    if isinstance(lab, str):
        m = re.search(r"Location\s+(\d+)", lab, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def moved_prefix(records: list[dict]) -> None:
    prev_id: int | None = None
    for r in records:
        if r.get("data_type") != "location":
            continue
        lid = _effective_location_id(r)
        if lid is None:
            continue
        if prev_id is not None and lid != prev_id:
            cur = r.get("location_label")
            if isinstance(cur, str) and cur and not cur.startswith("Moved to "):
                r["location_label"] = "Moved to " + cur
        prev_id = lid


def _movement_start_and_end_labels(r: dict) -> dict | None:
    """
    Mutate movement start row `r` activity; return a deep-copied end row template
    (caller sets time to end-of-movement = r.time + duration).
    """
    act = r.get("activity")
    if not isinstance(act, str):
        return None
    key = act.strip().casefold()
    end = deepcopy(r)
    if key in ("walking", "running"):
        r["activity"] = f"start {act}"
        end["activity"] = f"end {act}"
        return end
    if key == "in_vehicle":
        r["activity"] = "get on vehicle"
        end["activity"] = "get off vehicle"
        return end
    return None


def normalize_movement_activities(records: list[dict]) -> list[dict]:
    """Fix ``on_bicycle`` misclassifications and merge consecutive same-activity rows.

    Step 1 — every ``on_bicycle`` row is rewritten using:
    1. the next movement's activity, if it starts within 1 min of this row's end
    2. else the previous movement's activity, if it ended within 1 min of this row's start
    3. else ``walking`` as the safe default
    Replacements skip ``on_bicycle`` neighbors so chains can't echo themselves.

    Step 2 — adjacent movement rows that share an activity are merged when
    ``next.time - prev.end`` is within 1 min. The earlier row is kept and its
    duration is extended to span [first.start, last.end].

    Operates on the raw movement schema (no ``start ``/``end `` prefix on
    activity); call this before ``startend_sleep_and_movement_call``.
    """
    one_min = timedelta(minutes=1)
    movements: list[list] = []
    for r in records:
        if r.get("data_type") != "movement":
            continue
        t = _parse_event_time(r.get("time"))
        if t is None:
            continue
        secs = duration_to_seconds(r.get("duration", "")) or 0
        movements.append([t, t + timedelta(seconds=secs), r])
    movements.sort(key=lambda x: x[0])

    # bicycle 바꾸기
    original_activities = [m[2].get("activity") for m in movements]
    for i, (t, end, r) in enumerate(movements):
        if original_activities[i] != "on_bicycle":
            continue
        replacement: str | None = None
        if i + 1 < len(movements):
            nt, _ne, _nr = movements[i + 1]
            next_orig = original_activities[i + 1]
            if next_orig != "on_bicycle" and nt - end <= one_min:
                replacement = next_orig
        if replacement is None and i > 0:
            _pt, pe, _pr = movements[i - 1]
            prev_orig = original_activities[i - 1]
            if prev_orig != "on_bicycle" and t - pe <= one_min:
                replacement = prev_orig
        r["activity"] = replacement or "walking"

    # 연속되는 동일 activity 합치기
    drop_ids: set[int] = set()
    cur_start, cur_end, cur_record = movements[0]
    for i in range(1, len(movements)):
        next_start, next_end, next_record = movements[i]
        if next_record.get("activity") == cur_record.get("activity") and next_start - cur_end <= one_min:
            drop_ids.add(id(next_record))
            cur_end = max(cur_end, next_end)
            new_secs = int((cur_end - cur_start).total_seconds())
            cur_record["duration"] = _format_korean_duration(new_secs)
        else:
            cur_start, cur_end, cur_record = next_start, next_end, next_record

    return [r for r in records if id(r) not in drop_ids]


def startend_sleep_movement_call(records: list[dict]) -> list[dict]:
    """
    Treats each record's ``time`` as the START of the activity.

    - Repair duration Hangul placeholders for sleep/movement/call.
    - Sleep: add a Wake-up row at ``time_wakeup`` (class suffixed with ``끝``).
    - Movement: relabel original to ``start <activity>``; add ``end <activity>``
      row at ``time + duration`` (vehicle uses ``get on/off vehicle``).
    - Call: append end row at ``time + duration`` with ``call_type_label`` +
      ``call end`` (skip when 0초).
    - Sort by time, then by data_type.
    """
    extras: list[dict] = []
    for r in records:
        dt = r.get("data_type")
        if dt == "sleep":
            _fix_duration_on_record(r)
            tw = r.get("time_wakeup")
            if isinstance(tw, str) and tw.strip():
                end_row = deepcopy(r)
                end_row["time"] = tw
                end_row["class"] = end_row["class"] + " 끝"
                end_row.pop("time_wakeup", None)
                end_row.pop("time_fallasleep", None)
                extras.append(end_row)
            r.pop("duration", None)
            r.pop("time_wakeup", None)
            r.pop("time_fallasleep", None)
        elif dt == "movement":
            _fix_duration_on_record(r)
            start_t = _parse_event_time(r.get("time"))
            secs = duration_to_seconds(r.get("duration", ""))
            end_row = _movement_start_and_end_labels(r)
            if end_row is not None and start_t is not None and secs is not None and secs > 0:
                end_t = start_t + timedelta(seconds=secs)
                end_row["time"] = end_t.strftime(_TIME_FMT)
                extras.append(end_row)
            r.pop("duration", None)
        elif dt == "call":
            _fix_duration_on_record(r)
            start_t = _parse_event_time(r.get("time"))
            secs = duration_to_seconds(r.get("duration", ""))
            if start_t is not None and secs is not None and secs > 0:
                end_row = deepcopy(r)
                end_row["time"] = (start_t + timedelta(seconds=secs)).strftime(_TIME_FMT)
                ctl = r.get("call_type_label")
                if isinstance(ctl, str) and ctl.strip():
                    end_row["call_type_label"] = ctl.rstrip() + " call end"
                else:
                    end_row["call_type_label"] = "call end"
                extras.append(end_row)
            r.pop("duration", None)

    combined = records + extras
    combined.sort(
        key=lambda x: (
            x.get("time") or "",
            x.get("data_type") or "",
        )
    )
    return combined


def compact_datetime_strings(obj: Any) -> Any:
    """Recursively shorten 20YY-... timestamps by removing a leading '20' (saves two bytes each)."""
    if isinstance(obj, dict):
        return {k: compact_datetime_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [compact_datetime_strings(x) for x in obj]
    if isinstance(obj, str) and _DATETIME_20YY.match(obj):
        return obj[2:-3]
    return obj


def abbreviate_package_prefixes(obj: Any) -> Any:
    """Shorten dot-separated identifiers in ``package`` and ``app_name`` fields
    to their last two segments when the value has 3+ segments. Single-token or
    human-readable values (e.g., ``토스``, ``Knox Teams``) are left untouched.

    Run before ``drop_keys_in_app_usage`` so that when package is merged into
    app_name, the abbreviated form propagates.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in ("package", "app_name") and isinstance(v, str):
                parts = v.split('.')
                out[k] = '.'.join(parts[-2:]) if len(parts) > 2 else v
            else:
                out[k] = abbreviate_package_prefixes(v)
        return out
    if isinstance(obj, list):
        return [abbreviate_package_prefixes(x) for x in obj]
    return obj


def abbreviate_data_type_field(obj: Any) -> Any:
    """Replace verbose data_type with short codes (see _DATA_TYPE_ABBREV)."""
    if isinstance(obj, dict):
        out = {k: abbreviate_data_type_field(v) for k, v in obj.items()}
        dt = out.get("data_type")
        if isinstance(dt, str) and dt in _DATA_TYPE_ABBREV:
            out["data_type"] = _DATA_TYPE_ABBREV[dt]
        return out
    if isinstance(obj, list):
        return [abbreviate_data_type_field(x) for x in obj]
    return obj


def rename_data_type_and_package_keys(obj: Any) -> Any:
    """Recursively rename ``data_type`` -> ``dtype`` and ``package`` -> ``pkg`` for output."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            nk = "dtype" if k == "data_type" else "pkg" if k == "package" else k
            out[nk] = rename_data_type_and_package_keys(v)
        return out
    if isinstance(obj, list):
        return [rename_data_type_and_package_keys(x) for x in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress user usage JSON for prompts.")
    parser.add_argument("--i", default="user_usage_4weeks_1_filtered.json", help="Input JSON path.")
    parser.add_argument("--o", default="user_usage_4weeks_2_compressed.json", help="Output JSON path.")
    parser.add_argument("--pretty", default="user_usage_4weeks_2_pretty.json", help="Output JSON path.")

    args = parser.parse_args()
    with open(args.i, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    # Deep copy so we never mutate unexpected shared refs
    records = [deepcopy(r) for r in records]

    # sleep 레코드를 raw 이벤트로부터 재구성. 시스템 필터가 SCREEN_*/DEVICE_SHUTDOWN을 제거하기 전에 수행해야 함.
    records = reconstruct_sleep_records(records)

    # 시스템 데이터들들 드랍하고 시작 (ROW 단위 삭제제)
    before, before_words = len(records), len(json.dumps(records).replace(' ', ''))
    filtered: list[dict] = []
    for r in records:
        pkg = r.get("package")
        if isinstance(pkg, str) and any(name in pkg for name in [
            "usagehistory","notifhistory","notificationhistory","lifelog","minit","tmoney.manager",
            "permissioncontroller","packageinstaller","credentialmanager","android.voc","android.as","android.settings",
            "launcher","intentresolver","budsunitemgr","documentsui","photopicker"]):
            continue    # package name으로 drop
        if r.get("data_type") == "app_usage":
            if any(name in pkg for name in ["routines", "systemui","android.gms","omcagent","vending"]):
                continue    # app_usage에 한해서만 특정 package name으로 drop
            c = r.get("class")
            if isinstance(c, str) and c.rsplit(".", 1)[-1] == "ResolverActivity":
                continue    # class가 ResolverActivity인 app_usage drop
        if r.get("type") in _APP_DROP_BY_TYPES:
            continue    # type으로 drop
        filtered.append(r)
    records = filtered

    records = thin_consecutive_app_usage(records)
    records = thin_consecutive_connection(records)
    records = dedup_user_interaction_after_resume(records)

    # DUration 데이터(movement, sleep, call) 시작 끝 데이터 추가
    records = drop_sleep_records_away_from_home(records)
    records = drop_short_sleep_records(records)
    records = normalize_movement_activities(records)
    records = startend_sleep_movement_call(records)

    after_drop = len(records)


    # COLUMN 단위 삭제
    moved_prefix(records)
    records = abbreviate_package_prefixes(records)
    drop_keys_in_app_usage(records)
    records = strip_empty_keys(records)
    align_notification_time_to_seen_time(records)
    drop_keys_in_notification(records)


    # 단어 축약
    compress_location_numeric(records, coord_decimals=5)
    records = abbreviate_data_type_field(records)
    records = rename_data_type_and_package_keys(records)

    # add_weekday_field(records)

    records.sort(key=lambda r: r.get("time") or "")
    records = compact_datetime_strings(records)

    after_thin = len(records)

    with open(args.o, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    with open(args.pretty, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"), indent=2)

    print(
        f"rows: {before} -> after_drop={after_drop}\n"
        f"words: reduced to {round(len(json.dumps(records).replace(' ', ''))/before_words * 100, 1)}%\n-> written {args.o}"
    )


if __name__ == "__main__":
    main()
