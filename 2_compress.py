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
        "STANDBY_BUCKET_CHANGED", "CONFIGURATION_CHANGE", "NOTIFICATION_INTERRUPTION", "NOTIFICATION_SEEN",
        "SCREEN_INTERACTIVE", "SCREEN_NON_INTERACTIVE", "DEVICE_SHUTDOWN", "DEVICE_STARTUP"
    }
)

def thin_consecutive_app_usage(records: list[dict], simple: bool = False) -> list[dict]:
    """Collapse consecutive app_usage rows that share (package, type).
    If ``simple`` is False (default), ``class`` must also match — different
    activities within the same app are preserved as separate events.
    """
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
        cls = r.get("class")
        j = i
        while (
            j < n
            and records[j].get("data_type") == "app_usage"
            and records[j].get("package") == pkg
            and records[j].get("type") == typ
            and (simple or records[j].get("class") == cls)
        ):
            j += 1
        out.append(r)
        i = j
    return out


_MAC_ADDR_RE = re.compile(r"^[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}$")


def thin_consecutive_connection(records: list[dict]) -> list[dict]:
    """Drop a connection row if any of:
    - NETWORK with summary == "UNKNOWN" (uninformative),
    - BLUETOOTH with device_name that looks like a MAC address (unidentified),
    - it duplicates the previous connection of the same category.

    NETWORK dedup key: (event_kind, summary, ip);
    BLUETOOTH dedup key: (event_kind, device_name).
    Non-connection rows in between are ignored (no time bound).
    """
    out: list[dict] = []
    prev_key: tuple | None = None
    for r in records:
        if r.get("data_type") != "connection":
            out.append(r)
            continue
        cat = r.get("category")
        ek = r.get("event_kind")
        if cat == "NETWORK":
            summary = r.get("summary")
            if summary == "UNKNOWN":
                continue
            key = ("NETWORK", ek, summary, r.get("ip"))
        elif cat == "BLUETOOTH":
            dn = r.get("device_name")
            if isinstance(dn, str) and _MAC_ADDR_RE.match(dn):
                continue
            key = ("BLUETOOTH", ek, dn)
        else:
            key = (cat, ek, r.get("summary"), r.get("ip"), r.get("device_name"))
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

def duration_to_seconds(dur: str) -> int | None:
    """Parse durations in the ``'<H>h <M>m <S>s'`` form (e.g. ``'7h 39m 12s'``)."""
    if not isinstance(dur, str) or not dur.strip():
        return None
    total = 0
    if m := re.search(r"(\d+)\s*h", dur):
        total += int(m.group(1)) * 3600
    if m := re.search(r"(\d+)\s*m(?!s)", dur):
        total += int(m.group(1)) * 60
    if m := re.search(r"(\d+)\s*s", dur):
        total += int(m.group(1))
    return total


def _format_duration(total_seconds: int) -> str:
    """Inverse of ``duration_to_seconds`` for non-negative integer seconds."""
    if total_seconds < 0:
        total_seconds = 0
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"



_TIME_FMT = "%Y-%m-%d %H:%M:%S"

def _parse_event_time(t: Any) -> datetime | None:
    if not isinstance(t, str):
        return None
    try:
        return datetime.strptime(t, _TIME_FMT)
    except ValueError:
        return None


def thin_consecutive_locations(records: list[dict]) -> list[dict]:
    """Drop a location record whose ``location_label`` equals the previously-seen location label (in time-sorted order). 
    Each surviving location record marks a transition into a new cluster, 
    so its ``location_label`` is prefixed with ``"Moved to "``.
    """
    records.sort(key=lambda r: r.get("time") or "")
    out: list[dict] = []
    last_loc_label: str | None = None
    for r in records:
        if r.get("data_type") == "location":
            cur = r.get("location_label")
            if cur == last_loc_label:
                continue
            last_loc_label = cur
            if isinstance(cur, str) and cur and not cur.startswith("Moved to "):
                r["location_label"] = "Moved to " + cur
        out.append(r)
    return out


def normalize_movement_activities(records: list[dict]) -> list[dict]:
    """Fix ``on_bicycle`` misclassifications and merge consecutive same-activity rows.

    Step 1 — every ``on_bicycle`` row is rewritten using:
    1. the next movement's activity, if it starts within 1 min of this row's end
    2. else the previous movement's activity, if it ended within 1 min of this row's start
    3. else ``walking`` as the safe default
    Replacements skip ``on_bicycle`` neighbors so chains can't echo themselves.

    Step 2 — adjacent movement rows that share an activity are merged when ``next.time - prev.end`` 
    is within 1 min. The earlier row is kept and its duration is extended to span [first.start, last.end].

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
            cur_record["duration"] = _format_duration(new_secs)
        else:
            cur_start, cur_end, cur_record = next_start, next_end, next_record

    return [r for r in records if id(r) not in drop_ids]


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


def startend_sleep_movement_call(records: list[dict]) -> list[dict]:
    """
    Treats each record's ``time`` as the START of the activity.
    - Sleep: add a Wake-up row at ``time_wakeup`` (class suffixed with ``끝``).
    - Movement: relabel original to ``start <activity>``; add ``end <activity>``
      row at ``time + duration`` (vehicle uses ``get on/off vehicle``).
    - Call: append end row at ``time + duration`` with ``call_type_label`` + ``call end`` (skip when 0s).
    - Sort by time, then by data_type.
    """
    extras: list[dict] = []
    for r in records:
        dt = r.get("data_type")
        if dt == "sleep":
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
            start_t = _parse_event_time(r.get("time"))
            secs = duration_to_seconds(r.get("duration", ""))
            end_row = _movement_start_and_end_labels(r)
            if end_row is not None and start_t is not None and secs is not None and secs > 0:
                end_t = start_t + timedelta(seconds=secs)
                end_row["time"] = end_t.strftime(_TIME_FMT)
                extras.append(end_row)
            r.pop("duration", None)
        elif dt == "call":
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
    combined.sort(key=lambda x: (x.get("time") or "", x.get("data_type") or ""))
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

    Run before ``drop_keys_in_app_usage`` so that when package is merged into app_name, the abbreviated form propagates.
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
    parser.add_argument("--i", default="user_usage_7weeks_filtered.json", help="Input JSON path.")
    parser.add_argument("--o", default="user_usage_7weeks_2_compressed.json", help="Output JSON path.")
    parser.add_argument("--pretty", default="user_usage_7weeks_2_pretty.json", help="Output JSON path.")
    parser.add_argument("--simple", action="store_true",
                            help="Collapse consecutive app_usage by (package, type) only")

    args = parser.parse_args()
    with open(args.i, encoding="utf-8") as f:
        records: list[dict] = json.load(f)["events"]

    # Deep copy so we never mutate unexpected shared refs
    records = [deepcopy(r) for r in records]
    before, before_words = len(records), len(json.dumps(records).replace(' ', ''))
    
    # 시스템 데이터들들 드랍하고 시작 (ROW 단위 삭제)
    filtered: list[dict] = []
    for r in records:
        pkg = r.get("package")
        if isinstance(pkg, str) and any(name in pkg for name in [
            "usagehistory","notifhistory","notificationhistory","lifelog","minit","tmoney.manager",
            "permissioncontroller","packageinstaller","android.as","instant_app"]):
            continue    # package name으로 drop
        if r.get("data_type") == "app_usage":
            if any(name in pkg for name in ["systemui","android.gms","omcagent","vending"]):
                continue    # app_usage에 한해서만 특정 package name으로 drop
            c = r.get("class")
            if isinstance(c, str) and c.rsplit(".", 1)[-1] == "ResolverActivity":
                continue    # class가 ResolverActivity인 app_usage drop
        if r.get("type") in _APP_DROP_BY_TYPES:
            continue    # type으로 drop
        filtered.append(r)
    records = filtered

    records = thin_consecutive_app_usage(records, simple=args.simple)
    records = dedup_user_interaction_after_resume(records)
    records = thin_consecutive_connection(records)
    records = thin_consecutive_locations(records)

    # Duration 데이터(movement, sleep, call) 시작 끝 데이터 추가
    records = normalize_movement_activities(records)
    records = startend_sleep_movement_call(records)
    after_drop = len(records)

    # COLUMN 단위 삭제
    records = abbreviate_package_prefixes(records)
    drop_keys_in_app_usage(records)
    records = strip_empty_keys(records)
    align_notification_time_to_seen_time(records)
    drop_keys_in_notification(records)

    # 단어 축약
    compress_location_numeric(records, coord_decimals=5)
    records = abbreviate_data_type_field(records)
    records = rename_data_type_and_package_keys(records)
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
