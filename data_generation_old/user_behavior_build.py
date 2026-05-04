import bisect
import glob
import json
import os
import random
from datetime import datetime, timedelta
from itertools import accumulate
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
STM_PATH = os.path.join(REPO_ROOT, "user_usage_3weeks_compressed.json")
DEFAULT_SAMPLES_PATH = os.path.join(DATA_DIR, "user_behavior_samples.json")

SPLIT_TIME = datetime(2026, 4, 24, 0, 0)

TIME_FMT = "%y-%m-%d %H:%M"

NOISE_TYPES = {
    # APP에서 drop하고싶은 type들 넣으면 됨
}

PKG_HUMAN = {
    "com.google.android.youtube": "YouTube",
    "com.google.android.apps.youtube.music": "YouTubeMusic",
    "viva.republica.toss": "Toss",
    "토스": "Toss",
    "com.nhn.android.nmap": "NaverMap",
    "com.nhn.android.search": "NaverSearch",
    "com.kakao.talk": "KakaoTalk",
    "kakao.talk": "KakaoTalk",
    "com.kakaopay.app": "KakaoPay",
    "com.instagram.android": "Instagram",
    "com.sec.android.app.clockpackage": "Clock",
    "시계": "Clock",
    "com.sec.android.app.launcher": "Launcher",
    "com.sec.android.app.camera": "Camera",
    "com.sec.android.gallery3d": "Gallery",
    "com.samsung.android.calendar": "Calendar",
    "com.samsung.android.app.contacts": "Contacts",
    "com.samsung.android.dialer": "Dialer",
    "com.samsung.android.app.notes": "Samsung Notes",
    "com.samsung.android.messaging": "Messages",
    "com.samsung.android.app.routines": "Routines",
    "com.samsung.android.spay": "SamsungWallet",
    "com.sds.teams": "KnoxTeams",
    "Knox Teams": "KnoxTeams",
    "kr.co.skplanet.musicmate": "FLO",
    "com.btckorea.bithumb": "Bithumb",
    "com.starbucks.co": "Starbucks",
    "starbucks.co": "Starbucks"
}


def _parse_time(s):
    return datetime.strptime(s, TIME_FMT)


def _humanize_pkg(pkg):
    if pkg is None:
        return "?"
    if pkg in PKG_HUMAN:
        return PKG_HUMAN[pkg]
    return pkg.split(".")[-1] if "." in pkg else pkg


def _pkg_key(item, humanize=True):
    pkg = item.get("app_name") or item.get("pkg") or ""
    return _humanize_pkg(pkg) if humanize else pkg


def _serialize_entry(item, humanize_pkg=True):
    """One STM line per entry: time + dtype + identifier + type/info."""
    t = item.get("time", "")[3:]  # drop year prefix "26-" → "MM-DD HH:MM"
    dt = item.get("dtype", "?")
    if dt == "app":
        pkg = _pkg_key(item, humanize_pkg)
        return f"{t} app {pkg} {item.get('type','')}"
    if dt == "noti":
        pkg = _pkg_key(item, humanize_pkg)
        title = (item.get("title") or "")[:40]
        return f"{t} noti {pkg} {title}"
    if dt == "location":
        return f"{t} location {item.get('location_label','')}"
    if dt == "connection":
        return (
            f"{t} connection {item.get('category','')} "
            f"{item.get('event_kind','')} {item.get('device_name') or item.get('summary','')}"
        )
    if dt == "movement":
        return f"{t} movement {item.get('activity','')} {item.get('duration','')}"
    if dt == "sleep":
        return f"{t} sleep {item.get('class','')} {item.get('duration','')}"
    if dt == "calendar":
        return f"{t} calendar {item.get('class','')} {item.get('title','')}"
    if dt == "call":
        return (
            f"{t} call {item.get('call_type_label','')} "
            f"{item.get('contact_name','')} {item.get('duration','')}"
        )
    return f"{t} {dt}"


def _is_noise(item):
    return item.get("dtype") == "app" and item.get("type") in NOISE_TYPES


def _coalesce(lines_meta):
    """Collapse runs of (dtype,pkg,type) within ±2 min into '... ×N'."""
    out = []
    for entry in lines_meta:
        if (
            out
            and out[-1]["key"] == entry["key"]
            and (entry["t"] - out[-1]["t_last"]).total_seconds() <= 120
        ):
            out[-1]["count"] += 1
            out[-1]["t_last"] = entry["t"]
        else:
            out.append({**entry, "count": 1, "t_last": entry["t"]})
    rendered = []
    for e in out:
        s = e["line"]
        if e["count"] > 1:
            s = f"{s} ×{e['count']}"
        rendered.append(s)
    return rendered


def _format_context(ctx):
    if not ctx:
        return ""
    parts = []
    for k in (
        "time",
        "day",
        "weekend",
        "sleep_duration_of_day",
        "location_id",
        "moving",
        "connection",
        "bluetooth",
        "schedule",
    ):
        if k in ctx and ctx[k] is not None:
            parts.append(f"{k}={ctx[k]}")
    return ", ".join(parts)


def _load_stm(stm_path):
    with open(stm_path) as f:
        stm = json.load(f)
    for x in stm:
        x["_t"] = _parse_time(x["time"])
    stm.sort(key=lambda x: x["_t"])
    return stm


def _slice_stm(stm, stm_times, slice_time, window_days):
    """O(log N) range slice via bisect on pre-sorted timeline."""
    lo_t = slice_time - timedelta(days=window_days)
    lo = bisect.bisect_left(stm_times, lo_t)
    hi = bisect.bisect_left(stm_times, slice_time)
    return stm[lo:hi]


def _stm_lines(entries, coalesce_repeats, drop_noise_types, humanize_pkg):
    kept = []
    for e in entries:
        if drop_noise_types and _is_noise(e):
            continue
        line = _serialize_entry(e, humanize_pkg=humanize_pkg)
        key = (e.get("dtype"), _pkg_key(e, humanize_pkg), e.get("type", ""))
        kept.append({"line": line, "key": key, "t": e["_t"]})
    if coalesce_repeats:
        return _coalesce(kept)
    return [k["line"] for k in kept]


def _budget_lines(lines, max_stm_tokens):
    """Drop oldest lines so joined char count fits 4 * max_stm_tokens.

    No iterative trimming: build cumulative prefix sum once, bisect for cut
    index in O(log N).
    """
    if not lines:
        return lines
    cum = list(accumulate(len(l) + 1 for l in lines))
    total = cum[-1]
    budget = max_stm_tokens * 4
    if total <= budget:
        return lines
    # Smallest cut such that cum[cut-1] >= total - budget.
    threshold = total - budget
    cut = bisect.bisect_left(cum, threshold) + 1
    return lines[cut:]


def _format_input(
    seq,
    i,
    descs,
    ctx,
    slice_time,
    trigger_time,
    stm,
    stm_times,
    *,
    stm_window_days,
    max_stm_tokens,
    stm_coalesce_repeats,
    stm_drop_noise_types,
    stm_humanize_pkg,
):
    stm_entries = _slice_stm(stm, stm_times, slice_time, stm_window_days)
    lines = _stm_lines(
        stm_entries, stm_coalesce_repeats, stm_drop_noise_types, stm_humanize_pkg
    )
    lines = _budget_lines(lines, max_stm_tokens)
    stm_block = "\n".join(lines) if lines else "(no recent activity)"
    ctx_block = _format_context(ctx)
    trigger_desc = descs[i] if i < len(descs) else ""
    trigger_time_str = trigger_time.strftime(TIME_FMT)

    parts = [
        f"[User Log History — last {stm_window_days} days, oldest → newest]",
        stm_block,
        "[/User Log History]",
        "",
    ]
    if ctx_block:
        parts += ["[Context]", ctx_block, "[/Context]", ""]
    parts += [
        f"[Trigger @ {trigger_time_str}]",
        trigger_desc,
        "[/Trigger]",
        "",
        "What is user's the most probable next action considering user log history and the current context?",
    ]
    return "\n".join(parts)


def build_samples(
    out_path: str = DEFAULT_SAMPLES_PATH,
    *,
    max_stm_tokens: int = 6000,
    stm_window_days: int = 7,
    stm_coalesce_repeats: bool = True,
    stm_drop_noise_types: bool = True,
    stm_humanize_pkg: bool = True,
    valid_frac: float = 0.05,
    valid_seed: int = 42,
):
    """Build samples from raw data files; write a single JSON to ``out_path``."""
    stm = _load_stm(STM_PATH)
    stm_times = [x["_t"] for x in stm]

    train_pool, test_pool = [], []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "[ab]*.json")))
    for fp in files:
        with open(fp) as f:
            routines = json.load(f)
        for routine in routines:
            for occ in routine.get("data", []):
                seq = occ.get("pattern_sequence", [])
                descs = occ.get("pattern_description", [])
                if len(seq) < 2 or len(descs) < 2:
                    continue
                n = min(len(seq), len(descs))
                reasoning = occ.get("reasoning", "") or ""
                ctx = occ.get("context", {}) or {}
                trigger_time = _parse_time(occ["time"])
                bucket = test_pool if trigger_time >= SPLIT_TIME else train_pool
                # Sliding-window: predict only the immediately-next item.
                # n items → n-1 samples (i → i+1 for i in 0..n-2).
                for i in range(n - 1):
                    s_i = seq[i]
                    is_context = ("dtype" not in s_i) or s_i.get("dtype") == "context"
                    slice_time = (
                        trigger_time if is_context else _parse_time(s_i["time"])
                    )
                    input_text = _format_input(
                        seq,
                        i,
                        descs,
                        ctx,
                        slice_time,
                        trigger_time,
                        stm,
                        stm_times,
                        stm_window_days=stm_window_days,
                        max_stm_tokens=max_stm_tokens,
                        stm_coalesce_repeats=stm_coalesce_repeats,
                        stm_drop_noise_types=stm_drop_noise_types,
                        stm_humanize_pkg=stm_humanize_pkg,
                    )
                    bucket.append(
                        {
                            "input_text": input_text,
                            "gold_answer": descs[i + 1],
                            "gold_reasoning": reasoning,
                        }
                    )

    rng = random.Random(valid_seed)
    idxs = list(range(len(train_pool)))
    rng.shuffle(idxs)
    n_valid = max(1, int(len(idxs) * valid_frac))
    valid_set = set(idxs[:n_valid])
    valid = [train_pool[k] for k in range(len(train_pool)) if k in valid_set]
    train = [train_pool[k] for k in range(len(train_pool)) if k not in valid_set]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"train": train, "valid": valid, "test": test_pool},
            f,
            ensure_ascii=False, indent=2
        )
    print(
        f"Saved {len(train)} train + {len(valid)} valid + {len(test_pool)} test "
        f"samples to {out_path}"
    )
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_SAMPLES_PATH)
    p.add_argument("--max_stm_tokens", type=int, default=3000)
    p.add_argument("--stm_window_days", type=int, default=3)
    p.add_argument("--no_coalesce", action="store_true")
    p.add_argument("--no_drop_noise", action="store_true")
    p.add_argument("--no_humanize_pkg", action="store_true")
    p.add_argument("--valid_frac", type=float, default=0.05)
    p.add_argument("--valid_seed", type=int, default=42)
    args = p.parse_args()

    build_samples(
        out_path=args.out,
        max_stm_tokens=args.max_stm_tokens,
        stm_window_days=args.stm_window_days,
        stm_coalesce_repeats=not args.no_coalesce,
        stm_drop_noise_types=not args.no_drop_noise,
        stm_humanize_pkg=not args.no_humanize_pkg,
        valid_frac=args.valid_frac,
        valid_seed=args.valid_seed,
    )


if __name__ == "__main__":
    main()
