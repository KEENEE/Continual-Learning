# python filter_app_usage_stats.py --input "{}.json" --word "apple,cat,dog"

import argparse
import json
from typing import Any, List, Tuple


def parse_words(raw: str) -> List[str]:
    return [w.strip().lower() for w in raw.split(",") if w.strip()]


def should_remove_dict(d: dict, pkg_needles: List[str], noti_text_needles: List[str]) -> bool:
    if pkg_needles:
        v = d.get("package")
        if isinstance(v, str) and any(n in v.lower() for n in pkg_needles):
            return True
    if noti_text_needles and d.get("data_type") == "notification":
        for k in ("text", "big_text", "title"):
            v = d.get(k)
            if isinstance(v, str) and any(n in v.lower() for n in noti_text_needles):
                return True
    return False


def prune(obj: Any, pkg_needles: List[str], noti_text_needles: List[str]) -> Tuple[bool, Any]:
    """
    Returns (keep, transformed_obj).
    keep=False means caller should remove this object.
    """
    if isinstance(obj, dict):
        if should_remove_dict(obj, pkg_needles, noti_text_needles):
            return False, None
        new_obj = {}
        for k, v in obj.items():
            keep, new_v = prune(v, pkg_needles, noti_text_needles)
            if keep:
                new_obj[k] = new_v
        return True, new_obj

    if isinstance(obj, list):
        new_list = []
        for item in obj:
            keep, new_item = prune(item, pkg_needles, noti_text_needles)
            if keep:
                new_list.append(new_item)
        return True, new_list

    return True, obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove dictionaries whose package contains target words.")
    parser.add_argument(
        "--input",
        default=r"user_usage_4weeks.json",
        help="Input JSON file path",
    )
    parser.add_argument(
        "--word",
        default="지우고 싶은 패키지명들",
        help='Comma-separated words matched against package, e.g. "abc,foo,bar"',
    )
    parser.add_argument(
        "--noti-word",
        default="지우고 싶은 알림 텍스트들",
        help='Comma-separated words matched against notification text/big_text/title.',
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    pkg_needles = parse_words(args.word)
    noti_text_needles = parse_words(args.noti_word)
    keep, filtered = prune(data, pkg_needles, noti_text_needles)
    if not keep:
        filtered = {}

    out_path = args.input.split('.')[0] + '_1_filtered.json'
    # out_path = "timeline_filtered.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved filtered JSON to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
