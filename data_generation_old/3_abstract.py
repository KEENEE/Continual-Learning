"""Pre-abstract the compressed phone-usage log into LLM-friendly one-line events.

Reads:  user_usage_4weeks_compressed.json
Writes: user_usage_4weeks_abstracted.json

Each entry becomes a single dict {time, dtype, line} where ``line`` is a
human-readable, semantically rich, system-jargon-free description of one
event. Pre-computes what tasks/user_behavior's _serialize_entry does so the
build pipeline can skip per-event formatting.

Run:
    python 3_abstract.py
"""

import json
import os
import re
import unicodedata
from collections import defaultdict


# Unicode ranges for emoji / pictographs / dingbats / regional indicators / VS-16.
# Covers ⚠️ 🔔 📈 📉 💌 etc. Korean Hangul (U+AC00–U+D7A3) and CJK letters are
# outside these ranges, so this regex won't strip them.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric extended
    "\U0001F800-\U0001F8FF"   # arrows-C
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FA6F"   # chess
    "\U0001FA70-\U0001FAFF"   # symbols extended-A
    "\U00002300-\U000023FF"   # misc technical (⌚ ⌛ ⏰ ⏱ ⏳ etc.)
    "\U00002600-\U000026FF"   # misc symbols (☀ ☁ ⚠ etc.)
    "\U00002700-\U000027BF"   # dingbats (✂ ✈ ✉ etc.)
    "\U00002B00-\U00002BFF"   # misc symbols & arrows (⭐ ⬛ ⬜ etc.)
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U0000FE0F"              # variation selector-16
    "\U0000200D"              # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(s: str) -> str:
    """Remove emoji-class chars + Unicode format (Cf) and control (Cc) chars.

    Cf covers invisible bidi marks like LRM (U+200E), RLM, FSI/PDI (U+2068/U+2069),
    LRI/RLI/PDF, ZWJ/ZWNJ, BOM, etc.
    Cc covers ASCII control characters; titles shouldn't contain newlines/tabs
    after the upstream `.strip()` so this is safe.
    """
    if not s:
        return s
    s = _EMOJI_RE.sub("", s)
    s = "".join(c for c in s if unicodedata.category(c) not in ("Cf", "Cc"))
    return s.strip()


# ============================================================================
# APP_HUMAN — keys are the (truncated) app_name strings as they appear in the
# compressed log; values are LLM-friendly short names. Compiled by auditing
# user_usage_4weeks_compressed.json (count >= 2) against the original
# user_usage_4weeks.json (full package names).
# ============================================================================
APP_HUMAN = {
    # Toss / banking
    "토스": "Toss",
    "republica.toss": "Toss",
    "rainist.banksalad2": "BankSalad",
    "뱅크샐러드": "BankSalad",
    "wooriwm.txsmart": "Woori Bank",
    "우리WON뱅킹": "Woori Bank",
    "smart.npib": "Woori Bank",
    "wooriwm.mugsmart": "Woori Bank",
    "shinhan.sbanking": "Shinhan Bank",
    "신한은행": "Shinhan Bank",
    "kbstar.kbbank": "KB Bank",
    "KB스타뱅킹": "KB Bank",
    "knb.psb": "KNB Bank",
    "smart.banking": "NH Bank",
    "케이뱅크": "K Bank",
    "scxb.mbl": "Saemaeul Bank",
    "smg.spbs": "Saemaeul Bank",
    # Payment / wallet
    "android.spay": "Samsung Wallet",
    "삼성 월렛": "Samsung Wallet",
    "naverfin.payapp": "Naver Pay",
    "네이버페이": "Naver Pay",
    "kakaopay.app": "Kakao Pay",
    "카카오페이": "Kakao Pay",
    "samsungcard.mpocket": "Samsung Card",
    "android.smcard": "Samsung Card Monimo",
    "monimo": "Samsung Card Monimo",
    "삼성카드": "Samsung Card",
    "우리카드": "Woori Card",
    "wooricard.smartapp": "Woori Card",
    "신한카드": "Shinhan Card",
    "shcard.smartpay": "Shinhan Pay",
    "KB Pay": "KB Pay",
    "seoul.pay": "Seoul Pay",
    "app.cjonecard": "CJ One Card",
    "PASS": "PASS (auth)",
    "OK Cashbag": "OK Cashbag",
    "T 멤버십": "SKT Membership",
    # Real estate / housing
    "Hogangnono": "Hogangnono (real estate)",
    "hogangnono.hogangnono": "Hogangnono (real estate)",
    "공공주택 알리미": "Public Housing",
    "jun.public_housing": "Public Housing",
    "KB부동산": "KB Real Estate",
    "kbstar.land": "KB Real Estate",
    # Naver suite
    "android.nmap": "Naver Map",
    "android.map": "Kakao Map",
    "android.search": "Naver",
    "NAVER": "Naver",
    "navercorp.navershopping": "Naver Shopping",
    # Communication / messaging
    "kakao.talk": "KakaoTalk",
    "카카오톡": "KakaoTalk",
    "메시지": "Messages",
    "android.messaging": "Messages",
    "prod.dialer": "Call",
    "에이닷 전화": "Call",
    "android.gm": "Gmail",
    # Social
    "instagram.android": "Instagram",
    "instagram.barcelona": "Threads",
    "Threads": "Threads",
    "reddit.frontpage": "Reddit",
    # Knox / work
    "sds.teams": "Knox Teams",
    "Knox Teams": "Knox Teams",
    "퍼스널 데이터 인텔리전스": "Personal Data Intelligence",
    "SingleID Authenticator": "SingleID(Auth)",
    "singleid.pub": "SingleID(Auth)",
    # AI assistants
    "anthropic.claude": "Claude",
    "openai.chatgpt": "ChatGPT",
    "ChatGPT": "ChatGPT",
    "apps.bard": "Gemini",
    "빅스비": "Bixby",
    # Health
    "app.shealth": "Samsung Health",
    "삼성 헬스": "Samsung Health",
    "Health\xa0Connect": "Health Connect",
    "헬스\xa0커넥트": "Health Connect",
    "android.forest": "Digital Wellbeing",
    "디지털 웰빙": "Digital Wellbeing",
    "inbody2014.inbody": "InBody",
    "milk.periodapp": "Period Tracker",
    # Music / video
    "FLO": "FLO",
    "skplanet.musicmate": "FLO",
    "android.youtube": "YouTube",
    "YouTube": "YouTube",
    # Camera / gallery / photo
    "갤러리": "Gallery",
    "android.gallery3d": "Gallery",
    "갤러리 스토리": "Gallery Stories",
    "포토 에디터": "Photo Editor",
    "카메라": "Camera",
    "app.camera": "Camera",
    "무음 카메라": "Silent Camera",
    "peace.SilentCamera": "Silent Camera",
    "snowcorp.epik": "Epik",
    "campmobile.snow": "SNOW",
    "삼성 캡처": "Samsung Capture",
    "simplebestapp.camscanner": "CamScanner",
    # Productivity / files / docs
    "내 파일": "My Files",
    "app.myfiles": "My Files",
    "app.notes": "Samsung Notes",
    "todo.reminders": "Todo Reminders",
    "editors.sheets": "Google Docs",
    "apps.docs": "Google Docs",
    "HTML 뷰어": "HTML Viewer",
    "캘린더": "Calendar",
    # Shopping
    "N+스토어": "Naver Shopping",
    "무신사": "Musinsa",
    "musinsa.store": "Musinsa",
    "패밀리넷몰": "FamilyNet Mall",
    "samsung.familynetmall": "FamilyNet Mall",
    "kurly.m2": "Market Kurly",
    "kr.gmarket": "Gmarket",
    "com.elevenst": "11st",
    "원스토어": "SK ONE Store",
    "우리동네GS": "GS CVS",
    "배달의민족": "Baemin",
    "yanolja.nativeapp": "Yanolja",
    "mobile.consumer": "Agoda",
    "캐치테이블": "CatchTable",
    "cj.twosome": "TwosomePlace",
    "해피포인트": "Happy Point",
    "스타벅스": "Starbucks",
    "starbucks.co": "Starbucks",
    "com.clubartisee": "Artisee",
    "Google Play 스토어": "Google Play Store",
    "티머니매니저": "T-money",
    "카카오 T": "Kakao T",
    "com.tms": "SKT Membership",
    # Browser / search
    "android.chrome": "Chrome",
    "android.googlequicksearchbox": "Google Search",
    "app.sbrowser": "Samsung Internet",
    "삼성 인터넷": "Samsung Internet",
    "apps.maps": "Google Maps",
    # Phone system / settings / utility
    "Android 시스템": "Android System",
    "시스템 UI": "System UI",
    "Google Play 서비스": "Google Play",
    "android.gms": "Google Play",
    "디바이스 케어": "Device Care",
    "소프트웨어 업데이트": "Software Update",
    "삼성 계정": "Samsung Account",
    "삼성 클라우드": "Samsung Cloud",
    "AlwaysOnDisplay": "Always On Display",
    "긴급 상황 공유": "Emergency SOS",
    "긴급 SOS": "Emergency SOS",
    "인증서 관리자": "Certificate Manager",
    "AhnLab V3 Mobile Plus": "V3 Mobile (antivirus)",
    "ahnlab.v3mobileplus": "V3 Mobile (antivirus)",
    "블루투스": "Bluetooth",
    # Wearables / nearby
    "갤럭시 버즈": "Galaxy Buds",
    "Watch Manager": "Galaxy Watch",
    "app.watchmanager": "Galaxy Watch",
    "wearable.watchuniteplugin": "Galaxy Watch",
    "SmartThings": "SmartThings",
    "android.oneconnect": "SmartThings",
    "주변 디바이스 찾기": "Find Nearby Devices",
    # Misc
    "시계": "Alarm",
    "app.clockpackage": "Alarm",
    "날씨": "Weather",
    "samsung.planeat": "PlanEat",
    "minding.myroutine": "MyRoutine",
    "app.popupcalculator": "Calculator",
    "app.contacts": "Contacts",
    "android": "android",
    "com.minibigapp": "Minibig",
    "미니빅": "Minibig",
    "banhala.android": "banhala"
}

# Korean weekday → English (for calendar day_start).
WEEKDAY_KR_TO_EN = {
    "월요일": "Monday",
    "화요일": "Tuesday",
    "수요일": "Wednesday",
    "목요일": "Thursday",
    "금요일": "Friday",
    "토요일": "Saturday",
    "일요일": "Sunday",
}

# Sleep class → semantic verb.
SLEEP_CLASS = {
    "수면": "Fall asleep",
    "수면 끝": "Wake up",
    "낮잠": "Begin nap",
    "낮잠 끝": "End nap",
}

# Movement activity → semantic verb.
MOVEMENT = {
    "start walking": "Begin walking",
    "end walking": "Stop walking",
    "start running": "Begin running",
    "end running": "Stop running",
    "get on vehicle": "Board a vehicle",
    "get off vehicle": "Get off vehicle",
}

# Call type → semantic verb.
CALL_TYPE = {
    "outgoing": "Outgoing call",
    "incoming": "Incoming call",
    "incoming call end": "End incoming call",
    "outgoing call end": "End outgoing call",
    "missed": "Missed call",
    "rejected": "Reject call",
}


# ============================================================================
# Helpers
# ============================================================================

def _humanize_app_name(app_name):
    if app_name is None:
        return "?"
    if app_name in APP_HUMAN:
        return APP_HUMAN[app_name]
    if "." in app_name:
        return app_name.split(".")[-1]
    return app_name


def _app_label(item):
    return _humanize_app_name(item.get("app_name") or "")


def _truncate(s, n=80):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _simplify_duration(d):
    """Convert Korean duration like '5분 2초', '2시간 28초', '10시간 25분 27초'
    to English short form '5m 2s', '2h 28s', '10h 25m 27s'."""
    if not d:
        return d
    elif "시간" in d:
        return d.split("시간")[0]+"h"
    elif "분" in d:
        return d.split("분")[0]+"m"
    elif "초" in d:
        return d.replace("초", "s")


# ============================================================================
# Per-dtype abstractions
# ============================================================================

def abstract_app(item):
    app = _app_label(item)
    typ = item.get("type", "")
    # cls = item.get("class")
    # if cls:
    #     # Strip trailing "Activity" — keep the class name otherwise as-is.
    #     cls_short = cls[: -len("Activity")] if cls.endswith("Activity") else cls
    #     return f"Use {app} app ({cls_short})"
    return f"Use {app} app"


def _hangul_count(s: str) -> int:
    """Number of Hangul syllables (U+AC00–U+D7A3) in s."""
    return sum(1 for c in s if "가" <= c <= "힣")


_NUMBER_RE = re.compile(r"\d[\d,.]*")


def _strip_numbers(s: str) -> str:
    """Remove digit sequences (with commas/dots) and collapse extra spaces.
    Consolidates variants like '출금 100,000원' / '출금 50,000원' → '출금 원'."""
    s = _NUMBER_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def abstract_noti(item):
    """Produce a notification line.

    Two-stage title normalization:
      1) Strip digit sequences (incl. comma-grouped numbers) so that
         numeric-variant titles ('출금 100,000원' / '출금 50,000원') collapse
         to a single representative line ('출금 원').
      2) If the title still has ≥10 Hangul characters after stripping, drop it
         entirely (long titles are typically one-off promo/descriptive
         messages); keep only the sender app.
    """
    app = _humanize_app_name(item.get("app_name", ""))
    title = _strip_emoji((item.get("title") or "").strip())
    if not title:
        return f"Notification from {app}"
    title = _strip_numbers(title)
    if not title:
        return f"Notification from {app}"
    if _hangul_count(title) >= 10:
        return f"Notification from {app}"
    return f"Notification from {app}: \"{title}\""


def abstract_location(item):
    label = item.get("location_label", "") or ""
    # if label.startswith("Moved to "):
    #     label = label[len("Moved to "):]
    return label


def abstract_movement(item):
    activity = item.get("activity", "")
    duration = _simplify_duration(item.get("duration"))
    base = MOVEMENT.get(activity, activity)
    # if duration:
    #     return f"{base} (lasted {duration})"
    return base


def abstract_connection(item):
    cat = item.get("category", "")
    kind = item.get("event_kind", "")
    summary = item.get("summary")
    device = item.get("device_name")
    if cat == "NETWORK":
        net = "Wi-Fi" if summary == "WIFI" else (
            "Cellular network" if summary == "CELLULAR" else "Network"
        )
        if kind == "CONNECTED":
            return f"{net} connected"
        if kind == "DISCONNECTED":
            return f"{net} disconnected"
    if cat == "BLUETOOTH":
        if "DISCONNECT" in kind.upper():
            verb = "Disconnect Bluetooth"
        else:
            verb = "Connect Bluetooth"
        if device:
            return f"{verb} device ({device})"
        return verb
    return f"{cat} {kind}".strip()


def abstract_sleep(item):
    cls = item.get("class", "")
    duration = _simplify_duration(item.get("duration"))
    base = SLEEP_CLASS.get(cls, cls)
    # if duration:
    #     return f"{base} (slept {duration})" if "Wake" in base or "End" in base else base
    return base


def abstract_calendar(item):
    cls = item.get("class", "")
    title = item.get("title", "") or ""
    if cls == "day_start":
        weekday = WEEKDAY_KR_TO_EN.get(title, title)
        return f"Day starts: {weekday}"
    if cls == "user":
        return f"Calendar event: {title}"
    if cls == "system":
        return f"System calendar event: {title}"
    return f"Calendar: {title}"


def abstract_call(item):
    label = item.get("call_type_label", "")
    name = item.get("contact_name", "") or ""
    duration = _simplify_duration(item.get("duration"))
    if name == "저장되지 않은 번호":
        name = "unknown"
    base = CALL_TYPE.get(label, label)
    contact_part = f" with {name}" if "End" in base else (
        f" to {name}" if "Outgoing" in base else f" from {name}"
    )
    line = f"{base}{contact_part}"
    # if duration:
    #     line = f"{line} (duration: {duration})"
    return line


DISPATCH = {
    "app": abstract_app,
    "noti": abstract_noti,
    "location": abstract_location,
    "movement": abstract_movement,
    "connection": abstract_connection,
    "sleep": abstract_sleep,
    "calendar": abstract_calendar,
    "call": abstract_call,
}


def abstract(item):
    dt = item.get("dtype", "?")
    fn = DISPATCH.get(dt)
    if fn is None:
        return f"{dt} event"
    return fn(item)


# ============================================================================
# Main
# ============================================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(REPO_ROOT, "user_usage_4weeks_2_compressed.json")
OUT_PATH = os.path.join(REPO_ROOT, "user_usage_4weeks_3_abstracted.json")


def main():
    with open(IN_PATH) as f:
        data = json.load(f)

    out = []
    # by_dtype = defaultdict(list)
    for item in data:
        line = abstract(item)
        time = item.get("time", "")
        record = f"{time} | {line}"
        out.append(record)
        # by_dtype[item.get("dtype")].append((record, item))

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} abstracted events to {OUT_PATH}")

    # Print examples per dtype
    # print()
    # for dt in sorted(by_dtype.keys()):
    #     print(f"=== {dt} ({len(by_dtype[dt])} events) — first 10 examples ===")
    #     for record, raw in by_dtype[dt][:1]:
    #         print(f"  {record['time']}  | {record['line']}")
    #     print()


if __name__ == "__main__":
    main()
