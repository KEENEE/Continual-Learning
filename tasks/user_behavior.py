import bisect
import glob
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import vllm
from fishfarm.models.vllm_model import VLLMModel

from .base import LLAMA3_COT, Task, get_download_dir
from .user_behavior_eval import UserBehaviorEvaluator


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
STM_PATH = os.path.join(REPO_ROOT, "user_usage_3weeks_compressed.json")

SPLIT_TIME = datetime(2026, 4, 20, 0, 0)
VALID_FRAC = 0.1
VALID_SEED = 42
TIME_FMT = "%y-%m-%d %H:%M"

NOISE_TYPES = {
    "KEYGUARD_HIDDEN",
    "KEYGUARD_SHOWN",
    "SCREEN_INTERACTIVE",
    "SCREEN_NON_INTERACTIVE",
    "USER_PRESENT",
    "BATTERY_LOW",
    "BATTERY_OKAY",
    "BATTERY_CHANGED",
}

PKG_HUMAN = {
    "com.google.android.youtube": "YouTube",
    "com.google.android.apps.youtube.music": "YouTubeMusic",
    "viva.republica.toss": "Toss",
    "토스": "Toss",
    "com.nhn.android.nmap": "NaverMap",
    "com.nhn.android.search": "NaverSearch",
    "com.nhn.android.naverlogin": "NaverLogin",
    "com.naver.linewebtoon": "LineWebtoon",
    "com.kakao.talk": "KakaoTalk",
    "kakao.talk": "KakaoTalk",
    "com.kakaopay.app": "KakaoPay",
    "com.instagram.android": "Instagram",
    "com.instagram.threadsapp": "Threads",
    "com.sec.android.app.clockpackage": "Clock",
    "시계": "Clock",
    "com.sec.android.app.launcher": "Launcher",
    "com.sec.android.app.camera": "Camera",
    "com.sec.android.gallery3d": "Gallery",
    "com.samsung.android.calendar": "Calendar",
    "com.samsung.android.app.contacts": "Contacts",
    "com.samsung.android.dialer": "Dialer",
    "com.samsung.android.app.notes": "Notes",
    "com.samsung.android.messaging": "Messages",
    "com.samsung.android.app.routines": "Routines",
    "com.samsung.android.spay": "SamsungWallet",
    "com.samsung.android.samsungpass": "SamsungPass",
    "com.samsung.android.knox.containeragent": "KnoxAgent",
    "com.samsung.knox.teams": "KnoxTeams",
    "Knox Teams": "KnoxTeams",
    "Samsung Food": "SamsungFood",
    "FLO": "FLO",
    "kr.co.skplanet.musicmate": "FLO",
    "musicmate": "FLO",
    "com.bithumb.android": "Bithumb",
    "com.starbucks.co": "Starbucks",
    "starbucks.co": "Starbucks",
    "com.naverpay": "NaverPay",
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
    if item.get("dtype") == "app" and item.get("type") in NOISE_TYPES:
        return True
    return False


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


@dataclass
class UserBehaviorSample:
    input_text: str
    gold_answer: str
    gold_reasoning: str
    trigger_time: datetime
    occurrence_id: str
    # Provide a `question` field so utils.classify_samples (if ever invoked)
    # has a consistent attribute to read.
    @property
    def question(self):
        return self.input_text


SYSTEM_MSG = (
    "You are a personal assistant that predicts a user's next mobile-phone action.\n"
    "Given the user's recent phone-usage history (short-term memory) and a current "
    "trigger event, output the single most likely next action as a short phrase.\n"
    "Reply with the action only — no explanation."
)


class UserBehaviorTask(Task):
    def __init__(
        self,
        use_reasoning_in_target: bool = False,
        max_stm_tokens: int = 6000,
        stm_window_days: int = 7,
        embed_threshold: float = 0.7,
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        stm_coalesce_repeats: bool = True,
        stm_drop_noise_types: bool = True,
        stm_humanize_pkg: bool = True,
        max_model_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        # Chat template: same Llama3-style template for all three; gemma can fall back to None.
        llama_tpl = LLAMA3_COT.replace("Let\\'s think step by step", "")
        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": llama_tpl,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": llama_tpl,
            "google/gemma-4-E4B": None,
        }
        self.system_msg = SYSTEM_MSG
        self.target_metric_train = "sem_acc"
        self.target_metric_valid = "sem_acc"
        self.target_metric_test = "sem_acc"
        self.target_metric_transfer = "sem_acc"
        self.has_transfer_split = False
        self.has_training_split = True

        self.use_reasoning_in_target = use_reasoning_in_target
        self.max_stm_tokens = max_stm_tokens
        self.stm_window_days = stm_window_days
        self.embed_threshold = embed_threshold
        self.embed_model_name = embed_model
        self.stm_coalesce_repeats = stm_coalesce_repeats
        self.stm_drop_noise_types = stm_drop_noise_types
        self.stm_humanize_pkg = stm_humanize_pkg
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens

        self._stm = self._load_stm()
        self.train_samples: List[UserBehaviorSample] = []
        self.valid_samples: List[UserBehaviorSample] = []
        self.test_samples: List[UserBehaviorSample] = []
        self._build_samples()

    # ----- loading -----

    def _load_stm(self):
        with open(STM_PATH) as f:
            stm = json.load(f)
        for x in stm:
            x["_t"] = _parse_time(x["time"])
        stm.sort(key=lambda x: x["_t"])
        # Pre-cache the sorted timeline for O(log N) bisect lookup.
        self._stm_times = [x["_t"] for x in stm]
        return stm

    def _slice_stm(self, slice_time):
        """O(log N) range slice via bisect on pre-sorted timeline."""
        lo_t = slice_time - timedelta(days=self.stm_window_days)
        lo = bisect.bisect_left(self._stm_times, lo_t)
        hi = bisect.bisect_left(self._stm_times, slice_time)
        return self._stm[lo:hi]

    # ----- STM serialization -----

    def _stm_lines(self, entries):
        kept = []
        for e in entries:
            if self.stm_drop_noise_types and _is_noise(e):
                continue
            line = _serialize_entry(e, humanize_pkg=self.stm_humanize_pkg)
            key = (e.get("dtype"), _pkg_key(e, self.stm_humanize_pkg), e.get("type", ""))
            kept.append({"line": line, "key": key, "t": e["_t"]})
        if self.stm_coalesce_repeats:
            return _coalesce(kept)
        return [k["line"] for k in kept]

    def _budget_lines(self, lines):
        """Greedy oldest-drop until char-count fits 4 * max_stm_tokens (rough estimate)."""
        budget = self.max_stm_tokens * 4
        total = sum(len(l) + 1 for l in lines)
        i = 0
        while total > budget and i < len(lines):
            total -= len(lines[i]) + 1
            i += 1
        return lines[i:]

    # ----- prompt construction -----

    def _format_input(self, seq, i, descs, ctx, slice_time, trigger_time):
        stm_entries = self._slice_stm(slice_time)
        lines = self._stm_lines(stm_entries)
        lines = self._budget_lines(lines)
        stm_block = "\n".join(lines) if lines else "(no recent activity)"
        ctx_block = _format_context(ctx)
        trigger_desc = descs[i] if i < len(descs) else ""
        trigger_time_str = trigger_time.strftime(TIME_FMT)

        parts = [
            "[Short-term memory — last {} days, oldest → newest]".format(
                self.stm_window_days
            ),
            stm_block,
            "[/Short-term memory]",
            "",
        ]
        if ctx_block:
            parts += ["[Context]", ctx_block, "[/Context]", ""]
        parts += [
            f"[Trigger @ {trigger_time_str}]",
            trigger_desc,
            "[/Trigger]",
            "",
            "What is the user's next action?",
        ]
        return "\n".join(parts)

    # ----- sample building -----

    def _build_samples(self):
        files = sorted(glob.glob(os.path.join(DATA_DIR, "[ab]*.json")))
        for fp in files:
            file_id = os.path.splitext(os.path.basename(fp))[0]
            with open(fp) as f:
                routines = json.load(f)
            for r_idx, routine in enumerate(routines):
                for o_idx, occ in enumerate(routine.get("data", [])):
                    seq = occ.get("pattern_sequence", [])
                    descs = occ.get("pattern_description", [])
                    if len(seq) < 2 or len(descs) < 2:
                        continue
                    n = min(len(seq), len(descs))
                    reasoning = occ.get("reasoning", "") or ""
                    ctx = occ.get("context", {}) or {}
                    trigger_time = _parse_time(occ["time"])
                    bucket = (
                        self.test_samples
                        if trigger_time >= SPLIT_TIME
                        else self.train_samples
                    )
                    for i in range(n):
                        # slice time: real log uses its own time; synthetic context uses occurrence time
                        s_i = seq[i]
                        is_synthetic = ("dtype" not in s_i) or s_i.get("dtype") == "context"
                        slice_time = (
                            trigger_time if is_synthetic else _parse_time(s_i["time"])
                        )
                        input_text = self._format_input(
                            seq, i, descs, ctx, slice_time, trigger_time
                        )
                        for j in range(i + 1, n):
                            bucket.append(
                                UserBehaviorSample(
                                    input_text=input_text,
                                    gold_answer=descs[j],
                                    gold_reasoning=reasoning,
                                    trigger_time=trigger_time,
                                    occurrence_id=f"{file_id}#{r_idx}#{o_idx}#{i}->{j}",
                                )
                            )
        # split test → valid (10%) / test (90%) using fixed seed
        rng = random.Random(VALID_SEED)
        idxs = list(range(len(self.test_samples)))
        rng.shuffle(idxs)
        n_valid = max(1, int(len(idxs) * VALID_FRAC))
        valid_set = set(idxs[:n_valid])
        new_test, new_valid = [], []
        for k, s in enumerate(self.test_samples):
            (new_valid if k in valid_set else new_test).append(s)
        self.test_samples = new_test
        self.valid_samples = new_valid

    # ----- Task interface -----

    def get_train_data(self):
        train_eval_samples = self.train_samples + self.valid_samples
        n_train = len(self.train_samples)
        n_valid = len(self.valid_samples)
        train_ix = list(range(n_train))
        valid_ix = list(range(n_train, n_train + n_valid))
        return train_eval_samples, train_ix, valid_ix

    def get_evaluator(self):
        train_eval = UserBehaviorEvaluator(
            samples=self.train_samples + self.valid_samples,
            system_msg=self.system_msg,
            threshold=self.embed_threshold,
            embed_model=self.embed_model_name,
        )
        test_eval = UserBehaviorEvaluator(
            samples=self.test_samples,
            system_msg=self.system_msg,
            threshold=self.embed_threshold,
            embed_model=self.embed_model_name,
        )
        return (train_eval, test_eval)

    def get_rewards(self, res):
        return [float(2 * x["sim"] - 1) for x in res.sample_details]

    def get_prompt(self, tokenizer, samples, ix, model_id):
        s = samples[ix]
        chat_template = self.model_to_template.get(model_id)
        return tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": s.input_text},
            ],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_gold_target(self, sample):
        if self.use_reasoning_in_target and sample.gold_reasoning:
            return f"{sample.gold_reasoning}\nNext action: {sample.gold_answer}"
        return sample.gold_answer

    def get_vllm_model(self, model_id, tensor_parallel_size=1):
        max_model_len = self.max_model_len or (
            8192 if "Llama-3-8B" in model_id else 16384
        )
        max_tokens = self.max_tokens or (256 if self.use_reasoning_in_target else 64)
        model = vllm.LLM(
            model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            download_dir=get_download_dir(),
        )
        chat_template = self.model_to_template.get(model_id)
        m = model.llm_engine.model_executor.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
                stop=["\n\n", "<|eot_id|>", "</s>"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
