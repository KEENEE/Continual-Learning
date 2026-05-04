# User Behavior — Phase 2: Preference-Learning Pipeline (rev 2)

## Status of Phase 1 (frozen)

Phase 1 (LLM-extracted routines → sliding-window samples) is implemented and the artifacts now live in `data_generation_old/`. Phase 2 replaces only the **data-generation pipeline** — `tasks/user_behavior.py` runtime loading stays compatible.

## Phase 2 Context

Old pipeline problems:
- Manual taxonomy + LLM extraction is expensive, biased, slow.
- Class imbalance was severe ("Open Toss app" 9.4% of train).
- No principled treatment of noise / preference shift / new-habit emergence.

New pipeline:
```
3_abstract.py  →  user_usage_4weeks_3_abstracted.json (list[str])
                  ↓
4_extract.py   →  data/user_behavior_samples.json
                  (regular train/valid/test  +  continual phase1/phase2 splits)
                  ↓
scripts/train_user_behavior.sh / train_user_behavior_continual.sh
```

---

## Why 4 categories (always · emergent · decaying · shifted)? — Test rationale

Each pattern is classified by its 4-week temporal dynamic to measure **continual-learning capability**. User preferences exhibit four motions: stable, novel, fading, shifting; the model must learn, retain, and update each appropriately.

| Category | Definition | Training/eval semantics |
|---|---|---|
| **always** | Active in all 4 weeks | Stable core preference. Baseline accuracy. |
| **decaying** | Active only weeks 15–16, gone after | User no longer does it. Model **must still remember** after phase-2 (forgetting probe). |
| **emergent** | First appears in weeks 17–18 | New habit. Phase-2 must **learn** it. |
| **shifted** | Same trigger·context but dominant gold differs between phases | Preference itself changed. Phase-2 must **replace** old gold with new. |

### Test scenarios

| When | always | decaying | emergent | shifted |
|---|---|---|---|---|
| After phase 1 | correct | correct (old gold) | unknown | correct (old gold) |
| After phase 2 | correct (kept) | **still correct** (catastrophic-forgetting probe) | correct (newly learned) | **replaced with new gold** (old can be forgotten) |

### Phase-2 metrics

- `acc/phase2_test/always` — stability (should match phase 1 level)
- `acc/phase2_test/decaying` — **catastrophic forgetting probe** (low ⇒ forgetting)
- `acc/phase2_test/shifted_new` — replacement learning (high ⇒ updated)
- `Δacc/phase1_test/shifted` — old-gold residual (negative expected)
- `acc/phase2_test/emergent` — new-learning capability

---

## Vocabulary

Sample 4-tuple: **trigger, context, gold_answer, reasoning**

- **trigger**: the event or context onset that initiates a pattern
- **context**: minimum conditions that characterize the pattern
- **gold_answer**: predicted next action
- **reasoning**: human-readable pattern explanation
- **occurrence**: number of observed pattern instances (renamed from `support`)
- **confidence**: P(gold | trigger, context)
- **frequent event**: a unique line that appears ≥3 times in 4 weeks

(Renames: `support` → `occurrence`, `context_bucket` → `context`. The word "anchor" is no longer used.)

---

## Confirmed input format

`user_usage_4weeks_3_abstracted.json` = `list[str]`. Each entry: `"YY-MM-DD HH:MM | <abstracted line>"`.
- 11,423 events; ISO weeks 14 (partial)–18 (partial); core 4 weeks = 15, 16, 17, 18.
- All app events use `Use ...` prefix → **cannot split anchor/background by Open vs Use**.

---

## Stage 2 — `4_extract.py`

### Step A — parse + sort
Parse each entry by ` | ` into `(time, line)`, sort by time, build a sorted timeline for bisect.

**Input → output example:**
```
"26-04-05 16:24 | Use KakaoTalk app"   →  (datetime(2026,4,5,16,24), "Use KakaoTalk app")
"26-04-05 16:24 | Begin nap"           →  (datetime(2026,4,5,16,24), "Begin nap")
```

### Step B — frequent-event filter

Count occurrences of each unique line. Keep only lines that
1. occur **≥3 times** in 4 weeks AND
2. are not in the **background blocklist** (system noise).

Background blocklist (editable, top-of-file):
```
Use Notification History          Use Settings / Settings Intelligence
Use Permission Manager / Controller   Use Android System / System UI
Use App Picker / Package Installer    Use Always On Display
Use Software Update / Device Care     Use V3 Mobile (antivirus)
Day starts: ...                       (calendar; context-only)
Cellular network connected/disconnected   (state churn)
Network connected/disconnected
```

→ **No Open/Use distinction** — all app events are `Use ...`. Filtering is by frequency + blocklist.

**Estimated counts:**
- ~11,423 events → ~1,100 unique lines → ~280 with ≥3 → ~250 frequent events after blocklist.
- These ~250 are the candidates for both **trigger** and **gold_answer**.

### Step C — coalesce repeats
Collapse **2 or more** identical lines within **2 minutes** into one occurrence. (Rev: was 3+; corrected to 2+.)

### Step D — context bucket

Each timestamp t carries 4 dims:

1. **day**: `weekday` | `weekend`
2. **hour_bin**: `early_morning(07–09)` | `morning(09–11)` | `lunch(11–13)` | `afternoon(13–15)` | `late_afternoon(15–18)` | `dinner(18–20)` | `evening(20–22)` | `night(22–24)` | `dawn(00–07)`
3. **location**: most-recent `Location <N>` (or `unknown`)
4. **movement**: from latest movement event — `walking` | `running` | `in_vehicle` | `stationary` (default)

**Example:**
```
t = 2026-04-21 18:42 (Mon, after Board a vehicle, last_loc = Location 10)
→ context = {day: weekday, hour_bin: dinner, location: "Location 10", movement: in_vehicle}
```

### Step E — pattern mining (sequential + context-only)

For each occurrence of frequent event X:

**(a) Sequential trigger**: if another frequent event T occurred within Δt=10 min, candidate pattern = `(trigger=T, context_at_X) → X`.

**(b) Context-only trigger**: if no frequent event within last 10 min, trigger = `"context onset"`. Candidate pattern = `(trigger="context onset", context_at_X) → X`. Captures patterns where the context itself drives behavior (e.g., *"every weekday lunch user opens Knox Teams"*).

For each candidate (trigger, context) → gold:
- `occurrence(trigger, context, gold) = count`
- `confidence(trigger, context, gold) = occurrence / sum_g occurrence(trigger, context, g)`

Patterns satisfy `occurrence ≥ 3` AND `confidence ≥ 0.30`.

**Sequential example:**
```
trigger=Connect Bluetooth device (Buds2)
context={day=weekday, hour_bin=dinner, movement=in_vehicle}
golds: Use FLO ×7, Use Naver Map ×1
→ pattern: occurrence=7, confidence=0.88   ✓
```

**Context-only example:**
```
trigger="context onset"
context={day=weekday, hour_bin=lunch, location=Location 1, movement=stationary}
first frequent events in such buckets: Use Knox Teams ×6, Use Toss ×1
→ pattern: occurrence=6, confidence=0.86   ✓
```

#### Step E.1 — minimum context (generalize-or-branch)

Start with full 4-dim context, then iteratively:

1. Try removing each dim D.
2. If dominant gold unchanged when D is removed → **drop D** (more general pattern).
3. If gold branches by D AND **each branch has occurrence ≥ 3, confidence ≥ 0.30** → **keep D, split** into multiple patterns (conditional preference — high training value).
4. If branched but only one side meets thresholds → keep that branch only, drop other.

Result: each pattern's context contains only the minimum dims required to characterize it.

**Branch example:**
```
trigger=Use Toss app
  ctx={weekday, morning} → Use Calculator (×5)
  ctx={weekday, lunch}   → Use BankSalad   (×4)
→ both branches qualify → split into 2 patterns
```

**Generalization example:**
```
trigger=Connect Bluetooth device (Buds2)
  weekday + walking: FLO ×3
  weekday + in_vehicle: FLO ×4
  weekend + walking: FLO ×2
→ gold same regardless of day/movement → drop both dims
→ pattern ctx = {} (only trigger needed)
```

### Step F — shifted detection (run BEFORE temporal categorization)

shifted is the centerpiece of CL evaluation, so we look for it early with a lenient context-equivalence rule.

For each (trigger, gold) pair:
1. Split occurrences by phase (weeks 15–16 vs 17–18).
2. Identify trigger T whose **dominant gold differs** between phases AND each side has occurrence ≥ 3.
3. The two contexts of those occurrences are considered "same context" if **≥2 of 4 dims match** (full equality is too strict).
4. Persist as TWO entries:
   - `(T, generalized_ctx, gold_old, weeks_15-16, category=shifted_old)`
   - `(T, generalized_ctx, gold_new, weeks_17-18, category=shifted_new)`

**Example:**
```
weeks 15-16: trigger=Use Toss, ctx={weekday, lunch, Loc1, stationary} → Use Calculator (×4)
weeks 17-18: trigger=Use Toss, ctx={weekday, lunch, Loc1, walking}    → Use BankSalad  (×4)
→ shared dims: day, hour_bin, location (3) ≥ 2 → shifted ✓
```

### Step G — temporal categorization

After shifted is fixed, classify the rest:

| Category | Rule |
|---|---|
| **always** | occurrence ≥ 1 in BOTH weeks 15-16 and 17-18, AND total occurrence ≥ 3 |
| **emergent** | weeks 15-16 occurrence = 0, weeks 17-18 occurrence ≥ 3 |
| **decaying** | weeks 15-16 occurrence ≥ 3, weeks 17-18 occurrence = 0 |
| **shifted** | (already classified in Step F) |

Patterns failing all four are dropped.
By definition, emergent has `first_week ≥ 17` → never appears in `phase1_train` (fixes prior bug).

**Pattern table example:**
```
P_001  always       Use Toss app, {weekday,morning} → Use Calculator    [15..18] occ=12 conf=0.75
P_002  emergent     context onset, {weekday,lunch} → Use Knox Teams      [17..18] occ=4  conf=0.67
P_003  decaying     Use FLO app, {dinner,in_vehicle} → Use Naver Map     [15..16] occ=5  conf=0.83
P_004  shifted_old  Use Toss app, {weekday,lunch} → Use Calculator       [15..16] occ=4  conf=0.80
P_004  shifted_new  Use Toss app, {weekday,lunch} → Use BankSalad        [17..18] occ=5  conf=0.83
```

### Step H — sample generation

Per pattern occurrence:
1. `slice_time` = occurrence time. Slice abstracted log for the last 7 days → `[User Log History]`.
2. Apply `_budget_lines` to fit `max_stm_tokens=6000`.
3. `[Context]` = pattern's generalized context dims (kept by Step E.1).
4. `[Trigger]` = trigger event line (or `Context onset: <ctx summary>` for context-only).
5. `gold_answer` = pattern's dominant gold.
6. `gold_reasoning` = auto-templated from pattern stats.
7. Attach `pattern_id`, `pattern_category`, `weeks_seen`, `occurrence`, `confidence`, `polarity=+1`, `weight`.

#### Negative samples (RL contrastive)
For each (trigger, context) with a frequent dominant gold, any **occurrence-1** alternative gold becomes a negative sample with `polarity=-1`. (Trigger, context) pairs that have no frequent dominant gold at all → drop all their occurrences entirely.

**Sample JSON example:**
```json
{
  "input_text": "[User Log History — last 7 days, oldest → newest]\n... (sliced log) ...\n[/User Log History]\n\n[Context]\nweekday, dinner, in_vehicle\n[/Context]\n\n[Trigger @ 26-04-21 18:42]\nConnect Bluetooth device (Buds2)\n[/Trigger]\n\nWhat is user's the most probable next action considering user log history and the current context?",
  "gold_answer": "Use FLO app",
  "gold_reasoning": "In abstracted history (4 weeks), 7/8 times after 'Connect Bluetooth device (Buds2)' during weekday dinner in_vehicle, the user opens FLO.",
  "polarity": +1,
  "pattern_id": "P_0042",
  "pattern_category": "always",
  "weeks_seen": [15, 16, 17, 18],
  "occurrence": 7,
  "confidence": 0.88,
  "weight": 1.0,
  "trigger_time": "26-04-21 18:42",
  "trigger_event": "Connect Bluetooth device (Buds2)",
  "context": "weekday, dinner, in_vehicle"
}
```

### Step I — splits

#### Regular split
- Test: trigger_time ≥ 2026-04-29 (last 5 days), category-stratified.
- Valid: 5% of remaining, random seed=42.
- Train: rest.

#### Continual learning split (rev 2 — per-category random)

Random seed=42 per-category split:

| Category | weeks 15-16 occurrences | weeks 17-18 occurrences |
|---|---|---|
| **always** | 90% phase1_train, 10% phase1_test | 90% phase2_train, 10% phase2_test/always |
| **shifted** | 90% phase1_train (old gold), 10% phase1_test (old gold) | 90% phase2_train (new gold), 10% phase2_test/shifted_new (new gold) |
| **decaying** | 80% phase1_train, 10% phase1_test, **10% phase2_test/decaying** (retention probe) | (none — by definition) |
| **emergent** | (none — by definition) | 90% phase2_train, 10% phase2_test/emergent |

Critical: `phase2_test/decaying` is sourced from weeks 15-16 of decaying patterns, NOT held out from phase2. This validates that **after phase-2 incremental training, the model still answers decaying patterns with their old gold** (no forgetting).

`phase2_test` is structured as `dict[str, list]` with 4 sub-buckets: `always`, `shifted_new`, `emergent`, `decaying`. Each is reported separately.

valid: 5% of phase{1,2}_train, separate from regular valid.

#### Phase flow

```
Phase 1:
  train: phase1_train  (always 90% + decaying 80% + shifted_old 90%)
  valid: phase1 train의 5%
  test:  phase1_test   (always 10% + decaying 10% + shifted_old 10%)
  → save ckpt A

Phase 2 (load_ckpt=A):
  train: phase2_train  (always 17-18 90% + emergent 90% + shifted_new 90%)
  valid: phase2 train의 5%
  test:  phase2_test (4 sub-buckets reported separately)
  → save ckpt B

Phase 2 reporting:
  acc/phase2_test/always       (stability)
  acc/phase2_test/shifted_new  (replacement learning)
  acc/phase2_test/emergent     (new learning)
  acc/phase2_test/decaying     (retention; should stay correct on OLD gold)

  Δacc/phase1_test/always      (≈0 expected)
  Δacc/phase1_test/shifted     (negative expected — old gold can be forgotten)
  Δacc/phase1_test/decaying    (≈0 expected — drop = catastrophic forgetting)
```

### Step J — CLI
```bash
python 4_extract.py \
  --input  user_usage_4weeks_3_abstracted.json \
  --output data/user_behavior_samples.json \
  --min_occurrence 3 --min_confidence 0.30 \
  --delta_t_min 10 --max_stm_tokens 6000 --stm_window_days 7 \
  --week_split_phase 17     # weeks <17 = phase1, weeks ≥17 = phase2
```

---

## Stage 3 — `tasks/user_behavior.py`

Add `mode` + `polarity_filter` args:
```python
def __init__(self, samples_path=DEFAULT_SAMPLES_PATH,
             mode="regular",                 # regular | continual_phase1 | continual_phase2
             polarity_filter="positive_only",# positive_only (SFT) | all (RL)
             ...):
    data = json.load(open(samples_path))
    if mode == "regular":
        train, valid, test = data["train"], data["valid"], data["test"]
    elif mode == "continual_phase1":
        c = data["continual"]
        train, valid, test = c["phase1_train"], c["phase1_valid"], c["phase1_test"]
    elif mode == "continual_phase2":
        c = data["continual"]
        train, valid = c["phase2_train"], c["phase2_valid"]
        self._phase2_test_buckets = c["phase2_test"]   # {always, shifted_new, emergent, decaying}
        test = sum(c["phase2_test"].values(), [])
        self._phase1_test = c["phase1_test"]
    if polarity_filter == "positive_only":
        train = [s for s in train if s.get("polarity", 1) == 1]

def get_sample_weight(self, sample):  return getattr(sample, "weight", 1.0)
def get_sample_polarity(self, sample): return getattr(sample, "polarity", 1)
```

`UserBehaviorSample` adds optional fields with defaults.

## Stage 4 — `tasks/user_behavior_eval.py`

Add stratified reporting:
```python
agg = {
    "sem_acc": ...,
    "sem_sim_mean": ...,
    "sem_acc/by_category": {"always":..., "emergent":..., "decaying":..., "shifted":...},
    "sem_acc/weighted": <support-weighted>,
}
```

`mode=continual_phase2`: in `svd_reinforce_hydra.py` test branch, evaluate each sub-bucket separately and the phase1_test for `Δphase1_acc/<category>` forgetting metrics.

## Stage 5 — Loss / reward

**SFT** (positive-only, weighted):
```python
weights = [task_loader.get_sample_weight(train_data[i]) for i in batch_ix]
W = sum(weights) or 1.0
scaled_loss = loss * weights[j] / W
```

**REINFORCE** (contrastive):
- `evaluate()` adds `polarity` to each `details[k]`.
- `get_rewards`: `polarity * (2*sim - 1)` per sample.
- PG: `pg = -log_likelihood * rewards[j] * weights[j]`.

## Stage 6 — Cleanup of `data_generation_old/user_behavior_build.py`
Remove: `_serialize_entry`, `_humanize_pkg`, `_pkg_key`, `_is_noise`, `_coalesce`, `_stm_lines`, `PKG_HUMAN`, `NOISE_TYPES`.
Keep: `_load_stm` (parse `"YY-MM-DD HH:MM | text"`), `_slice_stm`, `_format_context`, `_budget_lines`, `_format_input`, `build_samples`, `main`.

---

## Files

### Create
- `4_extract.py`
- `cfgs/task/user_behavior_continual_phase1.yaml`
- `cfgs/task/user_behavior_continual_phase2.yaml`
- `scripts/train_user_behavior_continual.sh`

### Modify
- `tasks/user_behavior.py`
- `tasks/user_behavior_eval.py`
- `optim_modules.py` (`SupervisedSFT`, `Reinforce` per-sample weight)
- `cfgs/task/user_behavior.yaml`
- `data_generation_old/user_behavior_build.py`
- `svd_reinforce_hydra.py` (~10-line phase-2 eval branch)

---

## Hyperparameters

| Param | Default | Note |
|---|---|---|
| `min_occurrence` | 3 | drop patterns with <3 occurrences in 4 weeks |
| `min_confidence` | 0.30 | `P(gold|trigger,context) ≥ 0.30` |
| `delta_t_min` | 10 | next frequent event must occur within 10 min |
| `max_stm_tokens` | 6000 | STM token budget per sample |
| `stm_window_days` | 7 | STM look-back window |
| `week_split_phase` | 17 | weeks <17 = phase1, weeks ≥17 = phase2 |
| `shifted_ctx_dims_min` | 2 | min context dims to match for shifted classification |
| `negative_weight` | 0.5 | multiplier for `polarity=-1` samples |
| `decaying_weight` | 0.3 | down-weight for decaying-pattern training samples |
| Random seed | 42 | reproducibility |

---

## Verification

1. **Unit smoke**: `python 4_extract.py` produces `data/user_behavior_samples.json` with category counts printed.
2. **Distribution sanity**: top-20 gold answers + categories. Top-1 share should drop well below 9.4%.
3. **Continual sanity**: emergent and shifted counts > 0.
4. **Regular smoke**: `bash scripts/train_user_behavior.sh num_iters=2 batch_size=2`. `sem_acc/by_category` keys appear.
5. **Continual smoke**: `bash scripts/train_user_behavior_continual.sh`. Phase 2 logs show `phase2_test/{always,shifted_new,emergent,decaying}` and `Δphase1_acc/<category>` forgetting metrics.
6. **gsm8k regression**: 1-iter run still passes.
