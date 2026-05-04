# User Behavior — Phase 2: 사용자 선호 학습 파이프라인 (rev 2)

## Phase 1 상태 (구현 완료, freeze)

Phase 1 (LLM 기반 routine 추출 → sliding-window 학습 샘플 생성)은 이미 구현되어 있고, 관련 산출물은 `data_generation_old/`로 이동됨. `tasks/user_behavior.py` + `tasks/user_behavior_eval.py` + cfgs는 이미 JSON-cache 로딩 구조를 갖추고 있음. **Phase 2는 데이터 생성 파이프라인만 교체**하고 runtime task 로딩은 그대로 호환 유지.

## Phase 2 배경

기존 파이프라인의 문제:
- Manual taxonomy + LLM 추출은 비싸고 중복/편향이 많음
- Class imbalance 심함 ("Open Toss app" train의 9.4%)
- 노이즈 / 선호 변동 / 신규 습관 출현에 대한 원칙적 처리 없음
- 로그가 늘어날 때마다 LLM 재호출 → 확장성 부족

새 파이프라인:
```
3_abstract.py  →  user_usage_4weeks_3_abstracted.json (문자열 리스트)
                  ↓
4_extract.py   →  data/user_behavior_samples.json
                  (regular train/valid/test  +  continual phase1/phase2 분할)
                  ↓
scripts/train_user_behavior.sh / train_user_behavior_continual.sh
```

---

## 왜 4가지 카테고리(always · emergent · decaying · shifted)인가? — 테스트 전략

각 패턴을 4주에 걸친 시간적 동역학에 따라 분류해 **continual learning 능력**을 측정한다. 사용자의 행동·선호는 안정·신규·소멸·변화의 네 형태를 보이고, 모델은 이를 적절히 학습·기억·갱신해야 한다.

| 카테고리 | 정의 | 학습/평가 의미 |
|---|---|---|
| **always** | 4주 모두 등장 | 안정적 핵심 선호. 기준 정확도. |
| **decaying** | weeks 15-16에만 등장, 이후 사라짐 | 사용자가 더 안 하는 행동. **Phase 2 학습 후에도 *기억*은 해야 함** (forgetting 측정 대상). |
| **emergent** | weeks 17-18에만 처음 등장 | 새 습관. **Phase 2에서 새로 학습하는 능력**. |
| **shifted** | 같은 trigger·context인데 phase에 따라 dominant gold가 바뀜 (예: Toss → BankSalad) | 선호 자체가 변함. **Phase 2에서 새 gold로 *대체* 학습하는 능력**. |

### 테스트 시나리오

| 시점 | always | decaying | emergent | shifted |
|---|---|---|---|---|
| Phase 1 후 | 정확 | 정확 (옛 gold) | 모름 | 정확 (옛 gold) |
| Phase 2 후 | 정확 (안 잊음) | **여전히 정확** (옛 gold 유지 — CF 방지) | 정확 (새로 학습) | **새 gold로 갱신** (옛 gold는 잊어도 OK) |

### 측정 metric (Phase 2 후)

- `acc/phase2_test/always` — 안정성 (Phase 1 수준 유지해야 함)
- `acc/phase2_test/decaying` — **Catastrophic forgetting 척도** (낮으면 forgetting 발생)
- `acc/phase2_test/shifted_new` — 새 gold 갱신 학습 (높을수록 좋음)
- `acc/phase1_test/shifted` 변화량 — 옛 gold 잔존 정도 (낮아져야 정상)
- `acc/phase2_test/emergent` — 새 학습 능력

---

## Vocabulary

학습 데이터 4-tuple: **trigger, context, gold_answer, reasoning**

- **trigger**: 패턴을 시작하는 event 또는 context onset
- **context**: 패턴을 정의하는 최소 조건들
- **gold_answer**: 예측해야 할 다음 행동
- **reasoning**: 자연어로 작성된 패턴 설명
- **occurrence**: 패턴이 관찰된 횟수 (이전 plan의 `support`와 동일 — 변수명 통일)
- **confidence**: P(gold | trigger, context)
- **frequent event**: 4주에 ≥3회 등장한 unique line

(이전 명칭 `support` → `occurrence`, `context_bucket` → `context`. "anchor"라는 용어는 사용 안 함.)

---

## 입력 포맷 (확인됨)

`user_usage_4weeks_3_abstracted.json` = `list[str]`. 각 항목: `"YY-MM-DD HH:MM | <abstracted line>"`.
- 11,423 events; ISO weeks 14 (부분)–18 (부분); 핵심 4주 = 15, 16, 17, 18.
- 모든 라인이 자연어 (예: `"Use KakaoTalk app"`, `"Notification from Toss: …"`, `"Move to Location 10"`).
- **모든 app 이벤트가 "Use ..." prefix를 사용** (Open vs Use로 anchor/background를 가를 수 없음).

---

## Stage 2 — `4_extract.py` 알고리즘

### Step A — parse + sort

abstracted log 11,423줄 → `(time, line)` 튜플 리스트, 시간 정렬.

**입력 예시:**
```
"26-04-05 16:24 | Use KakaoTalk app"
"26-04-05 16:24 | Use Starbucks app"
"26-04-05 16:24 | Begin nap"
...
```

**출력 (parsed):**
```python
[
  (datetime(2026,4,5,16,24), "Use KakaoTalk app"),
  (datetime(2026,4,5,16,24), "Use Starbucks app"),
  (datetime(2026,4,5,16,24), "Begin nap"),
  ...
]
```

### Step B — frequent event 필터링

각 unique line의 등장 횟수를 카운트한 뒤 다음 두 조건을 통과한 line만 **frequent event**로 통과시킨다 (= 패턴의 trigger 또는 gold_answer 후보가 됨):

1. 4주에 걸쳐 **3회 이상** 등장
2. background blocklist에 없음

**Background blocklist** (시스템 노이즈, 편집 가능):
```
Use Notification History
Use Notification History (...)
Use Settings
Use Suggested Settings
Use Settings Intelligence
Use Permission Manager
Use Permission Controller
Use Android System
Use System UI
Use App Picker
Use Package Installer
Use Always On Display
Use Software Update
Use Device Care
Use V3 Mobile (antivirus)
Day starts: ...                  (calendar; context로만 사용)
Cellular network connected/disconnected   (high-frequency 상태 변화)
Network connected/disconnected
```

→ `Open / Use` 구분으로 anchor/background를 나누는 방식은 **사용하지 않음** (모든 app 이벤트가 `Use`로 작성됨).

**예시 통계 (예상):**
- 전체: 11,423 events, ~1,100 unique lines
- ≥3회 등장: ~280 unique lines
- blocklist 제거 후: ~250 unique frequent events
- 이 ~250개가 trigger 후보 AND gold_answer 후보

**frequent event 예시 (등장 횟수):**
```
Use Toss app                          387
Use KakaoTalk app                     263
Use FLO app                           198
Notification from Toss: "..."          84
Move to Location 10                    64
Connect Bluetooth device (Buds2)       28
Wake up (slept ...)                    36
Fall asleep                            38
...
```

### Step C — coalesce 반복

**2개 이상**의 동일 line이 **2분 이내**에 연속 등장하면 하나의 occurrence로 통합. (3개 이상 → 2개 이상으로 수정)

**예시:**
```
입력:
  04-12 12:00  Use KakaoTalk app
  04-12 12:00  Use KakaoTalk app
  04-12 12:08  Use KakaoTalk app    ← 8분 후, 별도

출력:
  04-12 12:00  Use KakaoTalk app  (×2 통합)
  04-12 12:08  Use KakaoTalk app  (×1)
```

### Step D — context bucket 추출

각 시점 t에서 4개 dim 결정:

1. **day**: `weekday` | `weekend`
2. **hour_bin**:
   - `early_morning` (07–09)
   - `morning` (09–11)
   - `lunch` (11–13)
   - `afternoon` (13–15)
   - `late_afternoon` (15–18)
   - `dinner` (18–20)
   - `evening` (20–22)
   - `night` (22–24)
   - `dawn` (00–07)
3. **location**: 가장 최근 location 이벤트 기준 (`Location <N>` 또는 `unknown`)
4. **movement**: 가장 최근 movement 이벤트 기준
   - `walking`: 직전 `Begin walking` 이후 `Stop walking` 전
   - `running`: 직전 `Begin running` 이후 `Stop running` 전
   - `in_vehicle`: 직전 `Board a vehicle` 이후 `Get off vehicle` 전
   - `stationary`: 그 외 (default)

**예시:**
```
시점 t = 2026-04-21 18:42 (월요일, 직전 'Board a vehicle' 발생, last location = Location 10)
→ context = {day: weekday, hour_bin: dinner, location: "Location 10", movement: in_vehicle}
```

### Step E — 패턴 마이닝 (sequential + context-only)

각 frequent event X (Step B 통과) 발생 시점마다 trigger를 다음 두 방식으로 후보화:

**(가) Sequential trigger**: 직전 Δt(=10분) 이내에 **다른 frequent event T**가 있으면 후보 패턴 = `(trigger=T, context_at_X) → X`.

**(나) Context-only trigger**: 직전 10분 이내 다른 frequent event가 **없으면** trigger를 "context onset"으로 처리. 후보 패턴 = `(trigger="context onset", context_at_X) → X`. context 자체가 사용자 행동을 유도한 셈.

각 후보 (trigger, context) → gold 조합에 대해:
- `occurrence(trigger, context, gold)` = 해당 조합이 관찰된 횟수
- `confidence(trigger, context, gold)` = `occurrence / sum(occurrence over all golds for same (trigger, context))`

**임계값**: `occurrence ≥ 3` AND `confidence ≥ 0.30`. 두 조건 모두 충족하는 (trigger, context, gold)이 패턴.

**예시 — sequential pattern:**
```
trigger:  "Connect Bluetooth device (Buds2)"
context:  {day: weekday, hour_bin: dinner, movement: in_vehicle}
golds 분포:
  Use FLO app           ×7
  Use Naver Map app     ×1
→ occurrence = 7, confidence = 7/8 = 0.88   ✓
```

**예시 — context-only pattern:**
```
trigger:  "context onset"
context:  {day: weekday, hour_bin: lunch, location: "Location 1", movement: stationary}
golds 분포 (해당 context 시작 후 첫 frequent event):
  Use Knox Teams app     ×6
  Use Toss app           ×1
→ occurrence = 6, confidence = 6/7 = 0.86   ✓ ("평일 점심 → 카카오톡" 같은 사용자 습관)
```

#### Step E.1 — minimum context (context generalization with branching)

처음에는 4개 dim 모두 포함하는 full context로 카운트 → 그 후 다음 절차로 context를 최소화:

1. 각 dim D를 후보로 선택해 그 dim을 무시하고 패턴들을 합쳐본다.
2. 합친 결과의 dominant gold가 변하지 않으면 → **D를 context에서 제거** (more general pattern).
3. 합친 결과의 dominant gold가 D 값에 따라 갈리고, **각 분기 모두가 occurrence ≥ 3, confidence ≥ 0.30**이면 → D를 유지하고 두 패턴으로 분리.
4. 분기인데 한쪽만 임계값 충족하면 → 충족하는 쪽만 패턴으로 유지, 다른 쪽 drop.

**예시 — 분기 사례 (D 유지):**
```
trigger: "Use Toss app"
  context (weekday, morning):    gold=Use Calculator app   (×5)
  context (weekday, lunch):      gold=Use BankSalad app    (×4)
→ 두 분기 모두 occurrence ≥3 → 분기 유지: 시간대로 다른 gold
  → 패턴 P_A: trigger="Use Toss app", ctx={weekday, morning}, gold="Use Calculator app"
  → 패턴 P_B: trigger="Use Toss app", ctx={weekday, lunch}, gold="Use BankSalad app"
```

**예시 — 일반화 사례 (D 제거):**
```
trigger: "Connect Bluetooth device (Buds2)"
  weekday + walking:    gold=Use FLO  ×3
  weekday + in_vehicle: gold=Use FLO  ×4
  weekend + walking:    gold=Use FLO  ×2
→ movement·day 무관 모두 FLO → 두 dim 모두 제거
  → 패턴 P_C: trigger="Connect Bluetooth device (Buds2)", ctx={}, gold="Use FLO"
```

→ 결과: 각 패턴은 그것을 정의하는 **최소 context dim**만 가짐.

### Step F — shifted 감지 (먼저 분류)

shifted는 continual learning 평가의 핵심이므로 카테고리화 전에 **먼저** 후보를 찾는다. context 동일성 판단을 다소 느슨하게 잡아 shifted 후보가 충분히 잡히도록.

각 (trigger, gold) 쌍에 대해:

1. weeks 15-16의 occurrence와 weeks 17-18의 occurrence 분리.
2. 두 시기 각각에서 같은 trigger의 dominant gold 확인.
3. 같은 trigger인데 두 시기의 dominant gold가 다르고, **각 시기 occurrence ≥ 3**이면 후보.
4. 두 시기의 context 비교: **4개 dim 중 2개 이상 일치**하면 같은 패턴의 shifted로 분류 (context 완전 동일은 너무 엄격).
5. shifted 패턴은 **두 entry**로 저장:
   - `(trigger, generalized_ctx, gold_old, weeks_15-16, category=shifted_old)`
   - `(trigger, generalized_ctx, gold_new, weeks_17-18, category=shifted_new)`

**예시:**
```
weeks 15-16: trigger="Use Toss app", ctx={weekday, lunch, Location 1, stationary}, gold=Use Calculator (×4)
weeks 17-18: trigger="Use Toss app", ctx={weekday, lunch, Location 1, stationary}, gold=Use BankSalad (×4)
→ context 4 dim 모두 일치 → shifted ✓
```

```
weeks 15-16: trigger="Wake up", ctx={weekday, early_morning, Location 1, stationary},   gold=Use KakaoTalk (×3)
weeks 17-18: trigger="Wake up", ctx={weekday, early_morning, Location 1, walking},      gold=Use Toss     (×4)
→ context dim 중 day, hour_bin, location 일치 (3개) → shifted ✓ (movement 다름은 무시)
```

### Step G — temporal categorization

shifted 분류 후, 나머지 (trigger, generalized_context, gold) 패턴들을 다음 규칙으로 분류:

| Category | 규칙 |
|---|---|
| **always** | weeks 15-16, 17-18 양쪽 모두 occurrence ≥ 1 (4주 내내 활성) AND 전체 occurrence ≥ 3 |
| **emergent** | weeks 15-16에 occurrence = 0 AND weeks 17-18에 occurrence ≥ 3 |
| **decaying** | weeks 15-16에 occurrence ≥ 3 AND weeks 17-18에 occurrence = 0 |
| **shifted** | (Step F에서 이미 분류) |

→ 위 4 카테고리에 들지 않는 (예: weeks 15-16 occurrence=2, 17-18 occurrence=2) 패턴은 임계값 미달로 drop.
→ emergent의 정의상 `first_week ≥ 17`이므로 phase1_train에는 emergent 절대 안 들어감 (이전 plan 오류 수정).

**예시 결과:**
```
P_001  always       trigger="Use Toss app", ctx={weekday, morning},
                    gold="Use Calculator app", weeks_seen=[15,16,17,18],
                    occurrence=12, confidence=0.75
P_002  emergent     trigger="context onset", ctx={weekday, lunch},
                    gold="Use Knox Teams app", weeks_seen=[17,18],
                    occurrence=4, confidence=0.67
P_003  decaying     trigger="Use FLO app", ctx={dinner, in_vehicle},
                    gold="Use Naver Map app", weeks_seen=[15,16],
                    occurrence=5, confidence=0.83
P_004  shifted_old  trigger="Use Toss app", ctx={weekday, lunch},
                    gold="Use Calculator app", weeks_seen=[15,16],
                    occurrence=4, confidence=0.80
P_004  shifted_new  trigger="Use Toss app", ctx={weekday, lunch},
                    gold="Use BankSalad app", weeks_seen=[17,18],
                    occurrence=5, confidence=0.83
```

### Step H — sample 생성

각 패턴의 occurrence마다 한 개의 sample을 생성:

1. `slice_time` = 그 occurrence의 발생 시각. abstracted log를 직전 7일 슬라이스 (bisect on times). → `[User Log History]` 블록.
2. `_budget_lines` 적용해 `max_stm_tokens=6000` 예산 내로.
3. `[Context]` 블록: 그 패턴의 generalized context dim들만 (Step E.1에서 남긴 것) 표시.
4. `[Trigger]` 블록: trigger event line, 또는 context-only 패턴이면 `Context onset: <generalized context summary>`.
5. `gold_answer` = 패턴의 dominant gold.
6. `gold_reasoning` = 패턴 통계 기반 자동 생성. 예: *"In abstracted history (4 weeks), 7/8 times after 'Connect Bluetooth device (Buds2)' during weekday dinner in_vehicle, the user opens FLO."*
7. 메타데이터 부착: `pattern_id`, `pattern_category`, `weeks_seen`, `occurrence`, `confidence`, `polarity=+1`, `weight`.

#### Negative samples (RL용 contrastive)

각 (trigger, generalized_context)에 dominant gold가 있을 때, 같은 (trigger, generalized_context)에서 **occurrence = 1**인 다른 gold(드문 대안)를 negative sample로 생성. `polarity = -1`, `gold_answer = 그 드문 대안`.

(Trigger, context) 자체가 frequent하지 않은 (즉 dominant gold가 없는) 일회성 occurrence는 완전 drop.

**최종 sample JSON 예시:**
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

#### Regular split (default training)
- Test: trigger_time ≥ 2026-04-29 (마지막 5일) occurrence held-out, 카테고리 stratified.
- Valid: 나머지의 5% random (seed=42).
- Train: 나머지.

#### Continual learning split (rev 2)

각 패턴 카테고리의 occurrence를 phase별로 random 분배 (seed=42):

| Category | weeks 15-16 occurrence | weeks 17-18 occurrence |
|---|---|---|
| **always** | 90% → `phase1_train`, 10% → `phase1_test` | 90% → `phase2_train`, 10% → `phase2_test/always` |
| **shifted** | 90% → `phase1_train` (옛 gold), 10% → `phase1_test` (옛 gold) | 90% → `phase2_train` (새 gold), 10% → `phase2_test/shifted_new` (새 gold) |
| **decaying** | 80% → `phase1_train`, 10% → `phase1_test`, **10% → `phase2_test/decaying`** (retention 검증) | (없음) |
| **emergent** | (없음 — 정의상 없음) | 90% → `phase2_train`, 10% → `phase2_test/emergent` |

> **중요**: `phase2_test/decaying`은 weeks 15-16의 decaying 패턴 occurrence 중 10%이며, **phase 1 학습 후 phase 2 학습 후에도 옛 gold를 잘 답해야 함** (catastrophic forgetting 검증).

`phase2_test`는 출처 카테고리별 4개 sub-bucket으로 나누어 따로 정확도 보고:
- `phase2_test/always` (weeks 17-18 always)
- `phase2_test/shifted_new` (weeks 17-18 shifted 새 gold)
- `phase2_test/emergent` (weeks 17-18 emergent)
- `phase2_test/decaying` (weeks 15-16 decaying — retention 검증)

valid는 phase별 train의 5%를 따로 떼어 best-checkpoint 선택용으로 사용 (regular의 valid와 별도).

#### Phase별 학습/평가 흐름

```
Phase 1 학습:
  train: phase1_train  (always 90% + decaying 80% + shifted_old 90%)
  valid: phase1_train의 5%
  test:  phase1_test   (always 10% + decaying 10% + shifted_old 10%)
  → checkpoint A 저장

Phase 2 학습 (load_ckpt=A):
  train: phase2_train  (always weeks 17-18 90% + emergent 90% + shifted_new 90%)
  valid: phase2_train의 5%
  test:  phase2_test (4개 sub-bucket으로 분리 보고)
  → checkpoint B 저장

Phase 2 후 평가 보고:
  acc/phase2_test/always       (안정 유지)
  acc/phase2_test/shifted_new  (전환 학습)
  acc/phase2_test/emergent     (신규 학습)
  acc/phase2_test/decaying     (retention; 옛 gold 유지)

  Δacc/phase1_test/always      (≈0 expected)
  Δacc/phase1_test/shifted     (음수 expected — 옛 gold는 잊어도 OK)
  Δacc/phase1_test/decaying    (≈0 expected — 잊으면 catastrophic forgetting)
```

### Step J — CLI

```bash
python 4_extract.py \
  --input  user_usage_4weeks_3_abstracted.json \
  --output data/user_behavior_samples.json \
  --min_occurrence 3 --min_confidence 0.30 \
  --delta_t_min 10 --max_stm_tokens 6000 --stm_window_days 7 \
  --week_split_phase 17     # phase 경계: weeks <17 = phase1, weeks ≥17 = phase2
```

---

## Stage 3 — `tasks/user_behavior.py` 수정

`UserBehaviorTask.__init__`에 `mode` arg + `polarity_filter` arg 추가:

```python
def __init__(self, samples_path=DEFAULT_SAMPLES_PATH,
             mode="regular",                 # "regular" | "continual_phase1" | "continual_phase2"
             polarity_filter="positive_only",# "positive_only" (SFT) | "all" (RL)
             use_reasoning_in_target=False,
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
        # phase2 test는 sub-bucket dict
        self._phase2_test_buckets = c["phase2_test"]   # {always, shifted_new, emergent, decaying}
        # 호환성을 위해 평가용 default test는 모두 합친 것
        test = sum(c["phase2_test"].values(), [])
        # phase 1 test도 보존 (forgetting 측정용)
        self._phase1_test = c["phase1_test"]
    if polarity_filter == "positive_only":
        train = [s for s in train if s.get("polarity", 1) == 1]
    ...

def get_sample_weight(self, sample):       # SupervisedSFT가 읽음
    return getattr(sample, "weight", 1.0)
def get_sample_polarity(self, sample):     # Reinforce가 읽음
    return getattr(sample, "polarity", 1)
```

---

## Stage 4 — `tasks/user_behavior_eval.py` 수정

기존 semantic-similarity 로직 그대로. **stratified 리포팅** 추가:

```python
agg = {
    "sem_acc": ...,
    "sem_sim_mean": ...,
    "sem_acc/by_category": {                 # NEW
        "always":   <acc>,
        "emergent": <acc>,
        "decaying": <acc>,
        "shifted":  <acc>,
    },
    "sem_acc/weighted": <support-weighted acc>,  # NEW
}
```

`mode=continual_phase2`인 경우 svd_reinforce_hydra.py 평가 분기에서 다음을 추가 평가:
- `phase2_test/always`, `phase2_test/shifted_new`, `phase2_test/emergent`, `phase2_test/decaying` 각각 따로 sem_acc 계산
- `phase1_test` 평가하여 `phase1_acc_after_phase2` 보고
- `Δphase1_acc/<category>` (phase 2 학습 전후 비교)

---

## Stage 5 — Loss / reward 수정

#### SFT (positive-only; weighted)
```python
weights = [task_loader.get_sample_weight(train_data[i]) for i in batch_ix]
W = sum(weights) or 1.0
for j, (prompt, gold) in enumerate(zip(prompts, golds)):
    ...
    scaled_loss = loss * weights[j] / W
    scaled_loss.backward()
```

#### REINFORCE (contrastive: positive + negative)
```python
# tasks/user_behavior.py – get_rewards:
def get_rewards(self, res):
    return [
        float(x.get("polarity", 1)) * (2.0 * x["sim"] - 1.0)
        for x in res.sample_details
    ]
```
`UserBehaviorEvaluator.evaluate`에서 each `details[k]`에 `polarity` 추가.
`Reinforce.step_optimization`에서 `pg = -log_likelihood * rewards[j] * weights[j]`.

순 효과:
- Positive 샘플, 모델이 gold 매치 → reward ≈ +1, gradient가 gold 쪽으로 push.
- Negative 샘플, 모델이 rare-gold 매치 → reward ≈ −1, gradient가 rare-gold에서 멀어짐.
- Negative 샘플, 다른 것 선택 → reward ≈ +1 (sim 낮음). 우발적 페널티 없음.

---

## Stage 6 — `data_generation_old/user_behavior_build.py` 정리

제거:
- `_serialize_entry`, `_humanize_pkg`, `_pkg_key`, `_is_noise`, `_coalesce`, `_stm_lines`
- `PKG_HUMAN`, `NOISE_TYPES`

유지:
- `_load_stm` (단, abstracted JSON 파싱: `"YY-MM-DD HH:MM | text"` 리스트), `_slice_stm`, `_format_context`, `_budget_lines`, `_format_input`, `build_samples`, `main`

---

## 파일

### 생성
- `4_extract.py` — 패턴 마이너 + 샘플 빌더
- `cfgs/task/user_behavior_continual_phase1.yaml` — `mode: continual_phase1`
- `cfgs/task/user_behavior_continual_phase2.yaml` — `mode: continual_phase2`
- `scripts/train_user_behavior_continual.sh` — phase1 다음 phase2를 `load_ckpt`로 실행

### 수정
- `tasks/user_behavior.py` — `mode`, `polarity_filter`, sample 메타데이터 필드, `get_sample_weight`, `get_sample_polarity`
- `tasks/user_behavior_eval.py` — stratified `sem_acc/by_category`, weighted accuracy
- `optim_modules.py` `SupervisedSFT` — per-sample weight in loss
- `cfgs/task/user_behavior.yaml` — `mode: regular` (default) 추가
- `data_generation_old/user_behavior_build.py` — 직렬화 helper 제거; STM 소스를 abstracted log로 변경
- `svd_reinforce_hydra.py` — `mode=continual_phase2`일 때 sub-bucket 평가 분기 (~10줄)

### 변경 없음
- `scripts/train_user_behavior.sh` (regular 모드 그대로)
- 모든 base-model class / cfgs

---

## 하이퍼파라미터·기본값 (CLI 조정 가능)

| Param | Default | 설명 |
|---|---|---|
| `min_occurrence` | 3 | 4주간 <3번 등장하는 패턴 drop |
| `min_confidence` | 0.30 | `P(gold|trigger,context) ≥ 0.30` |
| `delta_t_min` | 10 | 다음 frequent event가 10분 이내 발생해야 함 |
| `max_stm_tokens` | 6000 | 샘플당 STM 토큰 예산 |
| `stm_window_days` | 7 | STM context look-back 윈도우 |
| `week_split_phase` | 17 | weeks <17 = phase1, weeks ≥17 = phase2 |
| `shifted_ctx_dims_min` | 2 | shifted 동일 context 판정 최소 일치 dim 수 |
| `negative_weight` | 0.5 | `polarity=-1` 샘플 multiplier |
| `decaying_weight` | 0.3 | obsolete 행동 down-weight (학습 시) |
| Random seed | 42 | reproducibility |

---

## Metric·loss 요약

**Metric (eval call별)**:
- 기존: `sem_acc`, `sem_sim_mean`
- 신규: `sem_acc/by_category` (always/emergent/decaying/shifted), `sem_acc/weighted`
- phase 2: `phase2_test/{always,shifted_new,emergent,decaying}` + `Δphase1_acc/<category>`

**Loss (SFT)**:
- Per-sample `-log p(gold | prompt)` × `sample.weight`, batch 내 weight 합으로 정규화.

**Reward (RL, contrastive)**:
- `reward = polarity · (2·sim − 1)`, 다음 `pg = -log_likelihood · reward · sample.weight`.

---

## 검증

1. **Unit smoke**: `python 4_extract.py` 실행 → `data/user_behavior_samples.json` 생성. 카테고리 카운트 출력:
   ```
   patterns: always=N1, emergent=N2, decaying=N3, shifted=N4
   train: <count>, valid: <count>, test: <count>
   continual: phase1_train=<>, phase1_test=<>, phase2_train=<>,
             phase2_test={always:<>, shifted_new:<>, emergent:<>, decaying:<>}
   ```
2. **분포 sanity**: top-20 gold answer + 그들의 pattern_category 출력. Top-1 비중이 이전 9.4%보다 훨씬 낮아야 함.
3. **Continual sanity**: emergent와 shifted 패턴 카운트가 0보다 커야 함 (0이면 임계값 낮춤 검토).
4. **Regular 학습 smoke**: `bash scripts/train_user_behavior.sh num_iters=2 batch_size=2`. 로그에 `sem_acc/by_category` 키 확인.
5. **Continual 학습 smoke**: `bash scripts/train_user_behavior_continual.sh` 실행 — phase1 후 ckpt A 저장, phase2가 load_ckpt=A로 resume. phase2 로그에 `phase2_test/always`, `phase2_test/decaying` 등 sub-bucket과 `Δphase1_acc` 등 forgetting metric 확인.
6. **기존 task regression**: gsm8k 1-iter 실행 — `get_sample_weight`/`get_sample_polarity` 기본값으로 인한 breakage 없음 확인.

---

## 열린 하이퍼파라미터 / 정책 질문

- `min_occurrence` / `min_confidence` 튠: 첫 실행 후 분포 보고 조정.
- `BACKGROUND_PREFIXES` 리스트: 4_extract.py 상단에 두고 편집 가능.
- `delta_t_min` (10분) — 출퇴근 같은 30+분 routine 위해 `delta_t_min_long=30` 별도 추가 가능 (v1엔 미포함).
- `gold_reasoning` 사용 여부 — `use_reasoning_in_target=False` (default off, SFT는 gold만 학습); True로 키면 CoT-style 학습.
