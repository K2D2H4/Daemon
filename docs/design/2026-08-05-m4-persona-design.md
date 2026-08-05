# M4 설계 — 페르소나 진화

작성 2026-08-05. 구현 계약이다. 병렬로 일하는 에이전트가 서로 다른 것을 짓지
않게 하는 것이 이 파일의 목적이고, 산문보다 시그니처가 중요하다.

근거: PLAN §5(전체), §8.2 M4 행, §8.3. 규칙: CONTRACTS 비협상 1·3·5·6.

## 왜 지금 짓고, 게이트는 왜 열린 채로 남는가

M4의 게이트는 "2주치 실제 관찰로 성격 변화가 체감된다"다. 실측(2026-08-05):
`observations` 0행, `persona_rules` 0행, 상주 프로세스 없음(LaunchAgent 미설치).
**게이트를 통과할 입력이 존재하지 않는다.** M3가 같은 이유로 코드를 먼저 지었고,
M4도 같다 — 코드가 끝나는 날은 "M4 완료"가 아니다.

관찰이 0인 원인은 별도로 실측됐고 이 문서의 범위가 아니다: `messages_for_day`가
`recalled = 1` 행을 영구히 제외하고(CONTRACTS 위생규칙 2), 실제 하루에서 38건 중
29건이 그렇게 빠졌다. 빠진 쪽이 정확히 페르소나 관찰("짧게 대답해줄래",
"답장이 왜케 오래걸려")이고 남은 쪽은 웨이크워드 노이즈였다. **한 줄짜리 후속
항목으로 분리했다** — 이 설계는 관찰이 있다고 가정하고, 검증은 가짜 관찰로 한다.

## 파일

```
daemon/persona/loader.py   seed.md + learned.md 조립. 매 턴 읽는다.
daemon/persona/rules.py    learned.md 와 persona_rules 미러. 쓰기의 유일한 경로.
daemon/persona/evolve.py   주 1회 패스: 관찰 -> 규칙. 모델 호출 1회.
daemon/memory/store.py     persona_rules 접근 메서드 추가 (frozen 아님)
```

## loader.py

```python
SEED_FILE = Path("persona") / "seed.md"
LEARNED_FILE = Path("persona") / "learned.md"

def seed_path(data_dir: Path) -> Path
def learned_path(data_dir: Path) -> Path
async def load_persona(data_dir: Path) -> str
```

- seed 먼저, learned 다음. **매 턴 읽는다** — `seed.md`는 사람이 소유하고, 편집이
  재시작 없이 먹어야 한다(현재 `loop.py._read_seed`의 동작을 유지).
- 없으면 그 부분만 빈 문자열. `OSError`는 로깅하고 빈 문자열. 예외를 올리지 않는다 —
  대화 한 턴이 페르소나 파일 권한 때문에 죽어서는 안 된다.
- `seed`는 무가공으로 넣는다. `learned` 블록에는 한 줄 헤더를 붙여 모델이 "이건
  내가 이 사람에 대해 배운 것"임을 구분할 수 있게 한다. 문구는 구현자가 정한다.
- 둘 다 비면 빈 문자열을 반환하고, 호출자는 system 메시지를 넣지 않는다.

## rules.py

```python
MAX_BODY_CHARS = 200
HEADER: str          # 파일 상단. AI 소유임과 사람이 할 수 있는 일(읽기·삭제요청)을 적는다.

@dataclass(frozen=True)
class Proposal:
    body: str
    evidence: tuple[int, ...]          # observation ids
    supersession_key: str | None = None

def render(bodies: list[str]) -> str   # 파일 전체. 한 규칙 = 한 불릿, 개행은 접는다.

class LearnedRules:
    def __init__(self, data_dir: Path, store: Store) -> None
    async def add(self, proposals: list[Proposal], *, now: datetime | None = None) -> list[int]
    def active(self) -> list[sqlite3.Row]
    async def retire(self, rule_id: int, *, why: str, now: datetime | None = None) -> bool
```

- `add()` 순서는 **마크다운 → 미러 → 관찰 소비**다. 반대로 하면 CONTRACTS 1 위반이고,
  전원이 끊기면 존재하지 않는 규칙의 행이 남는다. 마크다운은
  `fs.write_private_replace`(원자적·fsync)로 통째로 재작성한다 — `curated.py`와 같다.
- 파일에는 **body만** 담는다. `created_at`·`evidence`·`supersession_key`·`status`는
  컬럼이다(CONTRACTS 3). 모델이 산문으로 쓸 수 있으면 위조하거나 뭉갠다.
- `supersession_key`가 있으면 같은 키의 활성 규칙을 은퇴시킨다. **배치 안에서 먼저
  해소한다** — PLAN §8.2.1이 기록한 결함 2(같은 키 2개가 순서대로 적용돼 결과가
  조용히 뒤집힘)를 반복하지 않기 위해서다. 남기는 쪽은 배치의 마지막 제안이 아니라
  결정론적 기준(먼저 온 것)이고, 버린 쪽은 결과에 보고한다.
- `retire()`는 사람의 삭제 요청이다. `status='retired'`, `retired_at`, `retired_why`를
  쓰고 마크다운을 재작성한다. **관찰의 `consumed_by`는 건드리지 않는다** —
  append-only(CONTRACTS 6)이기도 하고, 되돌리면 다음 주에 같은 근거로 같은 규칙이
  부활한다. 삭제 요청이 존중되는 방식이 이것이다.
- 존재하지 않거나 이미 은퇴한 id면 `False`. 예외를 올리지 않는다.

## store.py 추가

```python
def insert_persona_rule(self, *, body, created_at, evidence, supersession_key=None) -> int
def active_persona_rules(self) -> list[sqlite3.Row]
def retire_persona_rule(self, rule_id, *, when, why) -> bool
def count_active_persona_rules(self) -> int
def persona_rules_created_since(self, ts: str) -> int
def consume_observations(self, ids: Sequence[int], rule_id: int) -> None
def last_persona_rule_created_at(self) -> str | None
```

`evidence`는 JSON 텍스트로 저장한다(스키마가 `json_valid` 체크를 걸어 뒀다).
`consume_observations`는 `consumed_by IS NULL`인 행만 갱신한다 — 한 번만 소비된다.

## evolve.py

```python
OBSERVATION_BUDGET = 60              # 프롬프트에 넣는 최대 관찰 수
DIARY_SUBDIR = Path("persona") / "diary"

@dataclass(frozen=True)
class EvolutionResult:
    date: str
    observations_read: int
    proposed: int
    added: int
    retired: int
    skipped: str                      # 빈 문자열이면 실제로 돌았다
    problems: tuple[str, ...]

class PersonaEvolution:
    def __init__(self, data_dir, store, gateway, *,
                 max_active: int = 20, max_new: int = 3, min_observations: int = 5) -> None
    async def run(self, *, now=None, force: bool = False) -> EvolutionResult
    def diary_path(self, date: str) -> Path
```

`run()`의 순서. **앞의 세 관문은 모델을 부르지 않는다.**

1. 이번 주 일지 파일이 있으면 `skipped="already run this week"`. `force=True`면 무시.
   일지 파일이 곧 멱등성 마커다 — `reflection.py`가 `memory/reflections/`로 하는 것과
   같은 방식이고, 이유도 같다(스키마가 frozen이고 "이 주는 진화했다"는 상태는
   계약의 마크다운 쪽에 속한다).
2. 미소비 관찰 수 < `min_observations` → `skipped="not enough observations (n<k)"`.
   **관찰 3개로 성격을 바꾸지 않는다.** 이것이 결정론적 게이트다.
3. 활성 규칙이 `max_active`면 → `skipped="rule budget full (n/n)"`.
4. 프롬프트: `seed.md`(앵커, 읽기 전용) + 현재 활성 규칙 + 미소비 관찰 최대
   `OBSERVATION_BUDGET`개. `gateway.complete(Task.PERSONA_RULE, ...)` **1회**.
5. 파싱은 좁게. JSON 실패는 `problems`에 남기고 빈 결과로 끝낸다(예외 아님).
   각 제안: `body` 한 줄로 접고 `MAX_BODY_CHARS`로 자름, 빈 것 버림,
   `supersession_key`는 `[a-z0-9_]` 40자로 좁힘(`reflection.py`와 동일한 이유 —
   모델이 쓴 문자열이 키가 되면 안 된다), `evidence`는 실제로 존재하는 미소비 관찰
   id만 남김.
6. 변화율: `max_new`개까지, `max_active` 상한까지. **잘린 것은 `problems`에 남긴다.**
   조용히 버리면 다음 주에 왜 안 늘었는지 아무도 모른다.
7. 기존 활성 규칙과 body가 같으면 버림(`problems`에 남김).
8. `LearnedRules.add()` → 일지 파일 쓰기.

**일지** `data/persona/diary/YYYY-MM-DD.md`: 이번 주 추가된 규칙, 각 규칙의 근거
관찰 원문, 은퇴된 규칙, 스킵 이유. PLAN §5.5의 "주간 diff"와 §8.3의 세 번째 층이
같은 산출물이고, 동시에 1번의 멱등성 마커다.

**앵커의 한계를 기록한다.** 코드가 `seed.md`를 쓰지 않으므로 규칙이 앵커를 지울 수는
없다(구조적 보장). 그러나 `learned.md`에 "항상 동의해라"라는 규칙이 들어가는 것은
프롬프트로만 막는다 — 변화율 상한과 사람의 열람·삭제가 그 뒤의 방어선이다.

## 배선

- `loop.py`: `_read_seed` → `load_persona(self._data_dir)`. `self._seed_path`는
  `self._data_dir`로. 214-215행의 "M4가 이 자리를 바꾼다" 주석을 실제 상태로 갱신.
- `app.py`:
  - `persona_seed()` 호출자를 확인해서 **대화 표면(음성 포함)은 learned를 받게** 한다.
    `judge.py`는 seed만 유지한다 — 프롬프트를 의도적으로 최소로 둔 자리이고,
    바꾸려면 별도 판단이 필요하다. 유지하는 이유를 주석으로 남긴다.
  - `build_persona_evolution(settings)` — `build_reflection`을 그대로 따른다
    (객체 + closer 튜플, 함수-로컬 임포트).
  - 주 1회 잡: `PERSONA_DAY = "mon"`, `PERSONA_HOUR = 5`(04:00 성찰 다음).
    `timezone=None`(로컬 시간), `max_instances=1`, `coalesce=True`, 틱 함수에
    최상위 `except` — 예외를 올리는 잡은 한 번 로깅되고 그 뒤로 스케줄이 영원히
    건강하게 보인다. 아무 일도 없었을 때도 INFO로 남긴다.
- `cli.py`:
  - `daemon persona` — 활성 규칙, 생성 시각, 근거 관찰 수, 마지막 일지.
  - `daemon persona evolve [--force]` — 4-8을 지금 손으로. 주 1회 잡을 아무도
    보고 있지 않을 때 검증할 유일한 방법이다.
  - `daemon persona forget <id> --why "..."` — 사람의 삭제 요청.
  - `doctor`: 활성 규칙 수, 미소비 관찰 수, 마지막 진화 날짜, 다음 실행이 요건
    미달이면 그 이유. 빈 것과 도는 것이 밖에서 같아 보이면 안 된다.
- `config.py`: `persona_max_active_rules`(20, `DAEMON_PERSONA_MAX_ACTIVE_RULES`),
  `persona_max_new_per_cycle`(3, `DAEMON_PERSONA_MAX_NEW_PER_CYCLE`),
  `persona_min_observations`(5, `DAEMON_PERSONA_MIN_OBSERVATIONS`). 프리셋 3개는
  이미 `PERSONA_RULE`을 라우팅한다 — 건드릴 것 없다. 스위치는 두지 않는다:
  실패 비용이 성찰과 같은 종류이고(AI 소유 파일 + 상한), 성찰에도 스위치가 없다.
- `.env.example`: 새 설정 3개.
- `tests/test_reachable.py`: `PENDING_TASKS`에서 `Task.PERSONA_RULE` 삭제,
  `WIRED_CLASSES`에 `PersonaEvolution`·`LearnedRules` 추가.

## 테스트

`db`·`fake_provider` 픽스처를 쓴다. 네트워크·실제 키 금지. 반드시 덮을 것:

- 관찰 부족 / 일지 존재 / 규칙 상한 → **모델 호출 0회** (fake provider의 호출 수로 단정)
- 주당 상한을 넘은 제안이 `problems`에 남는가
- 마크다운이 미러보다 먼저 쓰이는가 — 미러 insert가 실패해도 파일이 남고, 그 역은
  아닌 것
- `seed.md`에 절대 쓰지 않는가 (mtime/내용 불변 단정)
- `retire` 후 같은 관찰로 같은 규칙이 부활하지 않는가
- `load_persona`가 두 파일을 합치는가 / 한쪽만 있어도 도는가 / 없으면 빈 문자열인가
- 같은 `supersession_key` 2개가 한 배치에 오면 결정론적으로 해소되는가
- 인수 테스트: `learned.md`에 규칙이 있을 때 실제 대화 프롬프트에 그 규칙이 들어가는가

## 검증 (사람이 아니라 내가 돌린다)

가짜 관찰 30개 → `daemon persona evolve` → `learned.md`에 규칙 → `daemon persona`로
읽힘 → 실제 대화 한 턴에서 그 규칙이 프롬프트에 들어감 → `daemon doctor`가 상태를
보고함. 각 단계의 실제 출력을 붙인다.

## 범위 밖

- `recalled = 1` 관찰 굶주림 (위 참조) — 별도 항목
- 유형E 연상 생성기 (PLAN §9)
- `judge.py`가 learned 규칙을 받게 하는 변경
- LaunchAgent 설치 — 시계는 사용자가 켠다
