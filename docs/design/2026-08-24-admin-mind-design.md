# 설계 — 어드민 Memory · Persona 탭

작성 2026-08-24. `docs/design/2026-08-07-m5-admin-web-design.md` §6이 명시적
후속으로 미뤄둔 **"배운 것" 열람 UI**를 닫는다. 산문보다 결정과 시그니처가
중요하다.

근거: PLAN §4.4(큐레이션 티어)·§5.1(앵커·변화율)·§8.1(로그 클럭). 규칙:
CONTRACTS 5(`seed.md` 인간 소유, 코드는 절대 쓰지 않는다)·6(`observations`
append-only)·10(오리진 게이트)·12(보이지 않는 능력은 없다).

## 무엇이고 무엇이 아닌가

콘솔의 나머지 탭은 전부 **"그녀가 뭘 했나"** 를 보여준다. 이 두 탭은 **"그녀가
뭘 알게 됐고, 나를 어떻게 다루기로 배웠나"** 를 보여준다. 실측으로 지금
어드민에서 접근 가능한 것은 Activity 탭의 `reflection` 칩 한 줄 —
"reflection 2026-08-19 - 1 fact, 2 entity notes, 2 observations" — 즉 **횟수만
보이고 내용은 안 보인다.** 사실 12개, 관찰 9개, 학습 규칙 3개, 성찰 원문 9일치는
`daemon persona` / 파일 직접 열기로만 볼 수 있다.

읽기 전용 거울이 아니다. 손잡이 셋을 단다(아래 결정 2). 편집기는 아니다 —
규칙 추가·수정은 없고, 사실 폐기도 없다.

## 결정 (뒤집지 말 것)

1. **탭 두 개**: `Memory`(사실·엔티티·성찰) + `Persona`(관찰·규칙·다이어리·앵커).
   나누는 선이 코드베이스의 선과 같다 — `daemon/reflection.py`+`memory/` 대
   `daemon/persona/`, PLAN의 M2 대 M4. 한 탭 `Mind`로 합치면 이 프로젝트의 핵심인
   후자가 사실 12개 밑에 묻힌다.

2. **손잡이 3개만**: `Forget`(규칙 은퇴) · `Reflect now` · `Evolve now`. 셋 다
   이미 있고 이미 검증된 함수를 부르는 것뿐 — `LearnedRules.retire`,
   `daemon reflect`, `daemon persona evolve`. **사실 폐기는 범위 밖**:
   `Store`에 단독 은퇴 API가 없어 memory 계약에 새 쓰기 경로를 뚫어야 하고,
   CLI에도 없어서 콘솔이 유일한 문이 된다. 실제로 잘못된 사실을 겪은 뒤에 한다.

3. **마크다운 본문은 페이로드에 인라인**. 실측 총량 11.6KB(엔티티 1.7 + 성찰 7.7
   + 다이어리 2.2). `/api/memory/reflection/{date}` 류를 만들지 않으므로 **경로
   파라미터가 파일시스템에 닿는 일이 없다** — traversal 버그가 생길 수 없다.
   상한은 **본문에만** 걸린다 — 목록은 언제나 전부 준다(결정 5와 같은 이유:
   빠진 줄은 없었던 일이 된다). 본문을 인라인하는 범위는 성찰 최근 14일,
   다이어리 최근 8개, 엔티티 `mention_count` 상위 60개, 그리고 세 종류 합계
   64KB. 넘어가는 항목은 목록에는 남고 본문만 없으며, 그 섹션이
   `bodies_truncated: true`를 달아 그렇다고 말한다(규칙 12). 본문 없는 항목은
   화면에서 `▸` 대신 파일 경로를 보여준다.

4. **이 두 탭은 15초 폴링에 넣지 않는다.** `index.html:1432`의
   `setInterval(refresh,15000)`은 health/today/activity/mcp를 돈다. 하루 한 번
   바뀌는 화면을 15초마다 11KB씩 받을 이유가 없다. 탭 진입 시 + 수동 새로고침 +
   쓰기 직후에만 로드.

5. **성찰 목록의 축은 `reflection_runs`가 아니라 `memory/reflections/`의 날짜.**
   실측: 테이블 5행, 아티팩트 9개 — 테이블이 M5에 생겨서 그렇다. 테이블을 축으로
   잡으면 08-06~08-10 나흘이 화면에서 사라진다. 파일이 날짜의 기록이고
   `reflection_runs`는 있으면 붙이는 부가정보(없으면 `artifact only`).

6. **마크다운→HTML 변환 없음.** 모든 텍스트는 기존 `esc()`(`index.html:537`)를
   통과해 `<pre>`로. 성찰 본문은 모델이 쓴 글이고 그 안에 웹·메일에서 읽은
   텍스트가 섞여 있을 수 있다.

## 발견: `catchup_lock`이 `app.state`에 없다

`daemon/app.py:238`의 `catchup_lock`은 **`_start` 안의 지역 변수**다. 잡 인자로만
넘어가고(`app.py:248`, `:264`) `app.state`에는 없다. 그 락이 막는 것은 바로 위
주석(`app.py:232-237`)에 적혀 있다 — 한 날짜에 `run()`이 두 번 돌면 append-only
성찰 아티팩트를 이중 기록하고 관찰을 중복 삽입해 **M4 로그 클럭이 깨진다**.

따라서 `Reflect now` / `Evolve now`는 반드시 같은 락을 잡아야 하고, 지금은 잡을
방법이 없다.

```python
# daemon/app.py _start(), catchup_lock 생성 직후
app.state.catchup_lock = catchup_lock   # mcp_persist_lock(app.py:157)과 같은 방식
```

이 한 줄이 없으면 04:00 크론이 도는 중에 버튼을 누르면 데이터가 깨진다.

`_reflect_tick` / `_persona_tick`은 전부 삼켜서 `None`을 반환하므로 브라우저에
결과를 줄 수 없다. 각각을 **결과를 반환하는 헬퍼**로 쪼개고, 기존 크론 래퍼는 그
헬퍼의 결과를 로그로 찍는다. 라우트에 로직을 복사하지 않는다.

```python
async def run_reflection_now(settings, lock) -> list[reflection.Result]
async def run_persona_evolution_now(settings, lock, *, force: bool) -> EvolutionResult
```

## 파일

```
daemon/admin/mind.py            신규 — 페이로드 두 개. activity.py와 같은 형태:
                                순수 함수, 읽기 전용, 모델 호출 없음
daemon/admin/routes.py          엔드포인트 5개 추가
daemon/admin/static/index.html  nav 2개 + section 2개 + 렌더 함수
daemon/app.py                   app.state.catchup_lock 1줄 + 헬퍼 2개 분리
daemon/memory/store.py          읽기 메서드 3개 추가
tests/test_admin_mind.py        신규
```

## 시그니처

```python
# daemon/admin/mind.py
MAX_REFLECTIONS = 14
MAX_DIARIES = 8
MAX_ENTITIES = 60
MAX_BODY_BYTES = 64 * 1024

def memory_payload(store: Store, data_dir: Path) -> dict[str, Any]
def persona_payload(store: Store, data_dir: Path, settings: Settings) -> dict[str, Any]
```

```python
# daemon/memory/store.py — 기존 active_* 는 활성만 주므로 세 개가 더 필요하다
def recent_entries(self, limit: int = 100) -> list[sqlite3.Row]        # 폐기 포함
def recent_observations(self, limit: int = 200) -> list[sqlite3.Row]   # consumed_by 포함
def retired_persona_rules(self, limit: int = 50) -> list[sqlite3.Row]  # retired_at/why
```

```
GET  /admin/api/memory
GET  /admin/api/persona
POST /admin/api/persona/forget    {id: int, why: str}
POST /admin/api/reflect           {date?: str, force?: bool}
POST /admin/api/persona/evolve    {force?: bool}
```

## 화면

### Memory

```
FACTS                                   11 active · 1 retired
 [9] 사용자는 개발자(AI/LLM 비서 벨라의 창조주/개발자)이다.
     user_job · 개발자, 직업, 일 · 8/19
 ▸ retired (1)

ENTITIES                                                  11
 김대현 person 5 · UJET.cx company 3 · llm-wiki project 2 …
 (클릭 → 노트 본문 + [[링크]])

REFLECTION                      9 days · 5 recorded   [Reflect now]
 2026-08-19  written  72 msg → 1 fact, 2 entities, 2 obs   ▸
 2026-08-14  artifact only                                 ▸
```

정렬: 사실은 `importance` 내림차순(동률은 `updated_at` 최신), 엔티티는
`mention_count` 내림차순, 성찰은 날짜 내림차순.

### Persona

```
ANCHOR
 active 3/20 · max +3/week · min obs 5 · last evolve 8/24 · unconsumed 0
 ▸ seed.md   26 lines — yours, code never writes this
 ▸ learned.md 3 lines — hers

LEARNED RULES                                         3 active
 [1] 시스템 오류나 문제 발생 시 변명 없이 …        [Forget]
     8/09 · 3 observations  ▸ (evidence id → 관찰 문장)
 ▸ retired (0)

OBSERVATIONS                          9 · consumed 9 / pending 0
 [0.85] 사용자는 테스트 도중 파일/도구 실행에 …  → rule 1 · 8/06

EVOLUTION                                  2 diaries  [Evolve now]
 2026-08-24  +2 rules  ▸
```

앵커 숫자의 출처: `settings.persona_max_active_rules`(20),
`persona_max_new_per_cycle`(3), `persona_min_observations`(5),
`store.last_persona_rule_created_at()`, `store.count_active_persona_rules()`,
미소비 = `len(store.unconsumed_observations())`.

**앵커가 계기판 먼저이고 원문이 접힌 이유**: 앵커는 "seed를 안 건드린다"는 사실
하나가 아니라 **변화가 얼마나 느린지**가 요점이고(PLAN §5.1 성격 붕괴 방지),
그건 숫자여야 보인다. `seed.md` 원문은 사람 소유라는 주장이 실제로 그 파일에
근거가 있어야 설득력이 있으므로 빼지 않고 접는다.

**은퇴 이력을 남기는 이유**: `learned.md`는 통째로 재작성되는 파일이다. 무엇이
사라졌는지 보이지 않으면 규칙이 조용히 없어져도 알 수 없다. 지금 은퇴 규칙은
0건이라 대부분 비어 있겠지만 비어 있음 자체가 정보다.

## Forget 흐름

클릭 → `why` 입력(필수, 빈 문자열 거부) → `POST /admin/api/persona/forget` →
`LearnedRules(data_dir, store).retire(id, why=why)`.

`LearnedFileDiverged`는 **일반 실패로 뭉개지 않고 실제 이유를 띄운다** —
"learned.md에 손으로 추가된 줄 N개가 있어 거부: …". 그게 이 예외의 요점이고
(`cli.py:1237-1239` 주석), 안 보여주면 사용자는 버튼이 고장난 줄 안다. 404는
`retire`가 `False`를 줄 때(그 id의 활성 규칙 없음).

## 안전·계약

- **CONTRACTS 5**: `seed.md`는 읽기만. 이 설계에 `seed.md`를 쓰는 경로는 없다.
- **CONTRACTS 6**: `observations` append-only. 이 탭은 관찰을 읽기만 한다.
- **CONTRACTS 10**: 두 탭 다 `_loopback_only`(`routes.py:112`) 아래. 새 가드 없음.
- **개인정보**: 사실 id 3이 집 주소다(importance 8). 이 화면은 소유자의 집 주소를
  브라우저에 렌더한다. 루프백 가드가 **유일한** 방어선이고, 인증이 없는 것은 M5
  설계 결정 1이다. 원격 접근이 후속으로 열릴 때 이 탭이 그 결정을 다시 봐야 하는
  첫 번째 화면이다.
- **모델 호출**: `Reflect now` / `Evolve now`만. 읽기 엔드포인트 둘은 모델을
  부르지 않는다(`activity.py`의 헤더가 같은 이유로 같은 말을 한다).

## 테스트

`tests/test_admin_mind.py`:

- 성찰 목록이 파일 축인지 — `reflection_runs`에 없는 날짜가 `artifact only`로 뜬다
- 폐기된 사실·은퇴된 규칙이 활성과 분리된다
- 상한 초과 시 목록은 남고 본문만 생략되며 그 섹션에 `bodies_truncated: true`가 붙는다
- `forget`의 divergence 거부가 이유를 담아 4xx로 나온다
- `forget`에 빈 `why`는 거부
- `Reflect now` / `Evolve now`가 `app.state.catchup_lock`을 잡는다 — 락이 잡혀
  있으면 두 번째 호출이 기다린다
- `seed.md`를 쓰는 경로가 라우터에 없다

**그리고 실 데이터로 실제 브라우저에서 확인한다.** 기존 `tests/test_admin*.py`
3개가 그린인 것은 증거가 아니다(메모리 `verify-by-running-real`,
`qa-drive-the-live-ux`). 확인 항목: 두 탭이 열리고, 한글 본문이 DM Mono 폴백으로
깨지지 않고, `Forget`이 실제로 `learned.md`에서 줄을 지우고, `Reflect now`가
결과 숫자를 돌려준다.

## 범위 밖 (명시적 후속)

- 사실 폐기(결정 2)
- 규칙 추가·편집 — `learned.md`가 AI 소유라는 것이 앵커다
- 엔티티 그래프 시각화 — `entity_links`는 있지만 11개 노드에 그래프는 과하다
- 회상(recall) 품질 열람 — `evals/golden_set`의 일이고 어드민의 일이 아니다
