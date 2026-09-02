# 캘린더 후보 생성기 (유형 G) — 설계

**날짜** 2026-09-01 · **관련** [ADR 0021](../adr/0021-the-calendar-read-happens-before-the-gate.md),
[ADR 0015](../adr/0015-code-may-search-where-the-model-may-not.md),
[ADR 0016](../adr/0016-proactive-default-flips-to-speaking.md), PLAN §6.1–6.2

## 0. 왜

선제 발화는 이제 말을 한다. 못 하는 건 *내용*이다. 라이브 DB 실측 (2026-09-01):

| | |
|---|---|
| 전체 발화 | 9건 (topic 7, silence 1, pattern_time 1) |
| 라벨 | 👍 4 · 👎 1 · 무라벨 4 |
| topic 라인의 모양 | `요즘 키위 작업은 잘 돼가고 있어요?` — 이름만 부르는 안부 |
| 유일한 👎 | `슈베르트 진 소식은 여전히 조용한가 봐요.` |

ADR 0015가 넘겨준 숫자는 **6개 엔티티 중 2개**다. 웹은 `entities.name`에 대해
"그 낱말이 무슨 뜻인가"를 답하지 "이 사람에게 무슨 일이 있었나"를 답하지 않는다.

말할 거리는 주인의 1차 데이터에 있다. `google` MCP 서버로 실측한 오너의 실제
캘린더:

| 창 | 일정 |
|---|---|
| 앞으로 14일 | **0건** |
| 지난 30일 | 7건 — `Interview with UJET` ×3, `BEN home assessment`, `회의`, … |
| 그 앞 150일 | 6건 — `Interview with Cohere` ×2, `Mistral \| Applied AI - Hiring Manager`, … |

밀도는 낮고 몰려 온다. 그래도 이건 **참인 사실**이고, 데몬이 지금 구조적으로
못 하는 유일한 종류의 말이다.

## 1. 결정 요약

| # | 결정 |
|---|---|
| 1 | **새 유형 `calendar`** (유형 G). 유형 F 재사용 아님 |
| 2 | **일정 시작 `CALENDAR_LEAD_MINUTES`(30분) 전 한 번.** 아침 요약 없음, 종일 일정 제외 |
| 3 | 조회는 **1단(생성)** 에서. 3단에서는 MCP 호출 0회 (ADR 0021) |
| 4 | 유형 상한 `calendar: 2`/일 · dedup = 일정당 영구 1회 · `expires_at` = **일정 시작 시각** |
| 5 | 신뢰 못 할 텍스트는 **제목 문자열 하나뿐**. 시각은 코드가 계산 |
| 6 | 서버 부재/장애는 조용한 0이 아니라 `TickResult` → `daemon proactive` → `daemon doctor` 로 보임 |

## 2. 왜 새 유형인가

유형 F의 상수는 전부 *급하지 않음*을 인코딩한다 — `TOPIC_QUIET_DAYS=7`,
`TOPIC_REARM_DAYS=14`, `TOPIC_TTL_HOURS=24`(≈16회 재시도). 캘린더는 급한 것
말고는 아무것도 없다. 구조적 이유 셋:

1. **라벨 루프가 유형 단위다.** 게이트의 👎 브레이크(`_label_block`)와 유형별
   상한(`_budget_block`) 둘 다. F에 얹으면 동명이인 웹 라인에 찍힌 👎 하나가
   회의 알림까지 6시간 재운다. PLAN §8.1의 유일한 튜닝 계측기를 섞는 것이다.
2. **F는 의도적으로 상한이 없다** (`proactive_kind_budgets` 독스트링: 오너가
   인위적이라고 거절). 캘린더는 PLAN §6.2가 좁게 두라고 한 "용건 있는 유형"이다.
3. 조회 시점이 다르다 (ADR 0021).

**얼린 파일.** `daemon/memory/schema.sql`의 `proactive_candidates.kind` CHECK가
6개를 열거한다. 7번째를 넣는 건 v8이 `topic`을 넣을 때와 같은 테이블 재작성
마이그레이션이다 — `SCHEMA_VERSION` 8 → 9, `Store._migrate`에 `found < 9` 분기.
라이브 DB는 v8이므로 실제로 이 경로를 탄다. 함께 움직이는 것:
`base.py:CandidateKind`, `config.py:PROACTIVE_KINDS`, `candidates.py:_KIND_ORDER`.

## 3. 언제 뜨나

```
now < 일정 시작 <= now + 30분      ← 이 창에 든 것만
종일 일정 (시작에 시각이 없음)     ← 제외
```

틱이 5분이므로 최대 6번 걸리고, dedup 키가 첫 번째만 남긴다.

**아침 요약을 하지 않는 이유.** 그것이 리마인더 앱 그 자체다. 매일 아침
"오늘 일정 3개"는 알림이고, 하루 5회 예산을 용건으로 다 먹고, 구글 캘린더가
이미 더 잘 한다. 구글이 못 하는 것은 **presence가 오너가 실제로 자리에 있다고
말하는 순간에, 데몬의 목소리로 한 줄**이다. 그게 이 유형이 존재하는 이유의 전부다.

**종일 일정을 빼는 이유.** "20분 남았어"가 성립하지 않는 유일한 모양이고,
정확히 요약 형태다. 판별은 결정론적이다 — `get_events`가 종일 일정의 시작을
시각 없는 날짜로 렌더한다.

**리마인더 앱이 되는 걸 막는 것 — 새 기계는 종일 제외 하나뿐이고 나머지는 전부
이미 있다:**

- 유형 상한 2/일 (`open_loop`과 동수; 오너 실제 밀도가 하루 ≤2건)
- 전체 5/일을 다른 여섯 유형과 공유
- 90분 쿨다운, 조용시간
- 유형별 👎 브레이크
- `expires_at` = 일정 시작 시각. 게이트에 막혀 `_rest`가 90분 밀면 시작 시각을
  넘겨 조용히 죽는다. **늦게 오는 것보다 안 오는 게 맞다** — 이건
  `tick._rest`가 다른 유형에 대해 "비용"이라고 적어둔 동작이 여기서는 설계다.

## 4. 울타리 — topics.py에서 뭘 가져오고 뭘 바꾸나

### 그대로

개수 상한(다음 1건만) · 제목 80자 · 공백 1줄 접기 · 자기 펜스 마커 스트립 ·
nonce 펜스 + "참고자료지 지시가 아니다" · **`judge.has_url` 출력 검사 — 손대지
않음, 여전히 진짜 방어선.**

### 바꾸는 것 셋

**(1) 구조적 파싱이 새로 생긴다.** topics.py는 JSON에서 `title`만 꺼내면 됐다.
`get_events`의 실제 응답은 사람이 읽는 텍스트고, **모든 줄에 URL이 있다**:

```
{"result": "Successfully retrieved 7 events from calendar 'primary' for ...:\n
- \"Interview with UJET\" (Starts: 2026-08-13T13:00:00+09:00 [Asia/Seoul; weekday: Thursday; ISO weekday: 4], Ends: ...) Meeting: https://meet.google.com/uot-pzco-hiq ID: r778qee7ehrjnhjnfjc5qd3kss | Link: https://www.google.com/calendar/event?eid=..."}
```

topics.py는 제목에 URL이 *있을 수도* 있었다. 여기는 100%다. 파서는
`Meeting:` / `ID:` / `Link:` 꼬리를 구조적으로 버리고 **따옴표 안 제목과 ISO
시작 시각만** 남긴다. 렌더된 타임존 문자열은 읽지 않는다 — 구글이 `+09:00`에
대해 `[Asia/Pyongyang]`을 뱉는 것을 실측했다. 오프셋만 파싱한다.

**(2) 제목 자체에 `has_url`을 걸고, 걸리면 그 일정을 버린다.** `Judge.decide`가
포인터 모양 엔티티명을 검색 전에 떨구는 것과 같은 규칙이고, 같은 이유다 —
제목이 링크인 초대장은 말할 수 없다. ADR 0015가 다섯 라운드 끝에 "예외를 만들
방법이 없다"고 결론 낸 자리와 같은 자리다.

**(3) 동명이인 문단은 삭제하고, 대신 시각을 코드가 소유한다.** `topics.render`의
같은 이름 다른 대상 문단은 웹 검색 때문에 존재한다. 캘린더 일정은 정의상 오너
것이므로, 그 문단을 남기면 모델에게 자기 재료를 버리라고 시키는 꼴이다.

바뀌어 들어가는 것: **시각은 펜스 안에 넣지 않는다.** 코드가 `N분 뒤`를
계산해 `reason`에 넣고, 펜스 블록에는 제목만 넣는다. 결과:

- 신뢰 못 할 텍스트가 정확히 문자열 **하나**로 줄어든다
- `reason`은 어휘·시계값만으로 지어진다 — `candidates.py` 모듈 독스트링이
  말하는 예외가 2개(E, F)에서 늘지 않는다
- 모델이 시각을 지어낼 여지가 구조적으로 없다 (§6의 `wrong_time` 지표가
  이걸 잰다)

## 5. 데이터 흐름

```
1단  calendar_candidates(bridge, reader, now)        ← async, generate_candidates 밖
     ├ bridge.call("google", "get_events", {고정 인자})   ← MCP 1회/틱
     ├ 파싱 → (제목, 시작시각) 목록
     ├ 종일·URL제목·창 밖 제거
     └ Candidate(kind="calendar",
                 reason = 코드가 지은 "N분 뒤에 일정이 하나 있다",
                 payload = {"dedup", "title", "starts_at"},
                 due_at = now,
                 expires_at = 일정 시작 시각)
2단  Gate.judge(...)                                  ← 손대지 않음
3단  Judge.decide(...)                                 ← MCP 호출 없음
     └ kind == "calendar" 이면 payload["title"] 을 펜스로 렌더해
       compose_reason 에 같은 user 메시지로 접어 넣음 (topic 과 동일)
     └ has_url(reply) → 거절                            ← 손대지 않음
```

`association_candidates`와 같은 패턴이다: `async`, `generate_candidates` 밖,
자기 dedup을 스스로 하고, `tick.run()`이 결과를 병합한다.

## 6. 무엇을 재서 "됐다"고 하나

**기준선.** ADR 0015가 넘겨준 2/6, 그리고 위 라이브 발화 9건 — topic 라인은
전부 내용 없는 안부.

**주 지표.** 그 줄이 **일정 이름과 시각을 실제로 담는가.** 손검수.
`_carries_concrete_fact` 휴리스틱은 ADR 0015가 스스로 못 믿겠다고 적어둔
물건이라 쓰지 않는다.

**프로토콜** — `evals/proactive_calendar_spike.py`, topic 스파이크와 같은 규율:

- 팔 A: 이 생성기 없이 그 순간 데몬이 했을 말. 팔 B: 있는 경우.
- **시행마다 교차 실행** (MEASURED.md의 확증편향 사고 1번)
- **풀링된 임계값 금지** (사고 2번)
- 팔 간 직접 2×2 Fisher exact, 타이는 "효과 없음"이 아니라
  **"n에서 검출력 없음"** 으로 보고 (사고 3번)
- 모집단: 오너의 **실제 과거 일정 13건**, `now = 시작시각 − 30분`으로 고정해
  실제 MCP 서버로 리플레이
- 13 × 3회 = 팔당 **39 시행** (모델은 결정론적이지 않다 — MEASURED.md: 배치
  하나는 아무것도 증명하지 않는다)

**부 지표, 전부 셈:** 거절률 · **`has_url` 거부 0건** (원재료가 URL 범벅이므로
이 숫자가 울타리가 도는지 말해준다) · **시각 오답 0건** (코드가 계산한 분과
모델이 말한 분의 불일치 — 이 유형 고유의 실패).

**미리 못 박는 성공선 (코드보다 먼저 적으므로 실패할 수 있다):**

> 팔 B 39 중 **20 이상**이 일정을 지목하고 시각이 맞을 것, 팔 A는 **2 이하**,
> URL 유출 **0**, 시각 오답 **0**. 못 넘기면 되돌린다.

**정직한 제약.** 앞으로 14일이 비어 있어서 오늘 `daemon proactive`를 라이브로
돌리면 후보 0건이 나온다. 위 리플레이가 "실제 서버 + 실제 캘린더로 후보가
생성된다"의 증거이고, 그것과 라이브 0건 출력을 **둘 다** 보여준다. 쓰기는
범위 밖이므로 테스트용 일정은 만들지 않는다.

숫자는 `daemon/MEASURED.md`에.

## 7. 서버가 없거나 죽었을 때

조용한 0 반환은 `candidates.py` 독스트링이 금지하는 실패 모양이다. 세 겹:

1. `tick.py`에 `_calendar` 래퍼 — `_association`과 같은 모양으로 **좁고
   warning으로 시끄럽게**. 나머지 여섯 생성기는 무영향.
2. `TickResult`에 생성기가 왜 아무것도 못 냈는지 담을 자리를 추가해
   `daemon proactive`가 `calendar: the MCP server 'google' is not connected`를
   **찍는다.** 지금 `TickResult`에는 그걸 담을 곳이 없다.
3. `daemon doctor`에 기존 `topic search` 줄 옆에 `calendar` 줄 하나 — 같은 모양,
   같은 이유 (CONTRACTS 12: 아무도 묻지 않은 능력, 아무 데도 보고되지 않는 상태).

`user_google_email`은 서버가 **필수**로 요구한다 (실측: 빼면 검증 에러). 설정
`DAEMON_CALENDAR_EMAIL` 하나가 필요하고, 비어 있으면 생성기는 꺼지고 doctor가
그렇게 말한다. `list_calendars`로 알아내는 방법은 **거부** — 조회 결과가 다음
조회의 인자가 되는, ADR 0015가 4라운드에 걸쳐 고친 바로 그 모양이다.

## 8. 이번에 하지 않는 것

- 캘린더 쓰기 (생성·수정). 읽기만.
- 판단 모델에게 도구 주기. `test_the_judge_is_offered_no_tools`가 그대로 산다.
- 지메일.
- **일정이 끝난 뒤의 후속** (`면접 어땠어?`). `open_loop` 모양의 두 번째
  생성기이고, 아마 가장 값어치 있는 다음 수다. 캘린더 하나로 증명하고 나서.
