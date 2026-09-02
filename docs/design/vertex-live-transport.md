# Vertex 전송 경로 — 음색과 지연을 동시에 갖기

**상태:** 구현됨 (2026-09-02). 결정은 [ADR 0020](../adr/0020-two-endpoints-serve-gemini-live.md),
이 저장소 자신의 클라이언트로 확인한 것은 `evals/vertex_live_spike.py`.
**측정일:** 2026-09-02, 이 머신에서 서울 → us-central1.

## 왜

`gemini-live-2.5-flash-native-audio`는 **Vertex AI에만 존재한다.** API 키
엔드포인트(`generativelanguage.googleapis.com`)에서는 1008
`models/gemini-live-2.5-flash-native-audio is not found`로 끊기고, 그 엔드포인트의
ListModels에도 나타나지 않는다. 같은 2.5 네이티브 오디오 계열의 preview 빌드만
API 키로 닿을 수 있고, 그것이 3초대를 쓴다.

`say -v Yuna`로 만든 3.56초 한국어 발화를 1배속으로 스트리밍하고 마이크를 열어둔
채, **마지막 음성 샘플 전송 → 첫 오디오 바이트**를 같은 SDK로 측정 (5회, arm 교차):

| 경로 / 모델 | 중간값 | 범위 | 무응답 |
|---|---|---|---|
| Vertex `gemini-live-2.5-flash-native-audio` | **1430 ms** | 1414–1455 | 0/5 |
| API 키 `gemini-3.1-flash-live-preview` | 1723 ms | 1681–1754 | 0/5 |
| API 키 `gemini-2.5-flash-native-audio-preview-12-2025` | 3137 ms | 2577–3349 | 0/5 |

편차가 41 ms 대 772 ms다. 빠른 것보다 **일정한** 것이 이 경로의 진짜 이득이다.

레지던트가 실제로 보내는 구성(툴 98개 + affective dialog + LOW 감도)을 전부 켠
상태에서도 **1739 ms** (1678–1780, 무응답 0/4) — 현재 3.1 구성의 1729 ms와 사실상
동일한 지연에, 소유자가 고른 음색과 3.1에는 없는 표현력 기능이 함께 온다.

## 이 경로에서 확인된 것 (API 키 경로에서는 문제였던 것들)

| | API 키 2.5 | Vertex 2.5 |
|---|---|---|
| `START_SENSITIVITY=low` | **12/12 무응답** (입력 전사조차 빔) | 1420 ms, 0/3 |
| 툴 선언 98개 | +1380 ms (3.7 → 5.1초) | **+127 ms** (1441 → 1568) |
| `enable_affective_dialog` | 지원 (4269 ms) | 지원, +147 ms (1588) |
| 입력/출력 전사 | 지원 | **둘 다 지원** (확인함) |

입력 전사는 선택 사항이 아니다 — 그것이 없으면 메모리와 페르소나 진화가 굶는다
(`gemini_live.py` `_setup_message` 주석). Vertex에서 동작함을 실제 세션으로 확인했다.

## 지역

라이브 모델이 있는 지역: us-central1, us-east1, us-east4, us-west1, europe-west4.
**asia-northeast1 / asia-northeast3 / asia-southeast1에는 없다** — ReadyTalk이
us-central1을 하드코딩한 이유이고, 코드에 주석으로도 남아 있다
(`readytalk/backend/app/routers/voice_chat.py:45`).

서울에서 us-west1과 us-central1은 **차이가 없다** (1441 ms 대 1441 ms, 3회씩).
지연은 RTT가 아니라 서빙 쪽에서 나온다는 뜻이므로 us-central1을 기본값으로 둔다.

**Vertex에 2.5보다 새로운 대화형 라이브 모델은 없다.** 7개 지역을 확인했고 나온 것은
`gemini-live-2.5-flash-native-audio`와 `gemini-3.5-transcribe-live-preview`(전사 전용)
뿐이다. 즉 이것은 "전부 Vertex로 옮긴다"는 결정이 아니다 — 새 세대는 API 키 쪽에 있고,
빠른 서빙은 Vertex 쪽에 있다. 두 전송 경로가 공존해야 한다.

## 바뀌는 것은 세 군데

프로토콜 본문은 동일하다 (같은 proto-over-JSON). `_setup_message`, `_decode`,
툴 처리, 전사 누적, 재연결 백오프는 손대지 않는다. `daemon/voice/base.py`의
`VoiceSession` 프로토콜도 그대로다 — 새 구현이 아니라 같은 구현의 다른 전송이다.

### 1. `daemon/voice/gemini_live.py` — URL, 인증 헤더, 모델 경로

| | 현재 (API 키) | Vertex |
|---|---|---|
| URI | `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent` | `wss://{location}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` |
| 헤더 | `x-goog-api-key: <키>` | `Authorization: Bearer <토큰>` |
| `setup.model` | `models/{id}` | `projects/{p}/locations/{l}/publishers/google/models/{id}` |

(google-genai SDK의 `live.py:992`, `live.py:1050`, `_transformers.py:180`에서 확인한
형식. 우리는 SDK를 쓰지 않고 같은 형식을 그대로 만든다.)

구현된 방식: 클래스는 이미 `url=`을 주입받았고, 거기에

- `auth: Callable[[], dict[str, str]] | None` — 연결 시도마다 호출된다(`_auth_headers`).
  **문자열 토큰을 받지 않는다:** 토큰은 1시간 만에 만료되고 세션은 재연결하므로,
  재연결 시점에 새 토큰을 받아야 한다. 생략하면 지금의 `{"x-goog-api-key": key}`.
  `auth`가 있으면 API 키는 비어 있어도 된다 — 그 엔드포인트는 키를 받지 않는다.
  블로킹이라 `asyncio.to_thread`로 부른다(토큰 갱신이 네트워크 I/O이고, 같은 루프가
  마이크를 비우고 있다).
- 모델 경로는 `projects/`·`publishers/` 접두사를 그대로 통과시킨다. 붙이면
  `models/projects/...`가 되어 아무도 설정하지 않은 모델 이름으로 1008이 온다.

`_redact`와 로그 필터가 토큰까지 가린다. 필터는 생성자 인자 하나만 들고 있었는데,
토큰은 프로바이더가 돌기 전까지 존재하지 않으므로 `remember()`로 나중에 배운다
(`websockets`가 핸드셰이크 헤더를 DEBUG로 찍는다).

### 2. `daemon/config.py` — 설정 네 개

```
DAEMON_GEMINI_LIVE_TRANSPORT=vertex   # api_key(기본) | vertex
DAEMON_VERTEX_PROJECT=<project-id>    # vertex면 필수
DAEMON_VERTEX_LOCATION=us-central1    # 기본값
GOOGLE_APPLICATION_CREDENTIALS=<path> # 비우면 ADC
```

불리언이 아니라 선택 축으로 만든 이유: 엔드포인트가 둘 중 하나이고, 어드민에서
모델 고르듯 고르는 물건이며, 셋째가 생길 여지도 있다.

검증: transport가 목록에 없거나, `vertex`인데 project 또는 location이 비면
`ConfigError` — 즉 첫 음성 턴이 아니라 설정 로드에서 실패한다. `voice_provider=openai`
일 때는 검사하지 않는다(아무도 읽지 않는 값을 실패시키지 않기 위해).
`daemon doctor`의 config 줄이 `voice-transport=vertex project=… region=…
credentials=…`를 보고하고, 기본값일 때는 아무것도 덧붙이지 않는다.

### 3. `daemon/app.py` — 조립

구현을 import하는 유일한 자리(CONTRACTS 레이어링 규칙)이므로, 여기서
`url` / `headers_provider` / 모델 경로를 골라 `GeminiLiveSession`에 넘긴다.
`daemon/admin/settings_io.py`의 `STR_FIELDS` / `BOOL_FIELDS`에 새 설정을 올리면
어드민웹에서 전환할 수 있다 (모델 목록도 전송 경로에 따라 달라져야 한다 —
`_live_model_lists`가 API 키로 ListModels를 부르고 있으므로 Vertex에서는 그 목록이
틀린다).

## 새 의존성 하나

이 저장소는 Google SDK를 쓰지 않는다 (`pyproject.toml`의 dependencies에 없다 —
REST와 WS를 직접 만든다). Vertex 토큰만은 예외로 두는 것을 제안한다:

- `google-auth>=2.0`을 `[voice]` extra에 추가하고 **지연 import**한다. Vertex를
  끄고 쓰는 설치는 이 패키지를 건드리지 않는다.
- 직접 만드는 대안은 서비스 계정 키로 JWT를 서명해 토큰 엔드포인트와 교환하는 것인데,
  RSA 서명 의존성을 새로 들이는 셈이라 이득이 없다.

## 열린 질문 — 코드 전에 답해야 하는 것

1. **과금.** ReadyTalk의 단가표는 오디오 in $3.00 / out $12.00 per 1M
   (`readytalk/backend/app/services/pricing.py:29`). API 키 경로의 preview 빌드와
   실제 청구가 어떻게 다른지 측정하지 않았다. 상주 프로세스는 하루 종일 세션을 열므로
   이 차이가 코드보다 중요할 수 있다.
2. **자격증명의 주인.** 위 측정은 ReadyTalk의 서비스 계정으로 했다. Daemon이 쓸
   프로젝트와 서비스 계정을 따로 만들지, ReadyTalk 것을 공유할지는 소유자 결정이다.
   self-host 배포에서는 "API 키 한 줄"이 "GCP 프로젝트와 서비스 계정"으로 바뀌므로,
   `daemon setup`의 온보딩 난이도가 올라간다 — 기본값은 API 키 경로로 남겨야 한다.
3. **`[voice]` extra가 벗겨지는 문제.** `google-auth`를 그 extra에 넣었는데,
   `pyproject.toml`의 주석이 말하듯 원라이너 설치와 `daemon update`는 bare
   `daemon-ai`를 깔아서 이 extra가 조용히 빠진 전례가 있다(그래서 websockets와
   sounddevice는 darwin 코어로 옮겨졌다). macOS 소유자가 transport를 vertex로
   넘기면 ImportError를 만날 수 있고, 메시지가 무엇을 설치하라고 말하기는 하지만
   기본 경로에서 그런 일이 생기면 안 된다. 실제로 밟으면 darwin 코어로 옮긴다 —
   지금 옮기지 않은 이유는 기본값이 아닌 기능을 모든 설치에 얹기 때문이다.
4. **15분 상한.** 오디오 전용 세션은 양쪽 모두 15분 제한이다. 지금 재연결 로직이
   그것을 어떻게 넘기는지는 이 변경과 무관하지만, 재연결마다 토큰이 필요해진다는
   점에서 1번 항목과 만난다.

## 테스트 계획

이 저장소의 규칙: 테스트는 네트워크나 키를 만지지 않는다. 라이브 확인은 `evals/`에 둔다.

1. **단위** — URL/모델 경로/헤더 생성 세 개의 순수 함수. 가짜 소켓으로
   `headers_provider`가 **연결 시도마다** 호출되는지(재연결 시 새 토큰), `_redact`가
   Bearer 토큰을 가리는지.
2. **기존 테스트** — 갈라질 것으로 예상했던 `tests/test_voice.py`의 단정들
   (`setup["model"]`, `headers == {"x-goog-api-key": KEY}`, 1008 문구)은 **하나도
   수정하지 않았다.** API 키 경로의 동작이 그대로라서 106개가 그대로 통과했고,
   새 단정 7개가 추가됐을 뿐이다. 예측이 틀린 방향이 이쪽이라 기록해 둔다.
3. **설정** — transport 목록 밖의 값, project 없는 `vertex`, location이 빈
   `vertex` → 전부 `ConfigError`(`daemon doctor`가 잡는 그 종류).
   `voice_provider=openai`에서는 검사하지 않는다.
4. **reachability** — `tests/test_reachable.py`가 `vertex.ws_url` /
   `model_path` / `auth_headers`를 daemon/ 안에서 부르는 파일이 있는지, 그리고 그것이
   `app.py`인지 본다. 전송 경로는 *인자* 세 개라서 이 파일의 기존 검사에는 안 보인다.
5. **eval** — `evals/vertex_live_spike.py`는 이 저장소 자신의 웹소켓으로 한 턴을
   돌린다(URI·모델 경로·베어러 헤더·전사). 지연은 일부러 재지 않는다: 텍스트 턴은
   서버의 발화 종료 대기를 건너뛰고, 그 대기가 owner가 체감하는 대부분이다.
5. **ADR** — "API 키 하나로 붙는다"는 전제를 뒤집는 결정이므로 `docs/adr/`에 항목이
   필요하다. 이 저장소의 ADR 넷이 측정으로 뒤집힌 그 장르에 정확히 해당한다.

## 순서 (각 단계의 검증 포함)

1. ✅ `daemon/voice/vertex.py`의 순수 함수 3개 + 단위 테스트
2. ✅ `GeminiLiveSession(auth=...)` → 기존 voice 테스트 106개 그대로 녹색
3. ✅ `google-auth` 지연 import 토큰 프로바이더 (자격증명 실패는 permanent)
4. ✅ config 4개 + 검증 + `daemon doctor`의 transport 줄
5. ✅ `app.py` 조립 + 어드민 select (transport에 따라 모델 필드가 목록/자유입력으로 바뀜)
6. ✅ `evals/vertex_live_spike.py` — 이 저장소 클라이언트로 실제 통과 확인:
   setup 2.16초, PCM 466 KB, 전사 도착, 자격증명 fetch 1회/시도
7. ⬜ 이 머신의 `.env`를 넘기고 실제로 대화 → 검증: 대화 로그에서 무응답 턴 수를 센다
   (체감 아님 — `data/memory/log/`를 센다). 자격증명 주인 문제(열린 질문 2)가 먼저다.

## 이 변경이 하지 않는 것

- API 키 경로를 대체하지 않는다. 기본값으로 남고, 새 세대 모델은 그쪽에만 있다.
- `VoiceSession` 프로토콜, OpenAI Realtime 경로, 오디오 I/O를 건드리지 않는다.
- 목소리와 페르소나를 바꾸지 않는다 (이미 Despina + seed로 정해져 있다).
