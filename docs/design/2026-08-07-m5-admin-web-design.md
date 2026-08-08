# M5 설계 — 어드민 웹

작성 2026-08-07. `/grilling` 한 패스로 수렴한 구현 계약이다. 병렬로 일하는
에이전트가 서로 다른 것을 짓지 않게 하는 것이 이 파일의 목적이고, 산문보다
결정과 시그니처가 중요하다.

근거: PLAN §3(웹은 후순위 인터페이스), §4.5(자체 UI 필요), §8.2 M5 행, §10(오리진
게이트·SSRF). 규칙: CONTRACTS 9(단일 프로세스)·10(오리진 게이트)·12(도구 기본 full,
MCP/브라우저는 자체 스위치)·13(데이터로 코드 만들지 않음). 재사용: 메모리
`conversational-mcp-install`이 수렴시킨 MCP 엔진 설계 — **재론하지 말고 재사용한다.**

## 무엇이고 무엇이 아닌가

운영·점검용 로컬 관리 화면이다. 네 가지: 헬스 열람, 무부작용 채팅 테스트, 설정
편집, MCP 서버 추가(OAuth 포함). **4번째 대화 채널이 아니다** — 진짜 대화는
텔레그램·보이스가 맡는다.

기존 `daemon/app.py`의 FastAPI에 라우트를 더 붙인다. 별도 서버·프로세스 없음
(CONTRACTS 9). 이미 uvicorn이 `settings.host:settings.port`(기본 `127.0.0.1:8787`)에
`/health` JSON을 서빙하고 있다 — 어드민은 그 위에 얹는다.

## 결정 (grilling에서 확정, 뒤집지 말 것)

1. **바인딩·인증**: loopback-only, **인증 없음**. 로컬=주인으로 신뢰. 비밀번호는
   필요 시 후속. → 이 결정이 아래 2를 강제한다.
2. **채팅 테스트 = 부작용 0**. 라우팅된 provider 왕복만. **메모리 기록 ✗, 도구 ✗,
   owner origin 아님.** 이유: 인증이 없으므로 웹이 owner origin을 정직하게 증명할 수
   없다. 도구 실행 경로가 웹에 존재하지 않아야 `127.0.0.1`을 때리는 임의 로컬
   프로세스·SSRF가 셸을 얻는 CONTRACTS 10 구멍이 원천적으로 막힌다. 채팅에 친 문장이
   `loop.py`의 "기록 후 임베딩"을 타 영구 회상 행이 되는 문제도 함께 사라진다.
3. **설정 즉시반영 = 자체 재시작**. 런타임 핫리로드 아님. 검증 후 `.env`에 쓰고
   graceful 종료 → launchd(macOS `KeepAlive`)/systemd(`RestartSec`)가 부활
   (`daemon/service.py`가 이미 설치). "CLI로 재시작"은 일반 유저에게 블로커라 배제.
4. **프론트 = 서버 렌더 + 바닐라 JS, 빌드 스텝 없음**. `site/index.html`의 디자인
   시트를 이식한다. API는 JSON으로 깔끔히 유지해 나중에 Vite/SPA로 갈아탈 여지만
   남긴다(공짜 보험). CDN 금지(오프라인 동작 보존).
5. **MCP는 OAuth 포함**, 2a(키)→2b(OAuth) 순서. 레지스트리 검색은 범위 밖(신뢰 불가
   자유 텍스트 = 인젝션 표면, 수율 낮음 — 후속).
6. **제외(명시적 후속)**: 원격 접근, "배운 것" 열람 UI, 레지스트리 검색, 런타임
   핫리로드, 비밀번호 인증.

## 디자인 시트 이식

`site/index.html`의 `:root` 토큰을 그대로 쓴다. 새로 정하지 말 것:

```
--accent:#A78BFA; --accent-mid:#8B5CF6; --accent-deep:#6D3FD4;
--canvas:#120F18; --surface:#1B1626; --raised:#241D33; --well:#0D0B12;
--line:#2E2740; --line-strong:#493D66;
--tp:#FFFFFF; --tb:#ECE7F5; --tm:#8E85A3; --tf:#7A7192; --td:#5A5470;
--ok:#5EE1A4; --warn:#FFC75A; --error:#FF6B5E;
--pix:'Silkscreen',...(제목);  --mono:'DM Mono',...(본문)
```

캐릭터: `docs/assets/banner.png` / `states.png`. 폰트는 벤더링(CDN 금지, 오프라인).

## 파일

```
daemon/admin/__init__.py
daemon/admin/routes.py       FastAPI APIRouter. app.py가 마운트한다.
daemon/admin/settings_io.py  .env 검증 후 쓰기(fs.write_private_replace 재사용, 0600).
                             쓰기 전 반드시 Settings(**candidate)를 구성해 성공할 때만 커밋.
daemon/admin/restart.py      supervised 여부 감지 + graceful 종료 트리거.
daemon/admin/static/         서버 렌더 산출물(HTML/CSS/JS/폰트/PNG). 빌드 없음.
daemon/mcp_catalog.py        (Phase 2) 신뢰 카탈로그. CatalogEntry 데이터클래스.
tests/test_admin.py          acceptance: TestClient로 API 직접 구동.
```

`daemon/app.py`는 라우터를 마운트하고 `app.state`에 필요한 핸들(gateway·settings·
mcp bridge)을 노출한다 — layering 예외는 app.py에만 (CONTRACTS 4).

## JSON API (계약 — 테스트는 이걸 때린다)

```
GET   /admin/                     서버 렌더 셸(정적)
GET   /admin/api/health           기존 /health 재사용/프록시
POST  /admin/api/chat-test        {text} -> {reply}. 부작용 0. gateway.complete만.
GET   /admin/api/settings         현재 설정. 비밀값은 마스킹("설정됨"/null).
PATCH /admin/api/settings         검증 -> .env 쓰기 -> {restart_required, supervised}
POST  /admin/api/restart          supervised일 때만. graceful exit.
--- Phase 2 (DAEMON_MCP_ENABLED 뒤) ---
GET   /admin/api/mcp/catalog      카탈로그 목록(auth kind: none|key|oauth, oauth_verified).
GET   /admin/api/mcp/servers      설정된 서버 + 연결 상태(_tools_health 재사용).
POST  /admin/api/mcp/connect      {name, secret?} -> mcp.json 먼저 쓰고 연결.
DELETE /admin/api/mcp/servers/{n} unregister + disconnect.
--- Phase 2b (OAuth) ---
POST  /admin/api/mcp/oauth/start  {name} -> {authorize_url}
GET   /admin/api/mcp/oauth/callback  code 수신 -> 토큰 저장(0600) -> 연결
```

편집 대상(3의 검증을 통과해야 커밋): `preset`, `hosted_provider`, `route_overrides`,
`voice_enabled`, `recall_limit`, `recall_half_life_days`, 도구 `mode`,
`DAEMON_MCP_ENABLED`, `DAEMON_BROWSER_ENABLED`, provider API 키(마스킹, 간접저장).
읽기 전용: `host`/`port`, 데이터 디렉토리. 제외: 스케줄 시각(CLI로 충분).

## MCP 엔진 (Phase 2 — 메모리 `conversational-mcp-install`에서 재사용)

- **해상**: 카탈로그(신뢰, 구조화 필드) 우선. 레지스트리 fallback은 이번 범위 밖.
- **비밀값 간접저장**: 값은 `.env`(0600), `mcp.json`엔 env-var *이름*만 → 설정 파일은
  비밀 없이 공유 가능. 연결 시 stdio는 env 한 개, url은 `Authorization: Bearer`
  헤더로 주입(**신규 코드** — 현재 `mcp.py`는 `streamablehttp_client(url)`만 호출,
  auth 없음). `os.environ` 전체를 자식에 넘기지 말 것(Daemon provider 키 유출).
- **핫리로드**: `Registry`에 `unregister` 추가(frozen 아님). `McpBridge`를 공유
  `AsyncExitStack` 하나에서 **서버별 stack**으로 전환 — 한 서버가 독립적으로
  연결/해제. register/unregister/specs는 sync라 lock 불필요하나, 드문 교차턴 설치는
  bridge 수준 `asyncio.Lock`으로 싸게 보호. `mcp.json` 먼저 쓰고 → 연결. 연결 실패는
  엔트리를 남긴다(load_config + failures + /health가 "설정됐으나 미연결"을 이미 처리).

## OAuth (Phase 2b)

SDK가 지원함(실측): `mcp.client.auth.OAuthClientProvider` 존재,
`streamablehttp_client(url, ..., auth=, headers=)`가 `auth`를 받음.

- **콜백**: `redirect_uri = http://127.0.0.1:<port>/admin/api/mcp/oauth/callback`.
  loopback 웹이라 성립하는 것 — CLI/채팅이 이걸 못 해서 이전 grilling이 v2로 미뤘다.
- **토큰 저장**: `TokenStorage` 구현, `fs.write_private_replace`로 0600 파일에
  at-rest 저장. 갱신은 SDK가 저장소가 있으면 처리. 서비스 unit 파일엔 비밀 금지
  (`service.py` 규칙과 동일).
- **호환성 플래그**: 카탈로그에 `oauth_verified`. 동적 클라이언트 등록(RFC 7591)
  지원 + localhost 리다이렉트 허용을 우리가 실제 확인한 서버만 원클릭 노출. (Notion이
  1차 타깃.)

## 테스트 게이트

CONTRACTS: reachable + acceptance + e2e, 그리고 **실제 구동**.

- **reachable**: 새 라우트/핸들러가 호출자 있음을 선언. Phase 2b(OAuth) 등 미구현은
  `PENDING_*`에 M5로 선언(스테일 PENDING도 실패하므로 착수 시 닫는다).
- **acceptance**(`tests/test_admin.py`, `TestClient`): PATCH가 검증 통과 후에만
  `.env`에 쓰이는가 / 잘못된 값은 쓰이지 않고 400인가 / chat-test가 메모리·도구를
  건드리지 않는가(가짜 provider로 reply만) / MCP connect가 가짜 서버를 등록하는가.
- **e2e**: 실제 `create_app` 부팅해 엔드포인트 확인.
- **네트워크·프로세스 엣지 = 수동 QA**: 자체 재시작의 "종료→부활"은 유닛 불가 →
  검증·`.env` 쓰기까지만 자동화. OAuth 토큰 교환은 네트워크 엣지라 그 지점만 가짜
  (모델·임베더·텔레그램을 가짜로 두는 것과 동일 원칙). 마무리로 `daemon run` 띄워
  브라우저로 직접 확인(qa 에이전트).

## 게이트 (PLAN §8.2 M5)

> 나는 브라우저에서 헬스를 보고, 채팅을 무부작용으로 테스트하고, 프리셋을 바꿔
> 자체 재시작으로 반영시키고, Notion을 원클릭 OAuth로 붙일 수 있다.

## 구현 후 QA에서 나온 것 (2026-08-07, 실제 부팅)

빌드 뒤 실제 `daemon run`으로 몰아본 결과 대부분 통과했고, 초록 테스트가 못 잡은
결함 하나가 나와 고쳤다.

- **결함(고침): MCP 브리지가 채널 토큰에 결합돼 있었다.** `_build_tools`(=`app.state.mcp`를
  세우는 곳)가 `_build_io` 성공 브랜치 안에서만 돌았고, `_build_io`는 Telegram 토큰이
  없으면 `_build_channel`에서 raise했다. 그래서 토큰 없는 로컬 전용 부팅(어드민 웹의 핵심
  유스케이스)에서 `app.state.mcp`가 None으로 남아 MCP 탭이 아무것도 연결 못 했다. 게다가
  `app.state.channel`/`memory` 대입 자체가 대화 루프 가드 안에 갇혀, 메모리가 실제로 떠
  있어도 `/health`가 이를 과소보고했다. **수정**(`daemon/app.py`): (1) `_build_io`가
  `_build_channel` 실패를 관용해 `channel=None`으로 계속하되 store·memory·recall·tools는
  유지 — `build_proactive_tick`가 이미 자기 `_build_channel`에 적용하던 관용과 동일. (2)
  `app.state.channel`/`memory` 대입을 루프 가드 밖으로 빼서, 루프(=채널 필요)만 가드 안에
  남김. 회귀 테스트: `tests/test_e2e.py::test_a_missing_channel_still_brings_up_memory_tools_and_mcp`.
  실제 재부팅으로 확인: 토큰 없이 `connect {"name":"time"}`가 더는 "bridge not running"이
  아니라 영속+정직한 연결실패(502)로 응답.

- **결정(코드 유지): `oauth/start`는 `oauth_verified`로 게이트하지 않는다.** UI는 미검증
  서버의 "Connect with OAuth" 버튼을 비활성화(기본 가드)하지만, API 경로 자체는 열어 둔다 —
  owner가 새 서버를 라이브로 검증해 `oauth_verified`를 플립하려면 바로 이 경로가 필요하고,
  여긴 loopback·owner 전용 표면이라 의도된 마찰로 충분하다.

- **수동 벽(코드 아님)으로 남는 것:**
  - **OAuth 라이브 검증** — Notion 실계정·실네트워크로 동적 클라이언트 등록 + localhost
    콜백을 확인한 뒤 `daemon/mcp_catalog.py`의 `notion.oauth_verified=True`로 플립. 메커니즘·
    조정·토큰저장·테스트(네트워크 엣지 가짜)는 완성. M3/M4처럼 코드-완료/게이트-열림.
  - **레퍼런스 uvx 서버 라이브 연결**은 이 환경의 `mcp` SDK ↔ `uvx`가 resolve하는 패키지
    버전 skew(`ImportError: McpError`)로 막힘 — 어드민 코드가 아니라 의존성 문제이고,
    어드민은 이를 영속+502로 정직하게 처리한다.
