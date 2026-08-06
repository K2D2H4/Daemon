# `daemon setup` guided finish — 설계

## 문제

지금 `daemon setup`은 모델·키·페르소나·페어링을 묻고 나면 화면에 세 줄을
출력하고 끝난다:

```
Next:
  daemon doctor     - checks Ollama, the data dir and the schema
  daemon run        - runs it here, in this terminal
  daemon install    - keeps it running after you close the terminal or reboot
```

즉 "상주로 올려두기"(`daemon install`)와 "잘 떠 있는지 확인"은 사용자가 나중에
직접 쳐야 하는 별개의 숙제로 남는다. 상주 등록은 proactivity의 전제조건인데
(docs/PLAN.md 3.1) — 상주 프로세스가 없으면 Daemon이 먼저 말을 걸 수 없다 —
온보딩이 그 지점을 안내 없이 떠넘긴다.

목표: setup 마지막에 "지금 상주로 올려둘까?"를 묻고, 예라면 설치 후 실제로
깨어났는지까지 한 흐름으로 보여준다.

## 범위 밖 (명시적으로)

- 새 install 구현. `daemon/service.py`의 `Service.install()`을 그대로 쓴다.
- 새 health 구현. 아래 "판정" 참고 — 이미 있는 두 신호만 쓴다.
- setup 흐름 재작성. 기존 `Wizard._finish`에 마지막 단계 하나를 얹을 뿐이다.

## 재사용하는 것

| 무엇 | 어디 | 쓰는 이유 |
|---|---|---|
| `Service.install()` → `ServiceAction` | `daemon/service.py` | 상주 등록. "이미 설치됨/plist가 다름/미지원 플랫폼(Windows→`ServiceError`)"을 이미 처리 |
| `Service.status()` → `ServiceStatus` | `daemon/service.py` | OS 레벨 "launchd/systemd가 실제로 띄웠나" (`installed`·`loaded`·`running`·`detail`) |
| `GET /health` | `daemon/app.py` | 상주 프로세스가 실제로 **응답**하는가. httpx로 블랙박스처럼 찌른다 — 기존 `Checks` 프로브(telegram·ollama·anthropic)와 동일한 방식이라 `app.py`/구현을 import하지 않는다 (CONTRACTS 레이어링 규칙 준수) |
| `service_for(settings)` | `daemon/cli.py` → `daemon/service.py`로 **이동** | settings로 `Service`를 만드는 단일 소스. cli.py·setup.py 둘 다 import (3줄짜리 생성 코드를 복제하지 않기 위해) |

### "daemon doctor를 재사용" 브리프에 대한 정정

브리프는 doctor의 health 로직을 재사용하라고 했으나, `daemon doctor`가 보는 것은
**설정**(스키마·Ollama 접속·데이터 디렉토리·라우팅)이다. 그 설정은 `_finish`
안에서 이미 검증된다(`Settings()`를 만들어 보고 실패하면 거기서 멈춤). doctor는
**돌고 있는 상주 프로세스를 확인하지 않는다.** 따라서 "깨어있다/문제있다"에
충실한 재사용은 doctor가 아니라 `service.status()` + 기존 `/health`다. (사용자
확인 완료.)

## 설계

### 위치와 조건

`Wizard._finish`에 **6단계 "Keep it running?"** 를 추가하고, 지금의 수동
"Next:" 세 줄을 이 단계가 대체한다. `STEPS`는 5 → 6.

이 단계가 **질문을 던지는** 조건: 대화형 터미널(tty)이고 **그리고** 지원
플랫폼(macOS/Linux)일 때만.

- **비대화형/CI**: 질문을 건너뛴다. wizard는 본래 대화형이고 CI는 `--check`
  경로(`report`)를 쓴다. tty 판정은 `Prompt`에 `interactive` 속성을 추가해
  기존 secret 경로의 `self._in is sys.stdin and self._in.isatty()` 검사와 같은
  신호를 쓴다.
- **미지원 플랫폼(Windows 등)**: 질문하지 않는다 (`service.install()`이
  `ServiceError`를 던질 것이므로 물어봐야 소용없다).

두 경우 모두 아래 "아니오" 안내와 같은 수동 힌트를 출력한다.

### 질문

기본값 **예**(Y/n). 상주가 이 제품의 의도된 최종 상태이고(proactivity의 전제),
`daemon uninstall`로 되돌릴 수 있다.

> Install it as a background service now, so it keeps running after you close
> this terminal and survives a reboot? This is what lets Daemon reach out on
> its own.

### 예 → 설치와 판정

1. `action = service.install(force=False)`.
   - `ServiceError`(미지원/경로 문제) → 그 메시지를 `status(..., "warn")`로
     보여주고 수동 힌트 유지. (여기 도달하는 건 뜻밖의 경우 — 위에서 플랫폼을
     이미 걸렀으므로 방어적.)
   - `applied=False` + `changes`(이미 설치됨인데 plist가 다름) → "이미 있고
     내용이 다릅니다. `daemon install --force`로 교체하세요"를 보여주고 판정은
     건너뜀.
   - 그 외(`applied=True`, 또는 "already installed and unchanged; loaded") →
     판정으로.

2. **살아있음 판정** (부팅 레이스 처리 위해 짧게 폴링, 최대 ~10초):
   - 먼저 `service.status().running`이 참이 되길 기다린다.
   - 그다음 `GET http://{settings.host}:{settings.port}/health`가 200 +
     `status == "ok"`로 응답하길 기다린다.
   - 둘 다 만족 → ✅ **"Daemon is running and answering."**
   - 타임아웃(프로세스가 안 뜸/응답 없음) → ⚠️ **"Installed, but it isn't
     answering yet."** + `status().detail` + 아래 안내.

   crash-loop(잘못된 설정으로 프로세스가 죽고 launchd가 `THROTTLE_SECONDS=30`s
   대기) 도 이 경로로 정확히 "문제있다"가 된다.

3. **문제있다일 때 안내** (doctor를 자동 실행하지 않는다 — 포인터만, 사용자
   확인 완료):
   - `daemon status` — 상주 상태 재확인
   - `daemon doctor` — 설정 점검
   - 에러 로그 경로: `service.err_log`

### 아니오 / 비대화형 / 미지원 → 수동 안내

> You can do this later with `daemon install`. Until then Daemon only runs
> while `daemon run` is open, and it can't reach out on its own.

## Seam (테스트 가능성)

이 모듈의 "모든 부수효과는 seam" 원칙(`setup.py` 모듈 docstring)을 따른다.

- **`/health` 프로브**: 주입되는 `Checks` 묶음에 `health: Callable[[str], HealthState]`
  추가. 실제 서버 없이 테스트.
- **`Service` 팩토리**: `run()`/`Wizard`에 `service_factory: Callable[[Settings], Service]`
  주입. 기본값은 `service.service_for`. 테스트는 스크립트된 `install()`/`status()`
  를 가진 가짜 `Service`를 넘겨 `launchctl` 없이 검증.
- cli.py의 `_setup`은 별도 인자 없이 기본 팩토리를 쓰면 된다(기본값이
  `service_for`이므로). `cli.service_for`를 monkeypatch하던 기존 테스트는
  `from daemon.service import service_for` 후에도 cli 네임스페이스에 이름이 남아
  그대로 동작.

## 레이어링

setup.py가 service.py를 import한다(`service_for`, `Service`, `ServiceAction`,
`ServiceStatus`, `ServiceError`). service.py는 config·fs만 의존하는 말단
유틸리티이고 cli.py가 이미 import한다 — 순환 없음, 프로토콜 뒤에 숨은
provider/channel 구현이 아니므로 레이어링 위반 아님. `/health`는 import이 아니라
HTTP 프로브라 app.py 내부에 손대지 않는다.

## 검증

- `python3 -m pytest`, `ruff check .`, `scripts/check_docs.py`,
  `scripts/check_landing_claims.py` 전부 통과.
- 유닛테스트로 끝내지 않는다: 이 머신(macOS/LaunchAgent)에서 `daemon setup`을
  상주 등록까지 실제로 끝까지 돌려 ✅/⚠️ 줄을 눈으로 확인. 실패 경로도
  (설치 후 프로세스가 안 뜨는 상황을 만들어) 한 번 본다.
