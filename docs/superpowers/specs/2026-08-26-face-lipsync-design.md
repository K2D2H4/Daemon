# 립싱크 — 스위치 뒤의 두 번째 렌더 경로

v1의 얼굴은 `speaking` 동안 클립을 틀고 `playbackRate`를 RMS 엔벨로프로 변조한다.
음소는 원리적으로 낼 수 없다(§3.4). 이 문서는 그 자리에 MuseTalk을 끼우는 설계다.

**전제는 측정이다.** 모델 선택, 지연 예산, 처리량, 화질, 기각된 대안은 전부
[2026-08-25-face-design.md](2026-08-25-face-design.md) §6에 있다. 여기서 되풀이하지
않고, 그 숫자가 이 설계의 형태를 어떻게 강제하는지만 적는다.

## 무엇이고 무엇이 아닌가

**`speaking` 렌더 경로만 대체한다.** activity 상태, mood 원샷, SSE, 크로스페이드,
`_playback_until` 전부 그대로다. 12개 클립도 그대로 필요하다 — 스위치가 꺼지면 v1이
돈다.

**기본값은 꺼짐이다.** 가중치 1.8GB를 받아야 하고, 켜져 있는 동안 1.75GB를 문다. 켜지 않은
설치는 아무것도 지불하지 않는다 — `daemon/app.py`가 `/face`를 지연 import하는 이유와
같다.

## 승인된 결정

| | 결정 | 근거 |
|---|---|---|
| 발화 시작 | **오디오를 붙잡지 않는다.** 첫 ~200ms는 입을 포기한다 | 붙잡으면 모든 응답 시작이 191ms 늦는다. 주인의 최우선 결정이 지연이었다 |
| 런타임 | **전부 MLX.** torch 계열 없음 | 패키징(원라이너 설치)과 프레임워크 왕복 제거. 직접 써야 하는 건 TAESD뿐 |
| 스위치 | **실시간 토글**, 기본 꺼짐 | 합격 기준이 "느낌"이라 같은 대화 안에서 A/B 해야 한다 |
| 전송 | **서버가 완성 프레임, MJPEG** | 패치 전송은 브라우저→데몬 피드백이 필요한데 §2가 일부러 없앤 방향이다 |

## 1. 모듈과 레이어링

**`daemon/face_lipsync.py`** — 렌더러. `mlx`, `numpy`, `cv2`만 import하고 **daemon
구현은 하나도 import하지 않는다**(비협상 4). 안에 든 것:

- 모델 셋 (UNet / whisper 인코더 / TAESD)
- 클립 캐시 핸들 (mmap)
- PCM 링과 렌더 루프
- 최신 JPEG 슬롯

**`daemon/face.py`** — **`FaceBus`가** sink 하나를 주입받고 `SpeechClock.fed()`가
그것을 호출한다. 기본 `None`이면 v1 동작 그대로이고 비용 0이다.

**`daemon/app.py`** — 플래그가 켜져 있고 에셋이 있을 때만 렌더러를 만들어 주입한다.
다른 어떤 모듈도 `face_lipsync`를 import하지 않는다.

## 2. 오디오 급수 — `SpeechClock`이 이미 옳은 타임라인을 갖고 있다

`fed()`는 청크를 받을 때 **그것이 들리기 시작하는 시각**을 계산한다:

```python
starts = max(self._until, at)      # 큐에 밀린 것이 있으면 미래
self._until = starts + seconds
```

립싱크가 필요한 것이 정확히 이것이다. `conversation.py`가 barge-in에 쓰는 것과 같은
산술이므로 이미 검증돼 있고, 새로 배선할 것이 없다.

```python
PcmSink = Callable[[bytes, float], None]   # (chunk, audible_at)
```

**sink는 `SpeechClock`이 아니라 `FaceBus`가 든다.** `conversation.py`는
`SpeechClock(face, sample_rate=..., bytes_per_frame=...)`로 **버스 하나만** 넘기므로,
sink가 버스에 얹혀 있으면 `voice/` 아래가 **한 줄도 바뀌지 않는다.** `SpeechClock`은
`fed()`에서 `self._bus.pcm(chunk, starts)`를 부르기만 한다.

§6이 울타리를 친 경로이고 PortAudio 데드락 이력이 있는 곳이라, 이 seam이 설계의
제약을 실제로 지키는 유일한 형태다.

이 급수는 발화 시작의 동작도 자동으로 정한다. 턴이 시작될 때는 큐가 비어 `starts == at`
이라 미래 오디오가 없다. MuseTalk은 범위 밖 인덱스를 엣지로 클램프하므로(§6, 측정)
첫 ~200ms가 중립으로 **알아서** 떨어진다. 특별 처리 코드가 필요 없다.

## 3. 렌더 루프

발화가 시작되면 태스크가 뜨고 `_playback_until`이 지나면 멈춘다. 틱마다:

```
재생 경과 → 구동 프레임 인덱스
PCM 링에서 200ms 윈도 (과거 120ms / 미래 80ms)
  → whisper 인코더 (30초 mel, 7.44ms)
  → UNet + TAESD, N=2 (33.17ms/프레임, 듀티 사이클 실측)
  → 구동 프레임에 numpy 합성 (2.42ms, CPU)
  → JPEG (0.46ms, CPU)
  → 슬롯에 덮어쓰기
```

GPU 쪽 합이 **7.44 + 33.17 = 40.61ms**이고 예산이 41.67ms다.

**N=2가 유일하게 가능한 배치다.** 배치 채우기 대기 `(N-1)×41.67ms`가 지연을 먹어
N=3은 250ms를 7.6ms 넘긴다(§6 측정).

**슬롯은 큐가 아니라 덮어쓰기다.** §2가 level에 대해 정한 것과 같은 이유 — 창이 가려져
소비가 느려질 때 밀린 프레임이 쌓이면 입이 소리보다 뒤처진 채 따라온다.

**CPU 합성은 GPU와 겹친다.** MuseTalk 자신의 `realtime_inference.py`도 별도 스레드로
돌린다. 처리량 제약은 합이 아니라 GPU 쪽 40.61ms 대 41.67ms이고, **여유가 2.5%다.**

## 4. 오프라인 전처리와 에셋

빌드 스텝 하나가 구동 클립의 캐시를 만든다. 런타임에 **얼굴 검출기가 필요 없다** —
그것이 전처리를 분리하는 이유다.

```
<data_dir>/face/lipsync/
  models/
    unet.safetensors        1.70GB  mlx-community/MuseTalk-1.5-fp16
    whisper/                        mlx-whisper tiny
    taesd.safetensors         9MB   madebyollin/taesd
  idle2/
    frames.raw             1.0GB   193 × 1080×1620×3, mmap
    latents.safetensors      6MB   참조 잠재 (마스크 + 참조 concat)
    masks.npz                      블렌드 마스크
    boxes.json                     bbox와 crop_box
```

**함정 하나 — `unet.safetensors`에 transpose를 걸면 안 된다.** 키 이름은 diffusers
스타일이지만 레이아웃은 이미 MLX(NHWC)로 변환돼 있다(`conv_in.weight`가
`(320,3,3,8)`, torch는 `(320,8,3,3)`). 로더는 키 이름 변경과 `ff.net.0.proj`
분할(686 → 718)만 하고 transpose는 건너뛴다. 조용히 망가지는 종류의 실수라 로더에
수치 검증을 붙인다.

가중치는 번들하지 않고 명령으로 받는다.

## 4-1. 전처리는 Apple Vision을 쓴다 — 런타임도 빌드도 torch가 없다

주인이 §4대로 자기 클립을 만들 수 있어야 하므로 전처리는 **돌아가는 스텝**이어야 하고,
캐시만 배포하고 끝낼 수 없다. 그런데 전처리에 필요한 것은 68점 랜드마크 하나뿐이고,
그것 때문에 torch를 끌어오면 "전부 MLX" 결정이 무너진다.

macOS의 **Vision 프레임워크**가 얼굴 랜드마크를 준다(pyobjc). `MuseTalk-Metal`이 같은
이유로 OpenMMLab을 그것으로 대체했다. 스파이크는 FAN을 썼지만 그건 측정 편의였고,
제품은 Vision을 쓴다.

**대체는 검증이 필요하다.** Vision의 랜드마크 체계는 iBUG-68이 아니므로 MuseTalk의 박스
공식이 인덱싱하는 해부학(`lm[29]` = 코 아래 능선, 턱 최하점)에 대응시켜야 한다. 스파이크가
FAN에 대해 한 것과 같은 검증을 건다 — 박스를 그려서 눈으로 확인하고, FAN 박스와의 차이를
잰다. 대응이 안 맞으면 화질이 조용히 나빠지는 종류다.

## 5. 전송과 페이지

**`GET /face/frames`** — `multipart/x-mixed-replace`. 발화 중에만 흐르고 그 외에는
열려만 있다. 174KB/프레임, 24fps에서 34.2 Mbit/s — 루프백이다.

기존 **`/face/stream` SSE는 손대지 않는다.** activity·level을 계속 실어 보낸다.

`face.html`은 매니페스트가 립싱크 가용을 알리고 activity가 `speaking`일 때 speaking
클립 대신 `<img src="/face/frames">`로 크로스페이드한다. §3.2의 "들어오는 요소만
페이드" 규칙 그대로다.

**`/face/*`는 읽기 전용으로 남는다.** 토글은 admin 표면에 둔다 — 그쪽은 이미 오리진
게이트가 있고, §2가 face 표면을 부수효과 없이 만든 것은 의도였다.

## 6. 스위치

```python
face_lipsync_enabled: bool = Field(default=False, alias="DAEMON_FACE_LIPSYNC_ENABLED")
```

관례 그대로다(`voice_enabled`, `wake_enabled`, `browser_enabled`가 전부 기본 꺼짐).

실시간 토글은 가중치를 그때 실체화하고(693ms, 측정) 끌 때 놓는다. MLX가 safetensors를
mmap하므로 `mx.load`는 3ms이고 상주는 수요 기반이다.

## 7. 열화

| 상황 | 동작 |
|---|---|
| 플래그 켜짐, 에셋 없음 | 렌더러를 만들지 않는다. 매니페스트가 `lipsync: false`. 페이지는 v1 클립 (§3.7 선례) |
| 렌더러가 중간에 실패 | v1 클립으로 떨어지고 로그는 **한 번만**. 프레임마다 재시도하지 않는다 |
| 유휴 | `mx.clear_cache()`. **최적화가 아니라 요구사항** — 빠지면 드리프트가 돌아온다(측정) |
| 긴 유휴 | 가중치를 놓는다. 재실체화 693ms |

**두 프레임워크를 한 프로세스에서 함께 돌리지 않는다.** 스파이크에서 MLX와 PyTorch
MPS를 같이 돌렸을 때 캐시가 자라 드리프트가 +41.2%까지 갔고 개발 머신이 한 번 멈췄다.
전부 MLX인 이유의 절반이 이것이다.

## 8. 테스트

렌더러는 주입받은 **엔진 프로토콜** 뒤에 있다:

```python
class LipsyncEngine(Protocol):
    def mouths(self, audio: NDArray, frame_indices: Sequence[int]) -> list[NDArray]:
        """256x256 BGR, `frame_indices`와 같은 길이. 모델이 사는 유일한 자리."""
```

링 관리·프레임 인덱스 계산·합성·전송은 전부 이 프로토콜 **밖**에 있으므로 모델 없이
테스트된다. 테스트는 단색을 돌려주는 가짜를 쓴다 —
테스트에 모델이 들어가지 않고 CI가 GB급 가중치를 만질 일이 없다.

- **단위** — PCM 링의 윈도 계산(80ms 룩어헤드, 턴 시작의 엣지 클램프), 재생 시각 →
  구동 프레임 인덱스, 슬롯이 큐가 아니라 덮어쓰기인지, 에셋 없을 때의 열화, 로더가
  transpose를 걸지 않는지
- **reachability** (`tests/test_reachable.py`) — `/face/frames`에 호출자가 있는지,
  렌더러를 무언가가 구성하는지
- **acceptance** — 스위치가 꺼졌을 때 v1 경로가 그대로 도는지
- **실물** (`evals/`, 손으로) — 얼굴 창을 열고 대화하며 토글을 껐다 켠다. 합격 기준이
  "느낌"이라 자동 검증이 구조적으로 닿지 않는다(§테스트). **이 스위치가 존재하는 이유가
  이 비교다.**

## 계약 준수

| 규칙 | 어떻게 지키는가 |
|---|---|
| 2 (Recall Lane 1은 LLM 호출 0) | 립싱크는 recall도 LLM도 아니다. 무관 |
| 4 (레이어링) | `face_lipsync.py`는 daemon 구현을 import하지 않는다. `app.py`가 조립한다 |
| 9 (단일 프로세스) | 새 프로세스도 브로커도 없다. 프로세스 내 태스크이고, 규칙이 명시적으로 허용하는 형태다 |
| 12 (툴 감사 행) | 툴을 만들지 않는다 |

프로즌 파일은 건드리지 않는다. `tasks.py`에 새 `Task` 없음, `schema.sql`에 새 테이블
없음, 프로토콜 파일 변경 없음. **`voice/` 아래도 건드리지 않는다.**

## 넣지 않는 것

- **패치 전송** — 대역폭 3.4배 이득이지만 브라우저→데몬 피드백이 필요하다. A가 돈 뒤의
  최적화로 남긴다
- **오디오 붙잡기** — 승인된 결정
- **구동 클립 외의 클립** — 립싱크는 하나만 구동원으로 쓴다. `speaking_soft`/`loud`는
  립싱크 경로에서 기여하지 않는다(마스크되어 버려진다)
- **립싱크로 표정 표현** — mood는 §5 그대로 원샷 클립

## 구현 순서

한 번에 가지 않는다. 각 단계가 끝나면 돌려볼 수 있어야 한다.

1. **로더와 엔진** — MLX UNet + whisper + TAESD를 붙이고, 수치 검증(코사인)을 테스트로
   건다. transpose 함정이 여기서 잡힌다
2. **전처리** — Vision 랜드마크 → 박스 검증 → 캐시 생성. 결과를 눈으로 본다
3. **렌더러** — PCM 링, 프레임 인덱스, 슬롯. 가짜 엔진으로 단위 테스트
4. **배선과 전송** — `app.py` 조립, `/face/frames`, 매니페스트, 페이지 전환
5. **스위치** — 설정 플래그, admin 토글, 열화 경로
6. **실물** — 얼굴 창에서 토글을 껐다 켜며 본다. 이것이 합격 판정이다

## 열린 질문

1. 토글의 admin 표면 정확한 형태 — 엔드포인트인가 CLI인가, 둘 다인가
2. 가중치 다운로드 명령의 이름과 위치 (`daemon face lipsync install`?)
3. 긴 유휴의 "긴"이 몇 분인가 — 실사용에서 조정
4. 긴 유휴 뒤 첫 발화의 693ms 예열을 어디서 숨기는가 — `thinking` 상태가
   그만큼은 되는지 실사용에서 본다
