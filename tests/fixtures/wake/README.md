# Wake-gate fixtures

Synthetic Korean speech, generated once with the only ko_KR voice this project's
target machine has:

```bash
say -v Yuna -o wake-alone.wav --data-format=LEI16@16000 "헤이 데몬"
say -v Yuna -o wake-and-question.wav --data-format=LEI16@16000 "헤이 데몬, 지금 뭐 하고 있어"
say -v Yuna -o no-wake-word.wav --data-format=LEI16@16000 "오늘 날씨가 참 좋네요"
```

16 kHz, mono, 16-bit LE - the rate `AudioIO.sample_rate` captures at, so a test can
feed them straight in with no resampling.

**These are TTS, not a person**, and that limit is the point of `daemon wake
calibrate`: the on-device recognizer never emits a coined name, so what it
actually returns for a given speaker is measured rather than assumed. Measured here,
stable across three runs each: `헤이 데몬` -> `헤이 대문`, `데몬` -> `질문`,
`루시` -> `루씨`, `루시야` -> `루시`. A real voice will have its own stable set,
which is why the aliases live in `.env` rather than in the code.

Committed rather than generated at test time: `say -v Yuna` exists on one developer's
machine and CI has no Korean voice, and a fixture that only some machines can build
is a test that only some machines run.
