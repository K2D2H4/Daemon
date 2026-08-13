# 0012 — Split the presence signals

**Status:** accepted · 2026-08-11 · measured

## Context

`Reading.audio_busy` was one bool over both directions of the default audio
device: true if the microphone was running for anybody, or the output was, or
both. [PLAN.md](../PLAN.md) §6.4 already knew this signal was noisy — it had
already been demoted once, from blocking the utterance to only costing the
speaker, after a system-wide audio EQ on the development machine kept it `True`
all day. What nobody had tried yet was turning on the feature that shares the
same hardware: the wake listener (`DAEMON_WAKE_ENABLED`) holds the microphone
for as long as it is armed, which on a voice-enabled install is effectively
always.

Measured on this machine: with the wake listener running, `audio_busy` read
`True` continuously — input running, output free, five seconds straight,
sampled with no call in progress and nothing playing. The gate's routing table
(`Gate._route`, `daemon/proactivity/gate.py`) reads a busy audio device as "a
call in progress" and downgrades the utterance to Telegram. So turning voice on
is what turned the speaker route off. A feature and its own precondition were
fighting each other through one shared bool, and the direction of the fight
means the bug was invisible until DAEMON_WAKE_ENABLED shipped — every earlier
measurement of the routing table was against a reading nothing else was
touching.

## Decision

Split `Reading.audio_busy` into `mic_busy` and `output_busy`
(`daemon/proactivity/base.py`, frozen), backed by two separate CoreAudio reads
(`kAudioDevicePropertyDeviceIsRunningSomewhere` on `dIn`/`dOut`,
`daemon/proactivity/presence.py`). `mic_busy` subtracts the daemon's own hold
before answering: `daemon/mic_hold.py` is a reentrant counter that the wake
listener and any active voice session increment and decrement, and
`MachinePresence._mic_busy` asks it first — if we hold the device, the answer
is "not busy" without touching CoreAudio at all, because holding it ourselves
is not a call. `daemon/proactivity/presence.py` cannot import the voice layer to ask it directly
(a text-only install has no PortAudio, and that install still has to answer
presence), so the audio layer *tells* `mic_hold` and presence *asks* it -
neither imports the other. Verified live and pinned:
`tests/test_presence.py::test_our_own_microphone_hold_is_not_a_call`,
`test_somebody_elses_microphone_hold_is_a_call`,
`test_reading_separates_microphone_from_output`.

Four more decisions came packaged with the split, because pulling one bool
apart moved the ground under four things that had been reasoning about it as a
whole.

### 1. Type E may quote memories whose `origin` is `'owner'`

Every other candidate generator (`daemon/proactivity/candidates.py`) builds
`Candidate.reason` from a fixed lexicon of surface forms it owns — clock times,
elapsed hours, matched stems — specifically so a forwarded message cannot
steer an unprompted line into the model's mouth. Type E (association) cannot
work that way: its entire subject is a specific old memory, and a reason built
only from elapsed days (`"90일 전에 무언가 있었다"`) is the same contentless shape
`silence` already produces, which the judge already declines for having
nothing to say about. Withholding the memory's own words would not make the
generator safer, it would make it pointless.

The exception is bounded: `origin` is a column a model cannot write, so
`association_candidates` quotes `RecalledItem.content` — up to
`ASSOCIATION_QUOTE_CHARS` (200) characters — only for items where
`origin == "owner"`. An item recalled with any other origin is dropped before
the loop that builds `reason` ever sees its content. Pinned:
`tests/test_candidates.py::test_an_old_owner_memory_becomes_a_candidate`,
`test_a_memory_the_owner_did_not_write_is_refused`. The rule and its exception
are both stated in `daemon/proactivity/candidates.py`'s own module docstring, which is where a
future reader will meet them first.

**Correction (whole-branch review, finding 2): two claims above did not hold.**
`origin` is **not** "populated only from `authored_by_sender=True` turns" —
`daemon/memory/reindex.py` also populates it, by inferring `owner` from
`role == 'user'` whenever it rebuilds the mirror from markdown, because the
markdown carries no provenance at all (non-negotiable 3). A forward is
`role='user'` exactly like the owner's own words, so a rebuild can hand it
`origin='owner'` it never earned — and "bounded the same way non-negotiable 10
is" was also wrong: that rule reads a *live* turn's
`InboundMessage.authored_by_sender` (`daemon/tools/policy.py`), decided when
the message arrived, not this column read back later by recall. Fixed by having
`MemoryRecall.associate` additionally exclude `messages.reindexed` rows
(`daemon/memory/recall.py`), so a reindex-fabricated `'owner'` cannot reach this
check. Pinned: `tests/test_recall.py::test_a_reindexed_row_is_excluded_from_associate`.

### 2. The headphone probe was designed, approved, built, and then deleted by one measurement

`Reading.headphones` has a field and no probe. There was a probe: read the
default output device's `Transport:` line from `system_profiler
SPAudioDataType` and treat `usb`/`bluetooth`/`headphone` as point-to-point.
Running it for real found the default output on the development machine is
*always* `MacBook Pro Speakers (eqMac)` — a virtual driver in front of the real
hardware — and it reports `Transport: USB`, indistinguishable from real USB
headphones, for the laptop's own built-in speakers, regardless of what was
actually plugged in:

```
MacBook Pro Speakers (eqMac)  -> USB       (the default output, always)
MacBook Pro Speakers          -> Built-in
LG FULL HD                    -> HDMI
Microsoft Teams Audio         -> Virtual
eqMac Export                  -> USB
```

Not a miscalibration to tune: the transport field cannot see past a virtual
device to the real hardware behind it, on this machine, ever. That matters more
here than a wrong reading usually would, because `headphones` is the *one*
presence signal in the whole `Reading` that only ever *widens* what the
speaker may do — every other field can only take the speaker away. A probe that
answers wrong in the widening direction is a false "headphones", which is
exactly the meeting-accident PLAN §6.4 exists to prevent: a room full of people
hearing the laptop announce something because the gate believed nobody but the
owner could hear it. That is worse than never answering. Removed entirely
(`daemon/proactivity/presence.py`, `daemon/proactivity/base.py` unchanged
otherwise) rather than left disabled, so a later reader does not find dead code
that looks load-bearing. It was also the single most expensive probe in the
file (0.21–0.30 s). `Reading.headphones` stays `None` forever until a mechanism
exists that can see past a virtual device — a `Manufacturer` field or a
denylist of known virtual drivers, and a separate decision. Pinned:
`tests/test_presence.py::test_headphones_has_no_probe_and_stays_unknown`,
`tests/test_gate.py::test_headphones_excuse_only_the_foreground_app`,
`test_headphones_unknown_does_not_excuse_the_foreground_app`.

### 3. PLAN §6.4, revisited

§6.4 already records one reversal: `audio_busy = True` used to block the
utterance outright, on the reasoning that holding the device meant a call and
waiting cost nothing. Measurement disagreed — the signal fired on a
system-wide EQ all day — and the section demoted it to costing only the
speaker, the same treatment as an unreadable probe or a meeting app in front.

Splitting the signal reopens that question, because the premise it was decided
under no longer holds: the noise that justified demoting `audio_busy` all
lived on the *output* side (an EQ, a notification chime, an autoplaying
video), and `mic_busy` is a different, cleaner signal now that it is not
carrying that noise and not carrying our own wake hold either. So the question
was asked again on purpose rather than left on autopilot, and the answer was
kept: `mic_busy` still costs only the speaker, never the utterance
(`Gate._route`, not `Gate.judge`'s blocking rules).

Two reasons, not one. First, the asymmetry PLAN §6.4 states is about
*channels*, not about which probe fired: an ignored Telegram message costs
nothing whether the reason it was sent to Telegram is a meeting app, a locked
screen, or a mic in use — a voice out of the speaker during any of those is
the accident, and blocking the utterance instead of rerouting it buys nothing
that rerouting does not already buy. Second, and specific to this signal:
blocking would destroy the 👎 label the brake (§6, decision 4 below) depends
on. `Gate._route` never returns `local_speaker` alone — it is always `both`
or `telegram` — precisely so the label buttons are always attached to
something. A candidate that gets blocked outright produces no row and no
buttons at all, so a call in progress would silently spend the day's candidate
without ever giving the owner a way to say "not now" about that kind. Costing
only the speaker keeps the message, the buttons, and the brake's only input
alive.

### 4. One voice switch, and the offline preset's speaker

`DAEMON_PROACTIVE_SPEAKER_ENABLED` used to be a second top-level switch beside
`DAEMON_VOICE_ENABLED`, split off originally because the two failure costs are
not comparable (an ignored notification vs. a voice in a meeting). But
`Gate._route` already carries that asymmetry on its own — seven rules that
only ever downgrade to Telegram — so the second switch bought nothing except
"voice on" meaning two different things depending which file was read. Merged
into one: `voice_enabled` now governs both whether a hosted voice session may
open and whether a proactive line may reach the local speaker.

The merge broke something PLAN §7 promises by name.
`Settings(preset="offline", voice_enabled=True)` started raising, because
`_check` required a routed voice task and a model id for *any* session the
moment `voice_enabled` was true — correct for a hosted session, wrong for
`/usr/bin/say`, which needs neither. A `ConfigError` at load time does not
degrade a feature; it takes the whole daemon down. And it made §7's own
sentence — *"PC 앞에 있을 때의 선제 발화는 로컬 스피커로 나가므로 어떤 경로도 타지
않는다"* — unreachable on the one preset, `offline`, that the sentence is
about: the `offline` preset deliberately omits `CHAT_VOICE` so that promise is
literally true, and the merged check then refused to load before the local
speaker ever got a chance to prove it.

Resolved by splitting what `voice_enabled` implies from when it is checked,
not by splitting the switch back apart. `Settings.voice_session_problems()`
holds every clause that is actually about a *hosted session* — a routed voice
task, a model id, a key — and it is applied in two places for two different
reasons: at load time when `wake_enabled` is on (a wake gate that can never
open a session is a misconfiguration worth refusing early), and at session
start inside `run_voice` (`daemon/app.py:1091`) for every other path, which is
where a session actually needs any of it. `voice_enabled` alone no longer
carries those requirements, so it can be true under `offline` without the
process refusing to start. Verified: `Settings(preset="offline",
voice_enabled=True)` now loads; `tests/test_config.py::test_offline_preset_refuses_voice`
still holds for what it is actually testing (no hosted session), and
`test_an_offline_install_may_speak_out_of_its_own_speaker` pins the new
behaviour directly.

### 5. macOS Focus was rejected as the brake

The C rhythm (6–10 utterances a day, PLAN §6.2) needs something that can say
"stop today" faster than a config edit, and the label buttons are already
under every utterance because `Gate._route` never sends `local_speaker` alone.
macOS Focus was the other candidate — reading the active Focus mode would let
"Do Not Disturb" or a custom Focus double as a live, OS-native version of the
same brake. Rejected: reading it means `~/Library/DoNotDisturb/`, which needs
Full Disk Access — measured on this machine, 2026-08-11 — a grant that changes
the machine's security settings and is not something this project prompts for
outside a onboarding flow that already asks for narrower things (Automation,
microphone). The 👎 button was promoted instead: no new permission, already
delivered with every utterance, and already the input the label-driven tuning
loop (§8.3) is built to read. `KIND_REST_HOURS` (6), `KIND_REST_REPEAT_HOURS`
(24) and `DAY_STOP_LABELS` (3) in `daemon/proactivity/gate.py` are the brake.
Pinned: `tests/test_gate.py::test_one_thumbs_down_rests_that_kind_for_six_hours`,
`test_two_in_a_day_rests_that_kind_for_twenty_four_hours`,
`test_a_single_thumbs_down_per_kind_does_not_trigger_the_repeat_rule`,
`test_a_thumbs_down_outside_every_window_does_not_block`.

## Consequences

The routing table's presence-driven rules went from three (an unreadable
probe, a meeting app in front, "audio busy") to seven: presence unknown, away,
screen locked, output muted, microphone busy, output busy, and a meeting app
in front without headphones. Every one of them still only loses the speaker,
never the candidate — §6.4's asymmetry survived the split unchanged, which is
the point of re-deciding it deliberately instead of by default.

One frozen file changed, and a second one did not have to: `daemon/proactivity/base.py`
lost `Reading.audio_busy` for `mic_busy` + `output_busy`, while
`daemon/memory/schema.sql` needed no change at all, because both new fields ride
in the same JSON `gate_snapshot` column `audio_busy` already used. A new module,
`daemon/mic_hold.py`, exists purely so `daemon/proactivity/presence.py` can
subtract a state the voice layer owns without importing it.

`daemon/proactivity/candidates.py` now carries the one exception to "no user
text in `Candidate.reason`", stated and bounded in its own docstring rather
than in `docs/CONTRACTS.md`'s non-negotiable 7 — that non-negotiable is about
model-call counting and stays true unchanged; the reason-text rule it is an
exception to was never written into CONTRACTS.md in the first place, only
into the module that has to obey it.

One thing this ADR could not fix without touching a frozen comment: `Reading`'s
own docstring (`daemon/proactivity/base.py:31`) says "See docs/adr/0010" —
written when 0010 was the next free number. A later merge from origin/main
(commit `1662b9e`) landed two upstream ADRs, `0010` and `0011`, about
supersession-by-id, before this one was written down, so this decision is
`0012` and that in-code cross-reference now points at the wrong file. Flagged
rather than edited, per this milestone's rule that frozen files are not
touched to make prose agree with them.

## What would change our mind

If a mechanism is found that can name a real audio device's transport past a
virtual driver — the `Manufacturer` field, or a denylist of known software
outputs like eqMac — `headphones` could get a probe back, and the gate's one
excuse-only rule would start firing for real instead of staying permanently
`None`. And if Full Disk Access ever becomes something this project already
asks for elsewhere, reading Focus directly would stop costing a permission
this decision was unwilling to spend on its own.
