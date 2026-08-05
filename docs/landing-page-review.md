# Landing page — review, fixes, and the image work that is actually left

## How it ships

`site/` holds the page and nothing else: **everything in that directory gets
published.** Working files, explorations and specimens go in `docs/design/`.

`.github/workflows/pages.yml` uploads `site/` to GitHub Pages on any push to
`main` that touches it. It deliberately does not point Pages at `docs/`, whose
markdown is written to be read on GitHub rather than served.

**One step cannot live in the repo:** Settings → Pages → Source → *GitHub
Actions*. Until that is set, the workflow fails at the deploy step with "Pages
site not found". After that the page is at `https://k2d2h4.github.io/Daemon/`.

Two absolute URLs are coupled to that address — `og:url` and `og:image`. A custom
domain later means editing both, or the share card silently breaks:

```bash
grep -n "og:url\|og:image" site/index.html
```

There are no per-PR preview deploys. GitHub Pages has one environment; previews
need Cloudflare Pages or Vercel, which means connecting the repo from their
dashboard — not something the repo can configure on its own.

`og.png` (1200×630) is referenced but does not exist yet; until it does the share
card renders text-only. Prompt and post-processing are at the end of this file.

Reviewed `site/index.html` as produced by Claude Design against
`docs/landing-page-prompt.md` and the identity handoff. Verified by running it:
served over `python3 -m http.server`, driven in a real browser, every widget
clicked, measured at 360 / 820 / 1440.

## What was already right

Worth stating, because it is most of the page:

- **All 12 sprite instances are the handoff SVGs byte-for-byte.** Verified by
  md5 of each inline body against `assets/*.svg`. No redraw, no trace.
- **The expression mapping is honest** — cards use `working` / `speaking`, the
  state switcher maps idle/speaking/working/asleep/grumpy correctly, footer is
  `asleep`. Only the two bobbing states bob.
- No `border-radius`, no `blur()`, no `backdrop-filter`, every `box-shadow` has
  a zero blur radius, every transition is exactly 120ms.
- No `fetch`, no `localStorage`, no cookies, no `Math.random`.
- The badge/hover/disabled colours that look invented (`#10221C`, `#C9B8FF`,
  `#5A5470`, …) are all specified in the handoff README. Not deviations.
- Copy is verbatim from the deck, and there is no fabricated social proof.

## Fixed

### Design-system violations

| | Was | Now |
|---|---|---|
| Scroll reveals | `.rv` fading up 340ms eased, staggered `d1/d2/d3` | removed, CSS + markup + IntersectionObserver |
| Log rows | fading in 70ms apart on scroll | appear with the page |

The brief's own done-check ("grep your own output for `border-radius`,
`linear-gradient`, `blur(`… each hit is a bug") passes on everything **except the
hero**, which the user has chosen to keep as designed — see below.

### The hero: reverted to the original at the user's request

These were removed, reviewed, and then **restored verbatim** because the user
preferred the original treatment. Recorded here so the state of the page is not
a mystery to the next reader:

| Element | What it is | What it costs |
|---|---|---|
| `.hero-bg` | 56px grid from two `repeating-linear-gradient`s at `opacity:.3` | fails the brief's `linear-gradient` grep, though the visual result is hard-edged |
| `.hero-vig` | `radial-gradient` vignette | a real visible gradient; the brief forbids a wash behind the hero |
| `.motes` | 9 squares rising on 19–31s loops | "floating particles" is on the forbidden list |
| `.chip a–d` | 4 squares drifting on 5.5–8s `ease-in-out` loops | eased decorative motion |
| `.hero-plinth` | grounding bar at `opacity:.5` | invented element, fractional opacity |
| Sprite size | `clamp(180px, 24vw, 336px)`, `160px` mobile | fluid sizing lands between multiples of 28 at mid widths; 160 is not a multiple of 28 |
| Entrance | headline typed line by line, then `.fadeup` at 2.3/2.45/2.6s | **the primary CTA does not exist for the first 2.45 seconds** |
| `h1` size | `clamp(30px, 3.6vw, 46px)` | display size is specified as 40px |
| Height | `min-height: calc(100svh - 64px)` | ~300px of dead space above and below the copy at 900px tall |

Two of these are worth separating from taste, because they are not aesthetic
tradeoffs:

- **The 2.45s CTA delay.** Every visitor waits that long before "GET STARTED"
  exists. Removing the three `.fadeup` classes in the hero markup keeps the
  typewriter and fixes this.
- **Fluid sprite sizing.** At 1400px+ it happens to land on 336 (28×12), so it
  looks correct; at ~1100px it renders at 264, which is not a multiple of 28 and
  puts the sprite off its own pixel grid. This is the one item here that
  degrades the sprite itself rather than the composition.

An intermediate version was tried and rejected: the grid rebuilt as an inline
SVG rect pattern (hard edges, no gradient, flat `#1A1624` — the exact colour
`--line` at 30% resolves to) and the squares snapping cell-to-cell on `step-end`
instead of drifting. It satisfied the system on paper. The user preferred the
original, which is their call to make; it is recorded here only so nobody
re-derives it from scratch.

### Factual errors

**"It considered speaking 288 times and did it once"** was not in the copy deck,
and the log above it showed 10 rows. Removed.

**The log contradicted the gate's own rules**, which matters because the caption
invites the reader to reproduce it:

- `11:20 pattern_time BLOCKED — Zoom in foreground, audio in use`. A foreground
  app and a busy audio device are **routing** conditions in the spec, not blocks
  — they send the message to Telegram instead of the speaker. This rendered the
  gate as stricter than it is and confused blocking with routing.
- `09:15 cooldown, last spoke 41m ago` — nothing had spoken yet that day, and
  anything 41 minutes earlier would have been inside quiet hours.
- `17:00 cooldown, last spoke 75m ago` — the only message was at 13:45, 195
  minutes earlier.
- `nothing worth saying` was rendered as a red `BLOCKED`, but no such gate rule
  exists. That outcome comes from the single LLM call that runs *after* the gate
  opens.

Rebuilt the day so the arithmetic holds and both mechanisms are visible:
`BLOCKED` in `error` is the deterministic gate, `SILENT` in `text-muted` is the
model declining once the gate has opened. Every interval now checks out against
the stated 90-minute cooldown, budget of 3, and `open_loop` cap of 1. The
caption claims "the same rules against your own machine's state" rather than
"the same verdicts", which was never true of a canned log.

### Accessibility

- Three long passages were `text-faint #7A7192` on `canvas`, which measures
  **4.16:1** — under AA, and the brief restricts that token to captions. The
  install explainer, the differentiator framing line and the gate explainer are
  body copy by length; moved to `text-muted #8E85A3` (**5.45:1**). The two real
  one-line captions (EARLY, MIT) keep `text-faint`.
- "Full plan and milestones" was a link distinguished by colour alone. Given the
  `prose` underline treatment.
- The tab list had no arrow-key navigation despite `role="tablist"`. Added
  Arrow/Home/End with roving `tabindex`.
- **At 360px the header nav overflowed the viewport by 6px**, putting the entire
  page into horizontal scroll. Fixed; nav now ends exactly on the 24px gutter.

### Weight

12 copies of the same 6 sprites were inlined — 784 `<rect>`s each. Moved to one
`<defs>`/`<symbol>` sheet with `<use>` references: **140KB → 76KB**, and the
file is now editable by a human.

### Additions

- `favicon.svg`, `theme-color`, and Open Graph / Twitter card tags. There were
  none, on a page whose stated job is receiving traffic from Hacker News.
- The favicon is generated **from `daemon-idle.svg` itself** plus a `#120F18`
  backing rect — not drawn, not generated.

## Deviations I am making knowingly

Per the brief's own rule, stating them rather than leaving them silent.

1. **The whole hero decoration layer**, per the table above. The user reviewed
   the compliant version and preferred the original.
2. **The hero sprite keeps its blink** — two layered sprites where the `idle`
   layer drops to `opacity:0` for 140ms on a `step-end` loop, revealing `asleep`
   underneath. This is one animation beyond the specified bob. It uses real
   sprites, snaps rather than fades, and stops under `prefers-reduced-motion`.
   Kept because it is the only thing on the page that makes the mascot feel
   resident. Delete `.blinkA` to remove it.
3. **Section 06 is "Works today / Not yet", not the M1a/M1b/M2/M3/M4 table.**
   That was Claude Design's call and it is better landing-page copy than
   milestone codes. Consequence: **the stepped progress component now appears
   nowhere on the page.**

## The one substantial gap: the gate widget

`docs/landing-page-prompt.md` specifies **"Ask the gate"** as the centrepiece —
nine controls, rules evaluated top-to-bottom in JS, defaults that produce a
block, so the visitor discovers for themselves that the loop mostly chooses
silence. **It was not built.** What shipped is the static log, which the reader
reads rather than discovers.

This is a feature build (~80 lines of markup and JS) that reshapes section 03,
not a fix, so I have left it out pending a decision. The log I rebuilt is now
arithmetically sound and stands on its own if you would rather skip the widget.

---

# The image question

You asked which detail elements to replace with GPT-generated images. Answering
honestly first, because most of the candidates are traps.

## Where generated raster art will fail here

- `gpt-image-1` asked for pixel art returns **near-pixel art**: cell boundaries
  are anti-aliased and cell size drifts across the image. It will not land on a
  28-grid.
- It does not honour exact hexes. Give it `#A78BFA` and you get a dozen nearby
  purples. On a page whose brief calls silent deviation the one unacceptable
  outcome, every generated image commits that violation automatically.
- It renders text badly. Anything with the wordmark in it should have the type
  composited afterwards, not generated.

So: **never** regenerate the mascot or add expressions (the brief forbids it and
a lookalike is a failed deliverable), never generate icons (this system has no
icon set on purpose), never generate the terminal frame (6 lines of CSS, sharper
and responsive), never generate a scanline or noise texture (forbidden), and
never generate the favicon (derived from the real sprite, already done).

## Where it genuinely earns its place

Only scenes — things that cannot be built from squares and type.

**1. `og.png` — the actual gap.** 1200×630. The meta tag is already in the head
pointing at it; until the file exists the share card renders text-only.

> Pixel art illustration, 1200×630, dark scene. A desk at 3am seen straight on:
> a boxy CRT-style monitor glowing faintly, a mechanical keyboard, a cold cup, a
> window behind showing a black sky. A small purple pixel creature sits on the
> desk beside the keyboard, awake, facing the screen. Restricted palette only:
> #120F18 background, #1B1626 and #241D33 mid-tones, #2B1B3D outlines, #A78BFA
> and #8B5CF6 and #6D3FD4 for the creature and the screen glow, #ECE7F5 for
> highlights. Hard-edged pixels on a consistent square grid, no anti-aliasing,
> no gradients, no glow bloom, no blur, no drop shadows, no text, no lettering,
> no watermark. Flat retro game aesthetic, 1990s Game Boy manual illustration,
> not painterly. Leave the left third visually quiet.

Composite `DAEMON` and the pitch line over that quiet left third yourself, in
Silkscreen — do not ask the model for type.

**2–4. Optional band vignettes.** Pick **at most three**, one per band, or the
page turns into an illustrated brochure. Same preamble as above (palette, hard
pixels, no text) plus:

- *Quiet hours*, for section 03: "A dark bedroom at night, one window, a phone
  face-down on a nightstand, its screen dark. The small purple pixel creature
  sits beside it, awake, not touching it. Nothing is glowing except starlight."
- *File ownership*, for section 05: "Two open notebook pages side by side on a
  dark surface. The left page has handwriting in warm off-white; the right page
  has neat typed lines in purple. A thin hard vertical line separates them. No
  legible words — texture of writing only."
- *One process*, for section 05: "A single small dark box on an empty floor,
  one purple light on its front, a cable running off-frame. Nothing else in the
  room. Emphasise the emptiness around it."

## The post-processing step that makes them conform

Non-negotiable if these are going next to the sprites. Generate at 1024, then
force the grid and the palette:

```bash
magick og-raw.png -filter point -resize 25% -filter point -resize 400% \
  -remap palette.png -colors 12 og.png
```

Where `palette.png` is a strip of exactly the tokens above. The two `-filter
point` resizes snap the cells to a real grid by destroying the anti-aliasing;
`-remap` forces the palette. Inspect at 400% before shipping — if edges are
still soft, the first resize was not aggressive enough.

Store the result as `site/og.png` and serve it same-origin. That is not a
third-party fetch, so it does not contradict the "no network calls" rule; only
third-party assets and phone-home behaviour would.
