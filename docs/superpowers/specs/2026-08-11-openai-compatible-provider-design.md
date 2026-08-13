# OpenAI-compatible hosted provider

**Date:** 2026-08-11
**Status:** design approved, not yet implemented
**Branch:** `claude/openai-compatible-provider`

## Problem

Daemon's LLM gateway is provider-agnostic, but its *config surface* is not.
`HOSTED_PROVIDERS` names exactly three vendors — Anthropic, OpenAI, Gemini — and
`Settings` rejects anything else at startup. Someone who wants Qwen, Kimi,
DeepSeek or their own vLLM box as the main model has no way to say so, and
onboarding never offers the choice.

Local open-weight models already work through Ollama (the default local chat
model is `qwen3:14b`), so the gap is specifically **hosted, key-authenticated,
non-big-three models**.

Two things block the obvious workaround of "just point a base URL somewhere
else":

1. No hosted provider has a base-URL setting. `daemon/llm/providers/openai.py`
   holds `API_URL` as a module constant, and the only configurable endpoint is
   `DAEMON_OLLAMA_BASE_URL`.
2. Even with one, the wire format is wrong. `openai.py` is written against the
   **Responses** API (`/v1/responses`, top-level `instructions`, `input` items).
   Every compatible vendor speaks **Chat Completions** (`/v1/chat/completions`,
   `messages` array). Swapping the base URL alone yields HTTP 400.

`DAEMON_OLLAMA_BASE_URL` is not a back door either: `OllamaProvider` calls
Ollama's native `/api/chat` and sends no `Authorization` header.

## Decisions

Settled during design. Each was a real fork; the rejected side is recorded so a
later reader does not re-litigate it silently.

| # | Decision | Rejected alternative |
|---|---|---|
| 1 | **One compatible endpoint at a time.** A single set of `.env` values. | Several simultaneous endpoints (indexed keys like `DAEMON_COMPAT_1_*`), so chat could go to Qwen while reflection goes to DeepSeek. Rejected: the presets resolve exactly one `HOSTED` provider, so this would reshape the preset model too, for a need nobody has stated yet. Reachable later by extending key names. |
| 2 | **All five vendor presets ship:** Qwen, Kimi, DeepSeek, OpenRouter, custom URL. | A narrower list. |
| 3 | **Two-step onboarding question.** The provider menu grows a fourth entry, `other`; choosing it opens a vendor sub-menu. | A flat eight-item top-level menu. Rejected: it would show `qwen` as a peer of `anthropic` while `.env`, `daemon doctor` and the gateway log all say `openai_compatible` — the choice displayed would not be the choice stored, which is the exact failure `DEFAULT_HOSTED_PROVIDER` was emptied to prevent. |
| 4 | **Key check probes `GET /models`, then falls back to a 1-token chat call.** | (a) Treating a missing `/models` as a failed key — too strict, `/models` is optional in the compatible spec. (b) No verification — contrary to the wizard's whole purpose. |
| 5 | **A separate provider module.** `openai.py` is not touched. | (a) Extracting a shared HTTP base class for both. (b) Rewriting `openai.py` to Chat Completions so one provider serves GPT and Qwen alike. Both rejected — see below. |

### Why decision 5 rejected the merge

Option (b) is the tempting one: OpenAI's own API also serves Chat Completions,
so one provider with a configurable base URL could cover GPT *and* every
compatible vendor, with fewer files than shipping a second module.

It was rejected for three reasons, in order of weight:

1. **It destroys provider identity.** A user running Qwen would have
   `DAEMON_HOSTED_PROVIDER=openai` in `.env`, and `daemon doctor` and every
   `llm.call` log line would read `provider=openai`. This is decision 3's
   problem relocated from the menu into the configuration and the logs, where it
   is harder to notice and harder to undo.
2. **It rewrites a working, live-verified path for zero user benefit.** Existing
   GPT users gain nothing and absorb all the regression risk.
3. **It pins GPT to the older surface.** OpenAI ships new capabilities to
   Responses first; binding GPT to Chat Completions to accommodate Qwen trades
   the GPT path's future for an unrelated feature.

Option (a) — a shared base class — buys back roughly 40 lines of duplicated HTTP
scaffolding at the cost of coupling two vendors' APIs in one type. This
repository has repeatedly chosen the opposite trade (`MODELS_URL`,
`SENSITIVITIES` and the wake defaults are all deliberately duplicated with a
comment explaining the copy), and the same reasoning applies.

**Consequence: `daemon/llm/providers/openai.py` is not modified by this work.**
Any diff touching it is out of scope.

## Non-goals

- **Voice.** `Task.CHAT_VOICE` stays pinned to Gemini. Native audio is the only
  voice implementation, and a compatible text endpoint cannot serve it. A user
  who picks a compatible provider *and* enables voice is still asked for a
  Gemini key, exactly as an Anthropic or OpenAI user is today.
- **Embeddings.** `Task.EMBED` stays on Ollama in every preset.
- **Replacing the Ollama path.** Local models keep their own provider.
- **Per-task routing across several compatible vendors.** Decision 1.

## Config surface

### Names

The provider name is **`openai_compatible`**. The rest is not free choice —
`Settings.provider_model` resolves `getattr(self, f"{provider}_model")` and its
error text is built as `DAEMON_{provider.upper()}_MODEL`, so the name determines
the field and env key. A mismatch would produce an error message naming a
variable that does not exist.

| Thing | Value |
|---|---|
| Module | `daemon/llm/providers/openai_compatible.py` |
| `PROVIDER_KEY_ENV` entry | `OPENAI_COMPATIBLE_API_KEY` |
| Model | `DAEMON_OPENAI_COMPATIBLE_MODEL` → `Settings.openai_compatible_model` |
| Endpoint | `DAEMON_OPENAI_COMPATIBLE_BASE_URL` → `Settings.openai_compatible_base_url` |

The module path is load-bearing: `tests/test_reachable.py` looks for
`daemon/llm/providers/{provider}.py` for every name in `PROVIDER_KEY_ENV` and
fails if it is absent.

`HOSTED_PROVIDERS` becomes `("anthropic", "openai", "gemini",
"openai_compatible")`. That single edit is what makes the admin dropdown
(`settings_io.py` publishes `list(HOSTED_PROVIDERS)`) and the onboarding
requirement calculation (`setup.py:needs_for`) pick the new provider up.

### Defaults and validation

Both new settings default to `""`. An empty value is a configuration to fix, not
a value to guess — the same judgement `DAEMON_GEMINI_LIVE_MODEL` records, and
for the same reason: a guessed endpoint fails at the first conversation instead
of at startup.

`Settings._provider_problems` gains one check alongside the existing "key is
empty" and "model is empty" ones: when the provider is `openai_compatible` and
`openai_compatible_base_url` is empty, that is a problem.

A field validator on the base URL enforces two things:

- It must be an absolute `http://` or `https://` URL. Trailing slashes are
  stripped.
- **It must not end in `/chat/completions`.** Vendor docs show the full endpoint
  URL, and pasting it whole is the predictable mistake; left alone, the provider
  would append the path a second time and the resulting 404 would explain
  nothing. The validator rejects rather than silently repairing, and the error
  message carries the corrected value:

  ```
  DAEMON_OPENAI_COMPATIBLE_BASE_URL must not include /chat/completions —
  use https://api.deepseek.com/v1
  ```

  Rejecting over repairing follows this module's existing posture (`loud beats
  degraded`): a value the user can read back in `.env` is worth one extra
  correction.

### The vendor is not stored

`.env` records the base URL and nothing else. There is no
`DAEMON_OPENAI_COMPATIBLE_VENDOR`. Two stored values that can disagree — a user
hand-edits the URL to Kimi while the vendor field still says `qwen` — leave no
way to tell which is the truth.

Display names are recovered by reverse lookup instead. `setup.py` holds a table
beside `HOSTED_CHOICES`:

| key | label | base URL | default model |
|---|---|---|---|
| `qwen` | Qwen (Alibaba Model Studio) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| `kimi` | Kimi (Moonshot) | `https://api.moonshot.ai/v1` | `kimi-k2.5` |
| `deepseek` | DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| `openrouter` | OpenRouter | `https://openrouter.ai/api/v1` | *(none — see below)* |

The fifth choice, custom URL, has no row: having nothing to prefill is what
defines it.

The table serves two purposes — prefilling answers during onboarding, and
mapping a stored URL back to a human label for `daemon doctor` and the admin
page. A URL not in the table displays as itself.

Qwen's row uses the **International (Singapore)** endpoint deliberately: the
new-account free quota is only granted there.

OpenRouter ships no default model because its catalogue rotates. Free model ids
carry a `:free` suffix and change without notice, so a hardcoded default would
go stale; the onboarding model list answers this question instead.

## Provider implementation

`OpenAICompatibleProvider` implements the `Provider` protocol in
`daemon/llm/base.py`. No neutral type changes — `Message` already carries
`role="tool"` and `tool_call_id`, which is Chat Completions' own shape, and
`decode_tool_arguments` already handles arguments arriving as a JSON string.
`provider_signature` stays `None` (Gemini-only).

Constructor takes `api_key` and `base_url`; either missing raises
`ProviderError`.

### Mapping, relative to `openai.py`

| | Responses (`openai.py`) | Chat Completions (this module) |
|---|---|---|
| System turn | hoisted to top-level `instructions` | stays in `messages` as `role: system` |
| Output cap | `max_output_tokens` | `max_tokens` |
| Tool declaration | `type`/`name`/`parameters`, flat | nested under a `function` object |
| Tool request | `function_call` items in `output` | `choices[0].message.tool_calls` |
| Tool result | `function_call_output` item | `role: tool` + `tool_call_id` |
| Token counts | `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` |
| Images | `input_image` | `image_url` with a `data:` URI |
| Reply text | `output_text` parts of message items | `choices[0].message.content` |
| Stop reason | `status` / `incomplete_details.reason` | `choices[0].finish_reason` |

`health()` is `GET {base_url}/models`.

### Error handling

Identical posture to the other hosted providers, because the failure modes are
the same:

- Exactly one retry, and only for `429` and `5xx`. Providers do not build retry
  chains; the gateway decides about fallback.
- `4xx` other than 429 fails immediately — a bad key, an unavailable model or a
  malformed request will not improve on a second attempt.
- Every failure leaves as `ProviderError`, including a body that is simply not
  shaped like the documented one.
- The API key is redacted from any upstream text that reaches an exception
  message. This project has leaked a key through an exception chain twice, and
  `tests/test_providers.py` asserts against it for every hosted provider.
- A JSON body that decodes to something other than an object is rejected
  explicitly — proxies in front of these endpoints do answer with bare arrays.

### Assembly

`daemon/app.py:_build_providers` gains one branch constructing the provider from
`settings.openai_compatible_api_key` and `settings.openai_compatible_base_url`.
This is the only new import of a concrete provider, and it stays inside
`app.py`, per the layering rule.

## Onboarding

`daemon setup` gains one menu entry and one sub-question. Illustrative run:

```
Which provider should the hosted work go to?
  1) anthropic   Claude.
  2) openai      GPT.
  3) gemini      Gemini. The one that shares a key with voice.
  4) other       Qwen, Kimi, DeepSeek, OpenRouter, or your own server —
                 anything that speaks the OpenAI API.
> 4

Which one?
  1) Qwen (Alibaba Model Studio)
  2) Kimi (Moonshot)
  3) DeepSeek
  4) OpenRouter
  5) custom URL
> 1

Endpoint [https://dashscope-intl.aliyuncs.com/compatible-mode/v1]:
>

API key: ****
  ✓ key works

Model id [qwen-plus]:
  1) qwen-plus  2) qwen-max  3) qwen3-…
>
```

Mechanics:

- The vendor choice is asked by `_choose_hosted` only when the user picks
  `other`, and it writes `DAEMON_OPENAI_COMPATIBLE_BASE_URL` (and the model
  default) into the pending updates. It is never asked under the `offline`
  preset, which resolves no hosted task at all.
- `needs_for` emits the endpoint, key and model `Need`s when
  `openai_compatible` is among the required providers. The model `Need` is
  `listed=True`, so the wizard offers the ids the probe returned.
- Re-running `daemon setup` re-asks the model id with the current value as its
  default, matching how the other model ids already behave; the key drops out
  once set, because nobody wants to re-paste a working credential.

### Key verification

`check_openai_compatible(key, base_url, model)` returns the same `Verdict` shape
as the existing probes.

1. `GET {base_url}/models` with a bearer token. On `200`, the key is proven and
   the ids populate `Verdict.models` for the menu. On `401`/`403`, the key is
   rejected — that is a definitive answer, so no fallback runs.
2. On any other failure — `404`, `405`, a connection error, a non-JSON body —
   the endpoint is treated as one that does not implement `/models`, and the
   probe falls back to `POST {base_url}/chat/completions` with the configured
   model and `max_tokens: 1`. Success proves both the key and the exact path the
   daemon will use; failure reports the status and body.
3. When the fallback ran, `Verdict.models` is empty. The wizard already handles
   this case — `Wizard.catalog` prints `NO_LIST_NOTE` and takes free text — so
   no new behaviour is needed.

Step 2 is worth the extra call beyond proving the key: `/models` succeeding does
not imply `/chat/completions` will, and OpenRouter in particular refuses
unfunded requests for paid models with `402` while listing them happily.

**Large catalogues need no new mechanism.** OpenRouter returns several hundred
models, but the wizard already folds any list to `MODEL_LIST_FOLD` (6) entries
with `?` to expand, and `order_models` already drops ids that are overlong or
unprintable. This work adds nothing here; it only has to pass the ids through,
as the other probes do.

## Admin UI

`daemon/admin/settings_io.py`:

- `openai_compatible_model` and `openai_compatible_base_url` join `STR_FIELDS`.
- `openai_compatible_api_key` joins `SECRET_FIELDS`.
- The `hosted_provider` dropdown needs no change; it renders
  `list(HOSTED_PROVIDERS)`.

The page stays English-only, consistent with the rest of the admin console.

## Testing

### Automated (`python3 -m pytest`, no network, no keys)

`tests/test_providers.py`, using `httpx.MockTransport` like every other provider
test:

- Request shape: messages array, `max_tokens`, temperature, nested tool
  declarations.
- Response mapping: text, `tool_calls` with arguments arriving as a JSON string,
  `usage.prompt_tokens`/`completion_tokens` landing on `Completion`.
- A full tool round trip: assistant turn with `tool_calls` → `role: tool` result
  → final answer.
- Korean text through the round trip, per the suite's standing rule.
- The API key never appears in the exception chain, for every failure path.
- Exactly one retry on 429 and 5xx; none on other 4xx.
- A non-object JSON body raises `ProviderError`.

`tests/test_config.py`:

- Routing to `openai_compatible` with an empty base URL is a startup problem.
- A base URL ending in `/chat/completions` is rejected, and the message contains
  the corrected URL.
- Trailing slashes are stripped.

`tests/test_setup.py`:

- `needs_for` asks for endpoint, key and model when the provider is chosen, and
  for none of them under `offline`.
- The probe falls back to the chat call when `/models` answers 404, and does not
  fall back on 401.

`tests/test_reachable.py` needs no new `PENDING` entry: the module exists and
`app.py` constructs it, so the existing parametrised check passes on its own.

### Live verification

The unit suite may not touch the network, a key or a socket. "It actually works"
is therefore proven by a spike run by hand, in the shape of
`evals/m1c_text_tools_spike.py`:

`evals/openai_compatible_spike.py` — takes the configured endpoint from `.env`,
runs a plain turn, then a turn that forces a tool call, and prints the raw
request and response for both.

**Acceptance is end-to-end, not the spike alone.** The bar agreed for this work:

1. The spike passes against both verification endpoints.
2. `daemon run` starts with the compatible provider routed, and a real
   conversation over Telegram produces a reply.
3. That conversation includes at least one successful tool call, since tools are
   central to the product and the most likely thing a compatible endpoint gets
   wrong.

**Bars 2 and 3 were met by substitution, not as written.** What ran is
`evals/openai_compatible_loop_spike.py` — the real loop, recall, tool registry,
origin gate and audit table, on OpenRouter — rather than a Telegram conversation
on Qwen. The reasons and what therefore stays unproven are recorded in
`evals/CLAUDE.md`; read that before quoting this list as satisfied.

### Verification coverage, stated honestly

The table below is what the design *planned* to verify. What actually happened,
recorded here because this section demands the same honesty of everyone else:

| Vendor | Status after this work |
|---|---|
| OpenRouter | **verified** — free key, `:free` model with tool support (`openai/gpt-oss-20b:free`, 2026-08-11); both spikes pass, including a tool call through the assembled loop |
| Qwen (DashScope) | **not verified — never exercised.** The available key is China-region: `403 AccessDenied.Unpurchased` on `/compatible-mode/v1`, `401` on the International endpoint. No request against it ever succeeded |
| Kimi (Moonshot) | code-supported, **unverified** — API access requires a prepaid top-up |
| DeepSeek | code-supported, **unverified** |
| custom URL | code-supported, **partly exercised** — `daemon setup` was driven end to end through the `custom URL` answer against OpenRouter's address typed by hand (2026-08-13), which is how the wizard's own defect on that path was found. No self-hosted server has been run |

**The plan called for two complementary endpoints and got one.** OpenRouter
proves the Chat Completions wire format cheaply and repeatedly. DashScope was
supposed to prove the base-URL handling, because `/compatible-mode/v1` is the most
unusual path prefix of the five and the one most likely to break a naive URL join
— OpenRouter's ordinary `/api/v1` would let such a bug through. That is the
specific thing the DashScope run existed to prove and **it remains unproven**: no
live request has ever gone to a base URL with a path prefix deeper than `/api/v1`.
Unit tests cover the join (`tests/test_providers.py` uses the DashScope URL as its
fixture), which is not the same as a socket agreeing.

`evals/CLAUDE.md` states this correctly and always did; this table said
"verified" and was wrong.

An unverified vendor must not be described as supported without this
qualification in release notes or docs.

## Risks

- **Tool-calling quality varies by model.** The provider forwards tool schemas
  faithfully, but whether a given model uses them well is the model's property.
  Gemini Live already demonstrated a model declining tools under a crowded tool
  set. This is a model-selection concern, not a provider defect, and the spike
  will show it per endpoint.
- **OpenRouter's free tier rotates.** Free ids appear and disappear without
  notice and are rate-limited at 200 requests/day. Fine for verification,
  unsuitable as a documented default — hence no default model for that row.
- **Qwen's free quota is time-boxed** (1M tokens per model, 90 days) and
  region-locked to Singapore. After expiry an account without a payment method
  fails rather than bills.
