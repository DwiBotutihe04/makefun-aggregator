# free-model-aggregator-01 — entry server

A dual-wire (OpenAI + Anthropic) aggregator over 10 simulated free-tier
providers. No dependencies, pure Node 24 (type stripping is native;
the harness runs it the same way, adding `--experimental-strip-types`
only on older Node).

## Run locally

```sh
MAKEFUN_PROMPT_FILE=<trial>/input/prompt.txt MAKEFUN_ENV_URL=<env origin> node main.ts
```

The brief is the trial's `input/prompt.txt` (prose plus embedded JSON
documents: providers, ratings, workload). The env is the kit's simulated
provider server; the harness starts it during `grade-local`, which is
the supported way to run the server end to end:

```sh
cd <kit> && node cli/makefun.mjs grade-local --kit . --bot <this dir> \
  --driver process --instances 3
```

`node main.ts --list` prints the catalog as one JSON document and exits
without starting the server. The catalog is built by probing each
provider's models through the env, so outside a trial (no env) it prints
`{"providers":[]}` — the same as the kit starter.

Endpoints: `POST /v1/chat/completions`, `POST /v1/messages`,
`POST /v1/models`, `GET /v1/models`, `GET /v1/providers`,
`GET /v1/providers/models`, `GET /healthz`. `--list` prints the catalog
(the server object) as one JSON line.

## Routing

- **Quality first, budget-aware.** Each request gets a candidate list of
  (provider, model) pairs sorted by the brief's rating for its
  requirements, deduplicated per provider (a provider's models share its
  call budget, so only its best model per request is ever tried).
- **Rating:** exact table for `(tools, json_mode, vision, min_context,
  tool_choice)`; `json_mode` only ever goes to OpenAI-wire providers
  (Anthropic-wire echoes `json_mode: false`); the two `none` rows
  (1475 openrouter `z-ai/glm-5.3-flash:free`, 1470) are the default for
  plain text; vision routes to the best vision model (gemini-3.8-flash
  1458, then glm-4.6v-flash 1432); `min_context` 8k/32k picks the cheapest
  model that still fits.
- **Failover:** sequential retries, one attempt per provider per pass.
  After 2 consecutive failures a provider is marked exhausted for the
  rest of the trial. A provider that is busy with a concurrent request
  is skipped for that request (it may be retried later in the same
  request). Budget exhaustion is learned from 429s, never assumed.
- **Concurrency:** the brief's `max_parallel` (1 for every provider in
  the practice brief) is honoured via a per-provider semaphore.

## Translation (`lib/translate.ts`)

Every wire pair is translated losslessly: system text, user text, tool
definitions (names/order), `max_tokens`, stops, image bytes, JSON mode,
and tool results (a model's own tool-call ids are rewritten to
`call-<n>` so providers that echo foreign ids still validate).
OpenAI streams get `stream_options.include_usage: true` (the client
sends none); the usage the provider echoes back is translated, never
recomputed. Anthropic messages whose `content` is a plain string (not
an array of blocks) are handled.

## Practice scores (kit scorer, `grade-local --driver process`)

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 31 |
| mean | **99** | 99 | 100 | 100 | 100 | 100 | 29.3 |

All 3 replays matched byte-for-byte. Quality 99 is the ceiling: the
bound assumes every request gets its best model, but the best model
(openrouter `z-ai/glm-5.3-flash:free`, 1475) has a call budget of ~10–12
for ~17 requests that want it, so a few requests must take 1470/1465
substitutes. The remaining upstream calls are the env's scheduled
429s (1–2 per provider at fixed call numbers) plus budget-exhaustion
retries — unavoidable without knowing the sealed fail tables.

## Files

- `main.ts` — brief parsing (`MAKEFUN_PROMPT_FILE`), probing, candidate
  ranking, failover, HTTP server, catalog (`--list`).
- `lib/providers.ts` — shared types and the brief/catalog schemas.
- `lib/wire.ts` — OpenAI ↔ Anthropic translation (requests, responses,
  SSE streams, model lists, error shapes).
- `lib/translate.ts` — OpenAI/Gemini ↔ Anthropic translation (Gemini
  `generateContent`/`streamGenerateContent` forms, JSON-mode mapping).
- `LICENSE` — MIT (from the kit starter).
