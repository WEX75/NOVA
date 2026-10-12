# NOVA

**By Tatomir Darius**

**An autonomous, local-first business assistant** that reads email, plans and critiques its own work across a multi-model pipeline, and executes tasks — while treating anything destructive, anything touching the OS, and anything arriving as untrusted input (email) as requiring explicit human sign-off.

The project brings together a state machine, model routing, permission system, secrets handling, systemd deployment, a wake-word voice pipeline, and a local dashboard — all on consumer hardware with no cloud infrastructure beyond two free-tier LLM APIs.

> See [`TECH_STACK.md`](TECH_STACK.md) for the full technology inventory.

---

## Why this project exists

Most "AI agent" demos either run once in a notebook or hand-wave the parts that make autonomy actually safe: what happens when the model is wrong, when it wants to delete something, when an email tells it to do something, or when the process gets SIGKILLed mid-task. NOVA is an attempt to take those questions seriously — the interesting engineering here is almost entirely in the boundaries, not the happy path.

## Architecture

```
                         ┌─────────────────────┐
   Outlook (Graph API)   │                      │   Voice (wake word →
   ───────────────────►  │   LangGraph state    │ ◄── local STT)
                         │      machine         │
   Dashboard chat        │                      │
   ───────────────────►  └──────────┬───────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
              ┌────────────┐  ┌────────────┐  ┌─────────────┐
              │  DIRECTOR  │  │   CRITIC   │  │  EXECUTOR   │
              │  Gemini    │─▶│    Groq    │─▶│  Qwen2.5-   │
              │ 2.5 Flash  │  │ Llama-3.3  │  │   Coder     │
              │  (cloud)   │  │70B (cloud) │  │  (local)    │
              └────────────┘  └────────────┘  └─────────────┘
                     │          1 review pass         │
                     │          — no debate loops      │
                     ▼                                 ▼
              privacy_sensitive tasks          file writes, OS-action
              (email, voice) bypass            launches, destructive ops
              cloud entirely, route                    │
              straight to Executor                     ▼
                                              ┌─────────────────────┐
                                              │     human_gate      │
                                              │  interrupt_before,  │
                                              │  genuinely blocks   │
                                              └─────────────────────┘
```

Every node speaks a single typed contract: it receives `NovaState`, returns a **partial state update** (never mutates state directly), and every hop is appended to an **append-only episodic log** — nothing is ever deleted, so any decision NOVA ever made can be reconstructed after the fact.

## What it actually does

- **Reads Outlook mail** via Microsoft Graph and turns it into tasks — but email content is treated as **data, never instructions**. An email that says "run this" becomes a proposed task requiring human approval; it never auto-executes. Covered by a dedicated prompt-injection test.
- **Plans, critiques, and executes** through three cooperating models, each with a different job and a different trust level — cloud models for planning/review, everything else forced to a local model when privacy-sensitive.
- **Launches applications** only from a fixed allow-list, only with arguments already present in state (never a string lifted straight from an email or chat), and never opens a shell as an OS action.
- **Gates anything destructive** — deletes, force-pushes, outbound email — behind a critic review *and* an explicit human confirmation. No exceptions, regardless of how the request is phrased.
- **Listens for a wake word** ("Hey NOVA" — a custom-trained openWakeWord model), transcribes locally with faster-whisper, and creates a task — with raw audio never touching disk at any point, enforced by a test that makes `open()` raise if anything tries.
- **Tutors**, separately from the task-executing persona: Socratic walkthroughs, code review, quiz mode — read-only, always local, never spends cloud quota, with its own context-window budgeting after a real silent-truncation bug was found and measured.
- **Runs unattended** as a `systemd --user` service, with headless approval (`nova approve/reject`) and desktop notifications standing in for the interactive console, and a bounded-retry preflight standing in for a systemd dependency that literally cannot be expressed across the user/system service boundary.
- **Recovers from `SIGKILL`** using only durable state — no code runs on the way down, so recovery cannot depend on a graceful shutdown path.

## Engineering details worth highlighting

These are the parts that took real investigation rather than following a tutorial, backed by measurement rather than assumption:

- **A silent context-truncation bug.** Ollama's default context window quietly front-truncates prompts with no error — a ~6000-token prompt was capped to 2050 evaluated tokens and the model confidently answered from the surviving tail. Traced, measured, and fixed by explicitly setting `num_ctx` on every local call.
- **Overflow isn't "just the excess."** When a prompt does exceed the context window, llama.cpp's context-shift discards to roughly `num_ctx / 2`, not the minimum needed — verified at two different context sizes. Every local-call budget in the codebase is sized with a measured margin against that specific cliff, not against the nominal ceiling.
- **Real GPU-level cancellation.** Aborting a local generation had to *actually* stop the GPU, not just stop waiting on it — verified live via `amdgpu` utilization dropping from 95% to 0% within ~7s of an abort call, after discovering that an abandoned (but unclosed) stream keeps the server computing a discarded answer to completion.
- **A wake-word retrigger bug traced to a stale internal buffer**, not a series of new detections — fixed with an energy gate plus a refractory window that keeps running real inference (to flush the buffer with live audio) while discarding the result, rather than the naive "pause for N seconds" fix that provably wouldn't have worked.
- **AMD iGPU (gfx1103) is off ROCm's official support list**, so GPU detection can silently fall back to CPU — the correct enable flag was found empirically via `ollama ps`, not from documentation, and is flagged for re-verification on every Ollama upgrade.

## Permission model

| Tier | Examples | Gate |
|---|---|---|
| `read_only` | read files, search, RAG recall | none |
| `file_write` | edit files, commit in workspace | logged |
| `os_action` | launch an allow-listed app | allow-list only, argument provenance required |
| `destructive` | delete, force-push, send email | critic pass **+** explicit human confirmation, always |

## Tech stack (short version)

Python · LangGraph · Pydantic · litellm · Gemini 2.5 Flash · Groq (Llama-3.3-70B) · Ollama (Qwen2.5-Coder, local) · SQLite · Chroma (embedded vector store) · Microsoft Graph API + MSAL · OS keyring (Secret Service) · openWakeWord · faster-whisper (CTranslate2) · systemd (user services) · vanilla JS/HTML/CSS dashboard, zero frontend dependencies.

Full breakdown, versions, and *why each one* in [`TECH_STACK.md`](TECH_STACK.md).

## Testing

248 test functions across 24 files (`pytest`), covering every routing branch, retry-to-escalation, quota-fallback triggering, permission-tier enforcement, cancellation semantics, and the email prompt-injection guarantee. Several of the trickiest behaviors (ALSA mute persistence, wake-word retrigger suppression, abort-cancels-the-GPU) are additionally verified live against real hardware, not only against mocks — documented explicitly where that distinction mattered.

## Running it

```bash
python main.py run --goal "..."      # one-shot: run a single goal through the graph
python main.py loop --interval 300   # persistent: poll inbox + drive the backlog
python main.py dashboard --port 8765 # local web UI, 127.0.0.1 only
python main.py voice                 # wake-word listener → local STT → task
python main.py approve <task_id>     # headless human-gate approval
python main.py reject  <task_id>
```

Designed to run as `systemd --user` services (unit files included) rather than a root daemon — deliberately, since OS-action launches need the live graphical session and secrets need the session's unlocked keyring.

## Project layout

```
nova_state.py        typed state schema for the whole graph (locked file)
route_next_node.py   pure routing logic between nodes (locked file)
os_actions.py         the app allow-list + launch mechanism (locked file)
graph.py             LangGraph wiring
nodes/                director, researcher, critic, executor, tutor, human_gate, email_ingest
llm.py                single litellm-backed transport, timeout + abort handling
backlog.py            SQLite task queue + append-only episodic log
memory.py             embedded Chroma long-term memory
secrets_store.py      OS-keyring-backed credential storage
outlook.py            Microsoft Graph / MSAL integration
voice.py              wake-word → local STT → task pipeline
dashboard.py / .html  local approval + chat UI
tests/                248 tests
```

---

*This is a personal project built to explore what a genuinely autonomous, safety-conscious agent system looks like when you can't hand-wave the hard parts — a single owner, real hardware constraints, and no infrastructure budget beyond two free-tier APIs.*
