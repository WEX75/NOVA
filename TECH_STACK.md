# Technology Stack

A full inventory of what NOVA is built with, and why each piece was chosen. Grouped by concern rather than alphabetically, since the *why* mostly follows from the constraint it solves.

## Language & runtime

| Technology | Role |
|---|---|
| **Python 3.12** | Entire project — orchestration, nodes, tooling, tests |

## Orchestration & state

| Technology | Role |
|---|---|
| **LangGraph** | The core state machine. Drives the Director → Researcher/Critic → Executor pipeline, with `interrupt_before=["human_gate"]` used for *genuine* execution halts (not a logged-and-ignored flag) on destructive or approval-required actions |
| **Pydantic** | Every node's input/output is a typed model, never a raw dict — `NovaState`, `Subtask`, `AgentStep`, `ToolCallResult`, etc. |

## LLM access

| Technology | Role |
|---|---|
| **litellm** | Single unified interface (`llm.py`) for every model call — no node imports a provider SDK directly. Handles both cloud and local (Ollama) backends through one `completion()` call |
| **Gemini 2.5 Flash** | Director / Researcher roles — cloud, free tier (~250 req/day) |
| **Groq — Llama-3.3-70B** | Critic role — cloud, free tier (~1000 req/day), exactly one review pass per subtask by design |
| **Ollama** | Local model runtime for the Executor and Tutor roles |
| **Qwen2.5-Coder** | The local model served by Ollama — handles all privacy-sensitive tasks (email-derived, voice-derived) and all file/code/git operations |

## Hardware & local inference tuning

| Technology | Role |
|---|---|
| **AMD Radeon 780M (gfx1103) / ROCm-adjacent iGPU path** | Target inference hardware — not on ROCm's official support list, so GPU pickup had to be empirically verified (`ollama ps`) rather than assumed from a documented flag |
| **`num_ctx` tuning** | Explicit context-window sizing on every local call after discovering Ollama's 4096-token default silently front-truncates prompts with no error |

## Storage & memory

| Technology | Role |
|---|---|
| **SQLite** (`sqlite3`, stdlib) | Task queue (`backlog.py`) and the append-only episodic log — no ORM, no deletes, ever |
| **Chroma** (embedded, `chromadb`) | Local vector store for RAG recall of past business context and preferences — no external vector DB service |

## Integrations

| Technology | Role |
|---|---|
| **Microsoft Graph API** | Reads/sends Outlook mail |
| **MSAL (Microsoft Authentication Library)** | OAuth token acquisition and refresh for Graph access |
| **`requests`** | HTTP transport for Graph API calls |
| **`keyring`** (OS Secret Service backend) | All credentials — Gemini/Groq API keys, the MSAL token cache, the MS client ID — stored here, never in `.env` files, plaintext JSON, or service-unit environment variables |

## Voice pipeline

| Technology | Role |
|---|---|
| **openWakeWord** | Wake-word detection; runs a custom-trained "Hey NOVA" ONNX model, replacing the stock "Hey Jarvis" default |
| **sounddevice** (PortAudio bindings) | Live microphone capture |
| **faster-whisper** (CTranslate2 backend) | Local, CPU-only speech-to-text — confirmed to have zero GPU/ROCm path on this hardware, so it never contends with the LLM for VRAM |
| **numpy** | In-memory audio buffer handling — audio never touches disk, verified by a test that makes `open()` raise during capture |

## Deployment & process management

| Technology | Role |
|---|---|
| **systemd (`--user` services)** | Runs the main loop, voice daemon, and dashboard as session-scoped services — required because OS-action app launches and keyring unlock both need the live graphical/login session, not a headless system daemon |
| **journald** | Log capture and rotation — stdout-only logging, no custom log-rotation code |

## Frontend (dashboard)

| Technology | Role |
|---|---|
| **Vanilla HTML/CSS/JS** | The local dashboard (`dashboard.html`) — no framework, no build step, no external CDN dependency; served by a plain `http.server`-based Python handler bound to `127.0.0.1` |

## Testing

| Technology | Role |
|---|---|
| **pytest** | 248 test functions across 24 files — routing branches, retry-to-escalation, quota fallback, permission-tier enforcement, cancellation semantics, and the email/voice prompt-injection guarantees |

## Design patterns worth naming on a CV

- **Typed message-passing state machine** — nodes never mutate shared state; they return partial updates that pure reducers merge.
- **Tiered permission model** (`read_only` / `file_write` / `os_action` / `destructive`) enforced structurally, not just by convention.
- **Human-in-the-loop as a real execution interrupt**, not a soft confirmation flag — via LangGraph's `interrupt_before`.
- **Provider-agnostic LLM transport** — swapping any of the three model backends touches one file (`llm.py`), never node logic.
- **Append-only audit logging** as a hard invariant (no delete path exists), enabling full post-hoc reconstruction of any autonomous decision.
- **Defense-in-depth on a single failure mode** — e.g. the voice wake-word retrigger fix uses two independent layers (an energy gate *and* a refractory window) rather than one patch.
