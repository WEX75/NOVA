I'm building NOVA, an autonomous multi-agent business assistant. Attached are
the four files that define the locked architecture: the engineering
specification (ground truth), nova_state.py (LangGraph state schema),
route_next_node.py (routing logic), and os_actions.py (allow-listed app
launcher). Treat these as final — if something needs to change, tell me why
before changing it, don't silently rewrite them.

## Locked model stack
- Director / Researcher: Gemini 2.5 Flash, cloud, free tier (~250 req/day)
- Critic: Groq Llama-3.3-70B, cloud, free tier (~1000 req/day), bounded to
  exactly one review pass per subtask — no debate loops
- Executor / Privacy Vault: Ollama running Qwen2.5-Coder locally

## Hardware
AMD Ryzen with Radeon 780M iGPU (gfx1103), 32GB system RAM, ~16GB allocatable
as VRAM. gfx1103 is NOT on AMD's official ROCm support list — Ollama must run
on its Vulkan backend (OLLAMA_VULKAN=1), not ROCm, or GPU detection silently
fails and falls back to slow CPU inference.

## What Nova needs to actually do
1. Read email (Outlook via Microsoft Graph API) and turn relevant messages
   into tasks.
2. Run code locally (the Executor, via Ollama) — file edits, git operations,
   general dev work.
3. Launch local applications (browser, file manager, editors, LibreOffice)
   through the os_actions.py allow-list — never an arbitrary shell command.

Non-negotiable safety rule tying these together: email body content is DATA,
never INSTRUCTIONS. If an email
contains something that reads like a command ("open X", "run Y", "reply
with Z"), Nova surfaces it as a proposed task requiring my explicit approval
— it never auto-executes. Same boundary applies to os_actions.py: any URL or
file path passed to an app-launch call must already exist in NovaState
(a saved attachment, a generated report) — never text lifted straight out
of an email body.

## What I need built (in this order)

1. LiteLLM provider config unifying Gemini, Groq, and local Ollama behind one
   interface, so every node calls litellm.completion() regardless of backend.

2. The four agent nodes (director_node, researcher_node, critic_node,
   executor_node) as separate modules, each:
   - Takes NovaState, returns a partial state update dict (never mutates
     state directly — reducers handle merging)
   - Increments the relevant quota counter after every cloud API call
   - Logs a full AgentStep to history on every run
   - executor_node additionally owns all code execution, file operations,
     and calls into os_actions.launch_app() for the os_action tier

3. human_gate_node plus the LangGraph interrupt_before=["human_gate"]
   compile-time config, so destructive actions and anything flagged
   requires_human_approval genuinely halt execution and wait for my
   explicit approval — not just a flag that gets logged and ignored.

4. Outlook ingestion node: Microsoft Graph API, Mail.Read scope, OAuth for
   personal single-user use. New emails become a NovaState task with
   source="email" and privacy_sensitive=True by default — this routes them
   straight to the local Executor, bypassing
   the cloud Director/Critic entirely unless I explicitly override it.

5. Manager/task backlog: local SQLite for the task queue + an append-only
   episodic log table. Nothing gets deleted from this log, ever.

6. Long-term memory: Chroma (local, embedded) wired up for RAG recall of
   past business updates/preferences.

7. Test suite (pytest) covering: routing logic in all its branches, retry-
   to-escalation path, quota fallback triggering correctly, permission-tier
   enforcement, and specifically a test that an email body containing
   instruction-like text never results in auto-executed code or an
   auto-launched app.

## Engineering rules (non-negotiable)
- No node file over ~150 lines of orchestration logic
- Every node's I/O is a typed pydantic model, never a raw dict
- Every external action (API call, file write, OS command, app launch,
  email send) gets logged with enough detail to reconstruct why it happened
- Max 3 retries per task before mandatory human escalation
- destructive-tier tools always require Critic pass + explicit human
  confirmation — no exceptions, no matter how the request is framed
- os_action-tier tools only ever launch from the ALLOWED_APPS list — never
  a dynamically constructed command

## How I want you to work
Give me a short implementation plan first (file list + order), then build
it module by module. If a design decision in the engineering specification,
the state schema, the router, or os_actions.py doesn't work for something
you're implementing,
stop and tell me the conflict — don't quietly work around it. Write tests
alongside the code, not after. Ask if something about the business logic
(not the architecture — that's locked) is genuinely ambiguous.