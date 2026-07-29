"""
NOVA — Voice daemon (§16): wake-word -> local STT -> task.

nova-voice.service (systemd --user) runs run_voice_daemon() persistently.
Flow, every cycle:
  1. Listen for the wake word (openWakeWord, ONNX, CPU-only — measured
     ~9.6% of one core idle, 267 MB RSS: negligible background cost).
  2. On trigger, capture a fixed window of speech into an IN-MEMORY numpy
     buffer only — raw audio is NEVER written to disk, at any point.
  3. Transcribe locally (faster-whisper, CPU). This hardware's CTranslate2
     build reports zero CUDA devices and has no ROCm/Vulkan path at all —
     confirmed, not assumed — so there is no GPU/VRAM contention with
     qwen2.5-coder possible: they physically cannot compete for the same
     resource. Measured CPU contention while qwen actively generates:
     transcription of 4s of speech took 3.82s (still faster than the
     speaker took to say it) versus 2.17s with qwen idle; qwen's own
     generation speed was unaffected (it's GPU-bound, whisper is CPU-bound).
  4. Write the TRANSCRIBED TEXT (never the audio) as a new task:
     source="voice", privacy_sensitive=True. The pre-existing
     privacy_sensitive routing path (route_next_node, unchanged, verified
     2026-07-30) sends it straight to the local Executor — no cloud, no
     Director/Critic, regardless of quota. Backlog.add_task's own
     "new task from {source}" audit entry covers this step; no extra
     logging needed here.
  5. The in-memory audio buffer goes out of scope immediately after step 3
     — nothing further references it, nothing persists it.

Mute is real, not cosmetic: while muted, this daemon skips wake-word
INFERENCE on captured audio entirely (no analysis happens at all — not
merely "a UI flag the daemon ignores"). Status (listening/muted/error) and
mute are independent writers on one shared row (Backlog.write_voice_status
is daemon-owned, Backlog.set_voice_muted is dashboard-owned) — see
backlog.py's §16 section for why neither can clobber the other.

The microphone is injected via the MicSource protocol specifically so
this module's actual logic — wake detection, mute enforcement,
transcription, task creation, and the audio-never-persisted guarantee —
is unit-testable without real hardware or the system PortAudio library
(libportaudio2; NOT installable from this sandboxed session — sudo here
requires an interactive password. RealMicSource needs it; the daemon's
logic does not).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from typing import Callable, Optional, Protocol

import numpy as np

from backlog import Backlog, TaskRecord

log = logging.getLogger("nova")

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1_280          # openWakeWord's expected 80ms frame @ 16kHz
CAPTURE_SECONDS = 6.0          # fixed window of speech captured after a trigger
WAKE_THRESHOLD = 0.5           # openWakeWord's own convention
STATUS_INTERVAL_SECONDS = 5.0  # how often "listening" is re-written while idle
MUTED_POLL_SECONDS = 1.0


class MicSource(Protocol):
    """Yields raw int16 mono chunks of exactly CHUNK_SAMPLES, one per call,
    blocking until real-time-equivalent data is available."""

    def read_chunk(self) -> np.ndarray: ...
    def close(self) -> None: ...


class RealMicSource:
    """The only piece of this module that touches actual hardware. Needs
    sounddevice + the system libportaudio2 (see module docstring)."""

    def __init__(self):
        import sounddevice as sd  # deferred: only this class needs it
        self._sd = sd
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=CHUNK_SAMPLES)
        self._stream.start()

    def read_chunk(self) -> np.ndarray:
        data, _overflowed = self._stream.read(CHUNK_SAMPLES)
        return data[:, 0]

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def _default_wakeword_model_path() -> str:
    import openwakeword
    return os.path.join(os.path.dirname(openwakeword.__file__),
                        "resources", "models", "hey_jarvis_v0.1.onnx")


def _load_models(wakeword_model_path: Optional[str], whisper_model_size: str):
    from faster_whisper import WhisperModel
    from openwakeword.model import Model as WakeModel

    wake = WakeModel(wakeword_model_paths=[wakeword_model_path or _default_wakeword_model_path()])
    whisper = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
    return wake, whisper


def _capture_and_transcribe(mic: MicSource, whisper, n_chunks: int) -> str:
    """The only place raw audio exists: a local list, concatenated into one
    local array, transcribed, and then never referenced again. Nothing here
    writes to disk."""
    frames = [mic.read_chunk() for _ in range(n_chunks)]
    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    del frames
    segments, _info = whisper.transcribe(audio, language="en")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    del audio  # explicit: the buffer's lifetime ends here, on purpose
    return text


def _make_voice_task(backlog: Backlog, text: str) -> Optional[str]:
    if not text:
        return None
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    backlog.add_task(TaskRecord(task_id=task_id, goal=text,
                                source="voice", privacy_sensitive=True))
    return task_id


def run_voice_daemon(backlog: Backlog, mic: Optional[MicSource] = None,
                     wakeword_model_path: Optional[str] = None,
                     whisper_model_size: str = "base",
                     wake_model=None, whisper_model=None,
                     stop: Optional[Callable[[], bool]] = None,
                     max_iterations: Optional[int] = None) -> None:
    """Persistent loop. `stop`, when given, is polled each cycle (matches
    the `threading.Event`-style cooperative shutdown used elsewhere in this
    project, e.g. main.loop's `stop.is_set()`). `max_iterations` exists for
    tests — the real service never sets it.

    `wake_model`/`whisper_model` are injectable for the same reason `mic`
    is (tests use lightweight fakes — this project's established pattern,
    see conftest.py's fake_llm — rather than downloading and loading real
    model weights on every test run); the real daemon leaves them None and
    gets the genuine openWakeWord/faster-whisper models."""
    stop = stop or (lambda: False)
    mic_owned = mic is None
    mic = mic or RealMicSource()
    try:
        if wake_model is None or whisper_model is None:
            real_wake, real_whisper = _load_models(wakeword_model_path, whisper_model_size)
            wake_model = wake_model or real_wake
            whisper_model = whisper_model or real_whisper
        log.info("voice daemon: models loaded, listening")
        last_status_write = 0.0
        iterations = 0
        while not stop():
            iterations += 1
            if max_iterations is not None and iterations > max_iterations:
                break
            try:
                if backlog.is_voice_muted():
                    backlog.write_voice_status("muted")
                    time.sleep(MUTED_POLL_SECONDS)
                    continue  # muted: no chunk is ever read, no inference ever runs

                chunk = mic.read_chunk()
                scores = wake_model.predict(chunk)
                triggered = any(score >= WAKE_THRESHOLD for score in scores.values())

                now = time.monotonic()
                if now - last_status_write >= STATUS_INTERVAL_SECONDS:
                    backlog.write_voice_status("listening")
                    last_status_write = now

                if not triggered:
                    continue

                log.info("voice daemon: wake word triggered, capturing speech")
                n_chunks = int(CAPTURE_SECONDS * SAMPLE_RATE / CHUNK_SAMPLES)
                text = _capture_and_transcribe(mic, whisper_model, n_chunks)
                task_id = _make_voice_task(backlog, text)
                if task_id:
                    log.info("voice daemon: created task %s from speech", task_id)
                else:
                    log.info("voice daemon: transcription was empty, no task created")
            except Exception as e:  # noqa: BLE001 — one bad cycle must not kill the daemon
                log.error("voice daemon: cycle failed: %s", e)
                backlog.write_voice_status("error")
    finally:
        if mic_owned:
            mic.close()


def main_cli(backlog: Backlog, whisper_model_size: str = "base") -> int:
    """Entry point for `nova voice` / nova-voice.service. Same cooperative
    SIGTERM/SIGINT shutdown as main.loop() — finishes the current cycle,
    then exits; a second signal force-quits."""
    stop_event = threading.Event()

    def handler(signum, frame):
        log.info("voice daemon: received %s — stopping after this cycle "
                 "(send again to force-quit)", signal.Signals(signum).name)
        stop_event.set()
        signal.signal(signum, signal.SIG_DFL)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)

    run_voice_daemon(backlog, whisper_model_size=whisper_model_size,
                     stop=stop_event.is_set)
    log.info("voice daemon: stopped")
    return 0
