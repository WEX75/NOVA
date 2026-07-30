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

Retrigger fix (§16 follow-up, 2026-07-30). One real utterance was
producing 4-8 spurious extra "tasks" from near-silence in the seconds
after a genuine trigger. Root cause, traced precisely: during
CAPTURE_SECONDS of speech capture, raw audio is read directly via
mic.read_chunk() and NEVER fed through wake_model.predict() — so
openWakeWord's internal feature buffer (melspectrogram/embedding history
inside its AudioFeatures preprocessor) stays frozen at whatever it held
at the moment of the original trigger. openWakeWord's own Model.reset()
does NOT clear this — it only clears the unrelated prediction_buffer used
by the opt-in `patience` option; there is no public API to flush the
feature buffer directly. The first predict() call(s) after capture ends
see mostly-stale, trigger-adjacent features, and score above threshold
again — a leak of the SAME trigger across multiple predict() calls, not
a new detection. Simply "not calling predict() for N seconds" (an
initial hypothesis) would NOT fix this: the buffer only advances on
predict() calls, not wall-clock time, so skipping inference during a
cooldown leaves it exactly as stale when the cooldown ends. The fix
below instead keeps calling predict() (to genuinely flush the buffer
with real, current audio) but discards every score for
REFRACTORY_SECONDS after any capture.

Live-measured RMS energy (2026-07-30, real mic, real room) showed the
spurious captures are genuinely near-silent — 0.0016-0.0044, close to
the measured ambient baseline of 0.0019 — while real speech measured
0.0197, ~10x higher. Whisper, true to its documented behavior, still
hallucinated short plausible-sounding text from that near-silence
("You", earlier "I'm not sure if I can make a video...") rather than
returning empty — confirming an energy gate is the right fix, not a
smarter transcription check. VAD_RMS_THRESHOLD (0.01) skips Whisper
entirely, and skips task creation entirely, for anything captured below
it — near-silence never reaches the LLM pipeline as a "task" in the
first place.

no_signal detection (§16 follow-up, 2026-07-30, same investigation).
Diagnosing the ALSA-mute finding above took a side probe script and a
manual `amixer` check — nothing in the daemon's own status surfaced that
"listening" actually meant "the process is running, hearing exact
silence" rather than "genuinely picking up the room." voice_status can
now report "no_signal": if NO_SIGNAL_CHUNKS_THRESHOLD consecutive idle
chunks read as bit-for-bit RMS 0.0 while unmuted, the periodic status
write reports "no_signal" instead of "listening". This is deliberately
distinct from "muted" (an owner choice) and "error" (a crash) — a dead
mic with no code error at all is a real, silent-failure third state, not
a variant of either. _ensure_alsa_capture_unmuted() (RealMicSource
startup) is a best-effort fix for the ONE confirmed cause on this
machine; no_signal is the general symptom detector that doesn't depend
on the cause being that one thing again.

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

# Retrigger fix (§16 follow-up, 2026-07-30) — see module docstring's
# "Retrigger fix" section for the full investigation. Both constants are
# calibrated from real live captures on this machine, not guessed:
# ambient/near-silence RMS measured 0.0016-0.0044 (quiet-room baseline
# 0.0019); real speech measured 0.0197. 0.01 sits with ~2x margin on
# both sides.
VAD_RMS_THRESHOLD = 0.01
REFRACTORY_SECONDS = 2.0

# no_signal detection (§16 follow-up, 2026-07-30): "listening" previously
# meant only "the process is running," not "real audio is reaching it" —
# the exact ambiguity that took a side probe script and an amixer check
# to resolve this session. A truly dead/muted input reads as EXACTLY 0.0
# RMS (bit-for-bit silence) on every chunk; genuine ambient room noise
# never does — the calibration baseline measured 0.0019, not 0.0. 25
# consecutive exact-zero 80ms chunks (~2s) is a deliberately generous
# margin against one fluke all-zero buffer before surfacing this as a
# distinct status, not a hair-trigger.
NO_SIGNAL_CHUNKS_THRESHOLD = 25


class MicSource(Protocol):
    """Yields raw int16 mono chunks of exactly CHUNK_SAMPLES, one per call,
    blocking until real-time-equivalent data is available."""

    def read_chunk(self) -> np.ndarray: ...
    def close(self) -> None: ...


def _ensure_alsa_capture_unmuted() -> None:
    """Best-effort defensive fix (§16 follow-up, 2026-07-30): this
    machine's ALSA capture switch was found muted at startup once already
    — traced to /var/lib/alsa/asound.state genuinely persisting
    `Capture Switch value false`, which alsa-restore.service re-applies
    on EVERY boot (confirmed by reading the state file directly, not
    guessed) — so a reboot silently re-mutes the mic with no daemon error
    at all. This can only paper over that one specific, already-confirmed
    cause; it is NOT a substitute for the no_signal daemon status below,
    which detects the symptom (no real audio reaching the daemon)
    regardless of cause. Never raises: a failed amixer call must not
    block the daemon from starting, and the card number is a guess
    (NOVA_VOICE_ALSA_CARD, default matching THIS machine) that may not
    hold on different hardware."""
    import subprocess
    card = os.environ.get("NOVA_VOICE_ALSA_CARD", "1")
    try:
        out = subprocess.run(["amixer", "-c", card, "sget", "Capture"],
                             capture_output=True, text=True, timeout=5)
        if "[off]" not in out.stdout:
            return  # already unmuted — nothing to do
        log.warning("voice daemon: ALSA capture on card %s was muted at "
                   "startup — unmuting (see project guide §16: this reverts "
                   "on every reboot unless `alsactl store` is run)", card)
        # "cap"/"nocap" are the ALSA simple-control keywords for this
        # capture switch on this hardware — "mute"/"unmute" both returned
        # "amixer: Invalid command!" when tried directly (verified live,
        # 2026-07-30), and a first version of this function used "unmute"
        # anyway (untested against a real re-mute) and silently did
        # nothing — hence the self-verification below, not a fire-and-
        # forget subprocess call.
        subprocess.run(["amixer", "-c", card, "sset", "Capture", "cap"],
                       capture_output=True, timeout=5)
        subprocess.run(["amixer", "-c", card, "sset", "Capture", "80%"],
                       capture_output=True, timeout=5)
        verify = subprocess.run(["amixer", "-c", card, "sget", "Capture"],
                                capture_output=True, text=True, timeout=5)
        if "[off]" in verify.stdout:
            log.error("voice daemon: attempted to unmute ALSA capture on "
                     "card %s but it is STILL muted — the daemon will run "
                     "but hear nothing; fix by hand with "
                     "`amixer -c %s sset Capture cap` (project guide §16)",
                     card, card)
        else:
            log.info("voice daemon: ALSA capture on card %s unmuted successfully", card)
    except Exception as e:  # noqa: BLE001 — best-effort only, see docstring
        log.warning("voice daemon: could not check/fix ALSA capture mute "
                   "(%s) — if voice seems dead with no errors, check "
                   "`amixer -c %s get Capture` by hand (project guide §16)",
                   e, card)


class RealMicSource:
    """The only piece of this module that touches actual hardware. Needs
    sounddevice + the system libportaudio2 (see module docstring)."""

    def __init__(self):
        _ensure_alsa_capture_unmuted()
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


def _capture_audio(mic: MicSource, n_chunks: int) -> np.ndarray:
    """Read n_chunks of raw audio into one local int16 buffer. Caller owns
    its lifetime and must discard it promptly — see the audio-never-
    persisted guarantee in the module docstring."""
    frames = [mic.read_chunk() for _ in range(n_chunks)]
    audio = np.concatenate(frames)
    del frames
    return audio


def _rms(audio_int16: np.ndarray) -> float:
    """RMS energy of a raw int16 buffer, normalized to [0, 1]. This is a
    DERIVED scalar — like the transcribed text — not audio data itself, so
    logging it does not touch the audio-never-persisted guarantee; the
    buffer it was computed from is still discarded immediately after use,
    exactly as before."""
    if audio_int16.size == 0:
        return 0.0
    samples = audio_int16.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _transcribe(whisper, audio_int16: np.ndarray) -> str:
    """The only place raw audio is fed to a model. Nothing here writes to
    disk; the caller discards audio_int16 right after this returns."""
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    segments, _info = whisper.transcribe(audio_f32, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


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
                     vad_rms_threshold: float = VAD_RMS_THRESHOLD,
                     refractory_seconds: float = REFRACTORY_SECONDS,
                     status_interval_seconds: float = STATUS_INTERVAL_SECONDS,
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
    gets the genuine openWakeWord/faster-whisper models.

    `vad_rms_threshold`/`refractory_seconds` are injectable so tests can
    run the refractory window at ~0s (it is a real wall-clock sleep loop
    in production — see the module docstring's "Retrigger fix" section)
    without every triggered-capture test taking REFRACTORY_SECONDS
    real seconds. `status_interval_seconds` is injectable so tests can
    force a status write on every iteration (0) — in production the
    periodic 5s gate exists to avoid hammering the DB every 80ms, but
    that same gate makes a fast test's status transitions (e.g.
    no_signal) unobservable across a handful of near-instant iterations."""
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
        consecutive_zero_chunks = 0
        iterations = 0
        while not stop():
            iterations += 1
            if max_iterations is not None and iterations > max_iterations:
                break
            try:
                if backlog.is_voice_muted():
                    backlog.write_voice_status("muted")
                    consecutive_zero_chunks = 0  # start clean whenever unmuted again
                    time.sleep(MUTED_POLL_SECONDS)
                    continue  # muted: no chunk is ever read, no inference ever runs

                chunk = mic.read_chunk()
                if _rms(chunk) == 0.0:
                    consecutive_zero_chunks += 1
                else:
                    consecutive_zero_chunks = 0
                scores = wake_model.predict(chunk)
                triggered = any(score >= WAKE_THRESHOLD for score in scores.values())

                now = time.monotonic()
                if now - last_status_write >= status_interval_seconds:
                    # "listening" previously meant only "the process is
                    # running" — no_signal distinguishes "genuinely no
                    # audio reaching the daemon" (bit-for-bit silence,
                    # unmuted) from real ambient-quiet listening (which
                    # never reads as exact zero) and from muted (an
                    # owner choice) and error (a crash). See module
                    # docstring's "no_signal detection" note.
                    status = ("no_signal" if consecutive_zero_chunks >= NO_SIGNAL_CHUNKS_THRESHOLD
                             else "listening")
                    backlog.write_voice_status(status)
                    last_status_write = now

                if not triggered:
                    continue

                log.info("voice daemon: wake word triggered, capturing speech")
                n_chunks = int(CAPTURE_SECONDS * SAMPLE_RATE / CHUNK_SAMPLES)
                audio = _capture_audio(mic, n_chunks)
                rms = _rms(audio)
                if rms < vad_rms_threshold:
                    # Near-silence never reaches Whisper or becomes a task —
                    # see module docstring: Whisper hallucinates plausible
                    # text from near-silence rather than returning empty,
                    # so skipping it here (not just checking its output) is
                    # what actually keeps ambient noise out of the pipeline.
                    log.info("voice daemon: capture below energy threshold "
                            "(rms=%.5f < %.5f) — near-silence, skipping "
                            "transcription, no task created", rms, vad_rms_threshold)
                else:
                    text = _transcribe(whisper_model, audio)
                    task_id = _make_voice_task(backlog, text)
                    if task_id:
                        log.info("voice daemon: created task %s from speech (rms=%.5f)",
                                 task_id, rms)
                    else:
                        log.info("voice daemon: transcription was empty, "
                                "no task created (rms=%.5f)", rms)
                del audio  # discarded immediately — see module docstring

                # Refractory: keep running REAL inference (to flush the wake
                # model's stale feature buffer with genuinely current audio)
                # but discard every score for refractory_seconds — see the
                # module docstring for why merely not calling predict() here
                # would leave the buffer just as stale when this ends.
                refractory_until = time.monotonic() + refractory_seconds
                while time.monotonic() < refractory_until and not stop():
                    wake_model.predict(mic.read_chunk())  # score discarded on purpose
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
