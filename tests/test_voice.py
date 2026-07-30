"""§16 voice daemon: wake-word -> local STT -> task. Uses fake mic/wake/
whisper models (this project's established pattern — see conftest.py's
fake_llm — rather than downloading real model weights on every test run).
The real openWakeWord/faster-whisper integration was proven live and
separately (measured CPU overhead, measured CPU contention with qwen,
confirmed zero GPU/VRAM path on this hardware) — see project guide §16.

Retrigger-fix tests (2026-07-30) pass refractory_seconds=0 in most cases —
production defaults to a real 2s wall-clock window (module docstring in
voice.py explains why it must be real time, not a mock), which would make
every triggered-capture test slow; the dedicated refractory tests below
use a small nonzero value instead, specifically to prove the window has
a real, measurable effect."""

from __future__ import annotations

import numpy as np
import pytest

from backlog import TaskRecord
from voice import CHUNK_SAMPLES, VAD_RMS_THRESHOLD, run_voice_daemon


class FakeMic:
    """amplitude controls the RMS of generated audio: the default (5000)
    is well above VAD_RMS_THRESHOLD — "speech" — so existing trigger/task
    tests don't need to know the VAD gate exists. amplitude=0 is genuine
    silence for VAD-gate tests specifically."""

    def __init__(self, amplitude=5000):
        self.reads = 0
        self.closed = False
        self.amplitude = amplitude
        self._rng = np.random.default_rng(0)

    def read_chunk(self):
        self.reads += 1
        if self.amplitude == 0:
            return np.zeros(CHUNK_SAMPLES, dtype=np.int16)
        return self._rng.integers(-self.amplitude, self.amplitude,
                                  CHUNK_SAMPLES, dtype=np.int16)

    def close(self):
        self.closed = True


class FakeWakeModel:
    """predict() returns {"hey_nova": score}. `trigger_on` is a set of
    1-indexed call numbers on which the score exceeds threshold.
    `trigger_from`, if given, means "this call and every one after" also
    scores above threshold — simulates the stale-buffer leak this fix
    addresses, for testing that refractory actually suppresses it."""

    def __init__(self, trigger_on=frozenset(), trigger_from=None):
        self.trigger_on = trigger_on
        self.trigger_from = trigger_from
        self.calls = 0

    def predict(self, chunk):
        self.calls += 1
        hit = self.calls in self.trigger_on or (
            self.trigger_from is not None and self.calls >= self.trigger_from)
        return {"hey_nova": 0.9 if hit else 0.0}


class _Seg:
    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    def __init__(self, text="turn on the living room lights"):
        self.text = text
        self.transcribe_calls = []

    def transcribe(self, audio, language="en"):
        self.transcribe_calls.append(audio)
        return ([_Seg(self.text)] if self.text else []), object()


def stop_after(n):
    count = {"n": 0}
    def _stop():
        count["n"] += 1
        return count["n"] > n
    return _stop


# --- mute is real, not cosmetic -------------------------------------------------

def test_muted_daemon_never_runs_wake_word_inference(temp_backlog):
    """The core requirement: while muted, no audio is ever analyzed at
    all — not merely a status flag the daemon ignores."""
    temp_backlog.set_voice_muted(True)
    mic = FakeMic()
    wake = FakeWakeModel()

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=FakeWhisperModel(), max_iterations=5)

    assert wake.calls == 0
    assert mic.reads == 0                      # no chunk is even read
    assert temp_backlog.read_voice_status()["status"] == "muted"


def test_unmuted_daemon_listens_and_writes_status(temp_backlog):
    temp_backlog.set_voice_muted(False)
    wake = FakeWakeModel()  # never triggers

    run_voice_daemon(temp_backlog, mic=FakeMic(), wake_model=wake,
                     whisper_model=FakeWhisperModel(), max_iterations=3)

    assert wake.calls == 3
    assert temp_backlog.read_voice_status()["status"] == "listening"


# --- trigger -> capture -> transcribe -> task ------------------------------------

def test_wake_trigger_creates_a_voice_task_privacy_sensitive(temp_backlog):
    wake = FakeWakeModel(trigger_on={1})
    whisper = FakeWhisperModel(text="what is on my calendar today")

    run_voice_daemon(temp_backlog, mic=FakeMic(), wake_model=wake,
                     whisper_model=whisper, refractory_seconds=0, max_iterations=1)

    tasks = temp_backlog.pending_tasks()
    assert len(tasks) == 1
    assert tasks[0].goal == "what is on my calendar today"
    assert tasks[0].source == "voice"
    assert tasks[0].privacy_sensitive is True
    # add_task's own audit entry covers this — no separate voice-specific log needed
    assert any(e["action_type"] == "task_created" and e["reason"] == "new task from voice"
              for e in temp_backlog.log_entries(task_id=tasks[0].task_id))


def test_empty_transcription_creates_no_task(temp_backlog):
    wake = FakeWakeModel(trigger_on={1})
    whisper = FakeWhisperModel(text="")

    run_voice_daemon(temp_backlog, mic=FakeMic(), wake_model=wake,
                     whisper_model=whisper, refractory_seconds=0, max_iterations=1)

    assert temp_backlog.pending_tasks() == []


def test_capture_reads_the_expected_window_of_audio(temp_backlog):
    """CAPTURE_SECONDS worth of chunks are read after a trigger — not the
    single triggering chunk alone."""
    from voice import CAPTURE_SECONDS, SAMPLE_RATE
    wake = FakeWakeModel(trigger_on={1})
    whisper = FakeWhisperModel()
    mic = FakeMic()

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=whisper, refractory_seconds=0, max_iterations=1)

    expected_capture_chunks = int(CAPTURE_SECONDS * SAMPLE_RATE / CHUNK_SAMPLES)
    assert mic.reads == 1 + expected_capture_chunks   # 1 for the wake chunk itself
    assert len(whisper.transcribe_calls) == 1
    captured = whisper.transcribe_calls[0]
    assert len(captured) == expected_capture_chunks * CHUNK_SAMPLES


# --- VAD/energy gate (§16 retrigger fix, 2026-07-30) ------------------------------

def test_near_silent_capture_skips_transcription_and_creates_no_task(temp_backlog):
    """The core VAD requirement: near-silence never reaches Whisper at
    all, not just "Whisper's output gets discarded" — proven by asserting
    transcribe() was never called, not merely that no task exists (Whisper
    is documented to hallucinate plausible text from near-silence rather
    than returning empty, so checking output alone would be the wrong
    test — see the real 'You'/'I'm not sure if I can make a video' live
    hallucinations recorded in project guide §16)."""
    wake = FakeWakeModel(trigger_on={1})
    whisper = FakeWhisperModel(text="hallucinated text that should never appear")
    mic = FakeMic(amplitude=0)   # genuine silence: rms will be exactly 0.0

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake, whisper_model=whisper,
                     refractory_seconds=0, max_iterations=1)

    assert whisper.transcribe_calls == []       # never invoked
    assert temp_backlog.pending_tasks() == []


def test_loud_capture_still_passes_the_vad_gate(temp_backlog):
    """The gate must not raise the bar for real detections — real-energy
    audio (the FakeMic default) still reaches transcription and creates
    a task, same as before the fix."""
    wake = FakeWakeModel(trigger_on={1})
    whisper = FakeWhisperModel(text="what time is it")
    mic = FakeMic(amplitude=5000)

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake, whisper_model=whisper,
                     refractory_seconds=0, max_iterations=1)

    assert len(whisper.transcribe_calls) == 1
    assert temp_backlog.pending_tasks()[0].goal == "what time is it"


def test_vad_threshold_matches_the_live_calibrated_constant():
    """Guards against silent drift: the constant used in production must
    stay the live-measured value (0.01, ~2x margin above the measured
    near-silence ceiling of 0.0044 and ~2x below measured real speech of
    0.0197 — see project guide §16), not a value someone changes without
    re-measuring."""
    assert VAD_RMS_THRESHOLD == 0.01


# --- refractory period (§16 retrigger fix, 2026-07-30) ----------------------------

def test_refractory_suppresses_retrigger_from_a_leaking_wake_model(temp_backlog):
    """The core fix, proven directly: even a wake model that scores above
    threshold on EVERY call from the trigger onward (worst-case simulation
    of the real stale-buffer leak) produces exactly ONE task, not one per
    predict() call — because scores during the refractory window are
    discarded, not acted on."""
    wake = FakeWakeModel(trigger_from=1)   # "stuck" — always triggers, every call
    whisper = FakeWhisperModel(text="hello")
    mic = FakeMic(amplitude=5000)

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake, whisper_model=whisper,
                     refractory_seconds=0.05, max_iterations=1)

    assert len(temp_backlog.pending_tasks()) == 1     # not one per leaked score
    # the refractory window DID keep running real inference (to flush the
    # stale buffer with genuine audio) — it just discarded the results
    assert wake.calls > 1


def test_refractory_does_not_block_reads_forever_when_stopped(temp_backlog):
    """stop() is still honored inside the refractory wait loop — shutdown
    during refractory does not hang. A timing assertion, not just a
    functional one: if `and not stop()` were ever dropped from the
    refractory loop's condition, this test would still pass functionally
    (the task was already created) but take ~5s wall-clock instead of
    near-instant — the elapsed-time bound is what actually catches that
    regression."""
    import time as _time
    wake = FakeWakeModel(trigger_on={1})
    mic = FakeMic(amplitude=5000)
    stop = stop_after(1)   # becomes True on the very next check

    t0 = _time.monotonic()
    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=FakeWhisperModel(), refractory_seconds=5.0, stop=stop)
    elapsed = _time.monotonic() - t0

    assert temp_backlog.pending_tasks()   # the trigger's own task still got created
    assert elapsed < 1.0                  # did NOT wait out the full 5s refractory


# --- raw audio is never persisted to disk ----------------------------------------

def test_capture_and_transcribe_never_touches_the_filesystem(temp_backlog, monkeypatch):
    """The whole point of §16's audio guarantee: proven, not asserted —
    open() must never be called anywhere in the capture->transcribe path."""
    import builtins

    real_open = builtins.open
    def guarded_open(*a, **kw):
        raise AssertionError(f"filesystem write attempted during voice capture: {a}")
    monkeypatch.setattr(builtins, "open", guarded_open)

    wake = FakeWakeModel(trigger_on={1})
    try:
        run_voice_daemon(temp_backlog, mic=FakeMic(), wake_model=wake,
                         whisper_model=FakeWhisperModel(), refractory_seconds=0,
                         max_iterations=1)
    finally:
        monkeypatch.setattr(builtins, "open", real_open)

    assert temp_backlog.pending_tasks()   # confirms the flow actually ran, not skipped


# --- resilience: one bad cycle must not kill the daemon --------------------------

def test_a_transcription_failure_is_logged_and_the_daemon_keeps_going(temp_backlog):
    class ExplodingWhisper:
        def transcribe(self, audio, language="en"):
            raise RuntimeError("simulated whisper crash")

    wake = FakeWakeModel(trigger_on={1, 3})
    mic = FakeMic()

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=ExplodingWhisper(), refractory_seconds=0,
                     max_iterations=3)

    assert wake.calls == 3                  # loop survived past the crash
    assert temp_backlog.read_voice_status()["status"] in ("error", "listening")


# --- backlog.py: independent writers on one shared row ---------------------------

def test_voice_status_and_mute_are_independent_writers(temp_backlog):
    temp_backlog.write_voice_status("listening")
    temp_backlog.set_voice_muted(True)
    row = temp_backlog.read_voice_status()
    assert row["status"] == "listening"     # mute write didn't clobber status
    assert row["muted"] == 1

    temp_backlog.write_voice_status("error")
    row = temp_backlog.read_voice_status()
    assert row["status"] == "error"
    assert row["muted"] == 1                # status write didn't clobber mute


def test_is_voice_muted_defaults_false_before_any_write(temp_backlog):
    assert temp_backlog.is_voice_muted() is False
    assert temp_backlog.read_voice_status() is None
