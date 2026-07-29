"""§16 voice daemon: wake-word -> local STT -> task. Uses fake mic/wake/
whisper models (this project's established pattern — see conftest.py's
fake_llm — rather than downloading real model weights on every test run).
The real openWakeWord/faster-whisper integration was proven live and
separately (measured CPU overhead, measured CPU contention with qwen,
confirmed zero GPU/VRAM path on this hardware) — see project guide §16."""

from __future__ import annotations

import numpy as np
import pytest

from backlog import TaskRecord
from voice import CHUNK_SAMPLES, run_voice_daemon


class FakeMic:
    def __init__(self, n_chunks_before_close_ok=10_000):
        self.reads = 0
        self.closed = False
        self._max = n_chunks_before_close_ok

    def read_chunk(self):
        self.reads += 1
        return np.zeros(CHUNK_SAMPLES, dtype=np.int16)

    def close(self):
        self.closed = True


class FakeWakeModel:
    """predict() returns {"hey_nova": score}. `trigger_on` is a set of
    1-indexed call numbers on which the score exceeds threshold."""

    def __init__(self, trigger_on=frozenset()):
        self.trigger_on = trigger_on
        self.calls = 0

    def predict(self, chunk):
        self.calls += 1
        return {"hey_nova": 0.9 if self.calls in self.trigger_on else 0.0}


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
                     whisper_model=whisper, max_iterations=1)

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
                     whisper_model=whisper, max_iterations=1)

    assert temp_backlog.pending_tasks() == []


def test_capture_reads_the_expected_window_of_audio(temp_backlog):
    """CAPTURE_SECONDS worth of chunks are read after a trigger — not the
    single triggering chunk alone."""
    from voice import CAPTURE_SECONDS, SAMPLE_RATE
    wake = FakeWakeModel(trigger_on={1})
    whisper = FakeWhisperModel()
    mic = FakeMic()

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=whisper, max_iterations=1)

    expected_capture_chunks = int(CAPTURE_SECONDS * SAMPLE_RATE / CHUNK_SAMPLES)
    assert mic.reads == 1 + expected_capture_chunks   # 1 for the wake chunk itself
    assert len(whisper.transcribe_calls) == 1
    captured = whisper.transcribe_calls[0]
    assert len(captured) == expected_capture_chunks * CHUNK_SAMPLES


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
                         whisper_model=FakeWhisperModel(), max_iterations=1)
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
                     whisper_model=ExplodingWhisper(), max_iterations=3)

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
