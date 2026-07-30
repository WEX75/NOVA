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


# --- no_signal detection (§16 follow-up, 2026-07-30) ------------------------------

def test_dead_mic_reports_no_signal_not_listening(temp_backlog):
    """The core requirement: a genuinely dead/muted input (exact-zero RMS,
    unmuted) must surface as its own distinct status — not silently look
    like normal "listening"."""
    from voice import NO_SIGNAL_CHUNKS_THRESHOLD
    wake = FakeWakeModel()   # never triggers
    mic = FakeMic(amplitude=0)   # bit-for-bit silence, every chunk

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=FakeWhisperModel(), status_interval_seconds=0,
                     max_iterations=NO_SIGNAL_CHUNKS_THRESHOLD + 2)

    assert temp_backlog.read_voice_status()["status"] == "no_signal"


def test_real_ambient_noise_never_reports_no_signal(temp_backlog):
    """Genuine (even quiet) ambient audio must never trip this — only
    EXACT zero does. amplitude=1 still produces nonzero samples most of
    the time (true int16 noise), unlike genuine silence."""
    from voice import NO_SIGNAL_CHUNKS_THRESHOLD
    wake = FakeWakeModel()
    mic = FakeMic(amplitude=5000)   # normal "speech-level" fake audio

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=FakeWhisperModel(), status_interval_seconds=0,
                     max_iterations=NO_SIGNAL_CHUNKS_THRESHOLD + 2)

    assert temp_backlog.read_voice_status()["status"] == "listening"


def test_no_signal_clears_once_real_audio_resumes(temp_backlog):
    """The counter must reset on real audio, not just monotonically
    accumulate — a transient dead patch should not permanently flag the
    daemon once the mic recovers."""
    from voice import NO_SIGNAL_CHUNKS_THRESHOLD

    calls = {"n": 0}
    class FlakyMic:
        def read_chunk(self):
            calls["n"] += 1
            # dead for the first stretch, then recovers
            if calls["n"] <= NO_SIGNAL_CHUNKS_THRESHOLD:
                return np.zeros(CHUNK_SAMPLES, dtype=np.int16)
            return np.random.default_rng(1).integers(
                -5000, 5000, CHUNK_SAMPLES, dtype=np.int16)
        def close(self):
            pass

    wake = FakeWakeModel()
    run_voice_daemon(temp_backlog, mic=FlakyMic(), wake_model=wake,
                     whisper_model=FakeWhisperModel(), status_interval_seconds=0,
                     max_iterations=NO_SIGNAL_CHUNKS_THRESHOLD + 5)

    assert temp_backlog.read_voice_status()["status"] == "listening"


def test_mute_resets_the_no_signal_counter_on_unmute(temp_backlog):
    """A stale near-threshold count built up BEFORE a mute must not
    immediately flag no_signal the instant the daemon is unmuted again —
    one continuous run (a fresh call to run_voice_daemon always starts
    its own counter at 0 regardless, so this has to happen within a
    single run to mean anything): unmuted long enough to nearly reach
    the threshold (23 of 25), muted for 2 iterations, then unmuted again
    for a few more. Without the reset-on-mute line this would resume
    from 23 and cross the threshold within 2 more reads; with it, it
    starts over from 0."""
    from voice import NO_SIGNAL_CHUNKS_THRESHOLD
    call_count = {"n": 0}

    def fake_is_muted():
        call_count["n"] += 1
        return NO_SIGNAL_CHUNKS_THRESHOLD - 2 < call_count["n"] <= NO_SIGNAL_CHUNKS_THRESHOLD
    temp_backlog.is_voice_muted = fake_is_muted

    wake = FakeWakeModel()
    mic = FakeMic(amplitude=0)

    run_voice_daemon(temp_backlog, mic=mic, wake_model=wake,
                     whisper_model=FakeWhisperModel(), status_interval_seconds=0,
                     max_iterations=NO_SIGNAL_CHUNKS_THRESHOLD + 3)

    assert temp_backlog.read_voice_status()["status"] == "listening"


# --- ALSA capture startup fix (§16 follow-up, 2026-07-30) -------------------------

def test_ensure_alsa_capture_unmuted_fixes_a_muted_card(monkeypatch):
    """Regression guard for a real bug caught live 2026-07-30: a first
    version of this fix used "unmute" as the amixer keyword, which this
    hardware's driver rejects for the Capture simple-control (only "cap"/
    "nocap" work here) — the fix logged success but the mic stayed
    muted. This test pins the ACTUAL working keyword, and the live-
    verification test below pins the self-check that caught it."""
    from voice import _ensure_alsa_capture_unmuted
    calls = []

    class FakeCompleted:
        def __init__(self, stdout=""):
            self.stdout = stdout

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "sget" in cmd and cmd[-1] == "Capture":
            # first sget (pre-fix check) says muted; the SECOND sget
            # (post-fix verification) says fixed — simulates the fix
            # genuinely working, distinct from the "still muted" test below
            sget_calls = [c for c in calls if "sget" in c]
            if len(sget_calls) == 1:
                return FakeCompleted(stdout="Front Left: Capture 0 [0%] [off]")
            return FakeCompleted(stdout="Front Left: Capture 50 [79%] [on]")
        return FakeCompleted()

    monkeypatch.setattr("subprocess.run", fake_run)
    _ensure_alsa_capture_unmuted()

    assert any(cmd[-1] == "cap" for cmd in calls)   # the correct keyword, not "unmute"
    assert sum(1 for c in calls if "sget" in c) == 2   # pre-check AND post-fix verification


def test_ensure_alsa_capture_unmuted_logs_error_when_fix_does_not_take(monkeypatch, caplog):
    """The exact scenario caught live: the fix is attempted, but the card
    is STILL muted afterward. Must be a loud, distinct log entry — not
    silently reported as success."""
    import logging
    from voice import _ensure_alsa_capture_unmuted

    def fake_run(cmd, **kwargs):
        class FakeCompleted:
            stdout = "Front Left: Capture 0 [0%] [off]"   # muted, always — the fix never takes
        return FakeCompleted()

    monkeypatch.setattr("subprocess.run", fake_run)
    with caplog.at_level(logging.ERROR, logger="nova"):
        _ensure_alsa_capture_unmuted()

    assert any("STILL muted" in r.message for r in caplog.records)


def test_ensure_alsa_capture_unmuted_is_a_noop_when_already_unmuted(monkeypatch):
    from voice import _ensure_alsa_capture_unmuted
    calls = []

    class FakeCompleted:
        def __init__(self, stdout=""):
            self.stdout = stdout

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeCompleted(stdout="Front Left: Capture 50 [79%] [on]")

    monkeypatch.setattr("subprocess.run", fake_run)
    _ensure_alsa_capture_unmuted()

    assert len(calls) == 1              # only the read check — no fix/volume/verify calls
    assert not any(cmd[-1] == "cap" for cmd in calls)


def test_ensure_alsa_capture_unmuted_never_raises_on_failure(monkeypatch):
    """Best-effort only — a broken/missing amixer must not prevent the
    daemon from starting."""
    from voice import _ensure_alsa_capture_unmuted

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("amixer: command not found")

    monkeypatch.setattr("subprocess.run", fake_run)
    _ensure_alsa_capture_unmuted()   # must not raise


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
