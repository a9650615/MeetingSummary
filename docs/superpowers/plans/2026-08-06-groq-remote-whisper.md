# Groq Remote Whisper Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Groq-hosted remote Whisper ASR backend (`whisper-large-v3` /
`whisper-large-v3-turbo`) that plugs into the existing unified backend
protocol in `backends.py`, selectable from the live, and re-transcribe model
dropdowns.

**Architecture:** One new factory function `groq_backend(model, language)` in
`backends.py`, dispatched from the existing `route()`/`make_backend()`
machinery exactly like every other engine (chatllm, qwen3mlx, firered, …). It
POSTs to Groq's OpenAI-compatible `/openai/v1/audio/transcriptions` endpoint
and maps `verbose_json` segments straight into the same
`[{start, end, text}]` shape every backend already returns. A tiny
`_load_dotenv()` in `app.py` loads `GROQ_API_KEY` from the repo-root `.env`
at startup. Two existing HTML `<select>` blocks in `app.py` (live page,
re-transcribe panel) get a new optgroup pointing at the two model ids.

**Tech Stack:** Python, `httpx` (already a project dependency — no new
package), FastAPI, pytest + `monkeypatch`.

## Global Constraints

- No new dependencies — reuse `httpx` (already in `requirements.txt`).
- Do not modify `live.py` — Groq is just another tier in the existing
  `AdaptiveBackend` fallback chain; no special-casing needed there.
- No quota dashboard/UI — only a stderr warning when quota is running low.
- Fail-soft on every network error (bad response, timeout, connection error):
  log to stderr, `return []`. Never raise from inside a call — one bad window
  must not kill a live session (same rule `mlx_whisper_backend` follows).
- The one exception: a missing `GROQ_API_KEY` raises `RuntimeError` at
  `groq_backend()` construction time (config problem, not transient).
- Client-side RPM guard: skip (don't send) once a model has made
  `_GROQ_RPM_SAFE = 18` calls in the trailing 60 seconds (Groq's free-tier cap
  is 20 requests/min per model). Skipped audio is not lost — a later
  re-transcribe can still pick it up.
- Segment filtering reuses the exact thresholds `mlx_whisper_backend` already
  uses: `no_speech_prob < 0.6`, `compression_ratio < 2.4`, `avg_logprob > -1.0`.
- Audio shorter than 3200 bytes (~0.1s) never triggers a call.

---

### Task 1: `groq_backend()` + routing in `backends.py`

**Files:**
- Modify: `backends.py:10-27` (`route()`), `backends.py:191-192` (insert new
  code after `_takes_path`, before `make_backend`), `backends.py:229-232`
  (`make_backend()`)
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces: `backends.route(model: str) -> str` — now also returns `"groq"`
  for any id starting with `"groq-"`.
- Produces: `backends.groq_backend(model: str, language: str | None = None) ->
  Callable[[bytes | str], list[dict]]` — the returned callable takes PCM
  bytes, a `.pcm` path, or any container path (via `_pcm_bytes`, already
  defined at `backends.py:154`), and returns `[{"start": float, "end": float,
  "text": str}, ...]` in seconds.
- Produces: `backends._groq_can_call(model: str) -> bool` — RPM guard,
  records the attempt as a side effect when it returns `True`.
- Produces: `backends._GROQ_MODELS: dict[str, str]` — maps our synthetic ids
  (`groq-whisper-large-v3`, `groq-whisper-large-v3-turbo`) to the literal
  Groq model names (`whisper-large-v3`, `whisper-large-v3-turbo`).
- Consumes: `backends._pcm_bytes(audio) -> bytes` (existing, line 154),
  `recorder.pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1) -> bytes`
  (existing, `recorder.py:81`).

- [ ] **Step 1: Write the failing routing test**

Add to `tests/test_backends.py` (append to `test_route_qwen3_vs_whisper` or as
a new test — add as a new test to keep it independently reviewable):

```python
def test_route_groq():
    assert backends.route("groq-whisper-large-v3") == "groq"
    assert backends.route("groq-whisper-large-v3-turbo") == "groq"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py::test_route_groq -v`
Expected: FAIL — `assert 'whisper' == 'groq'` (route() doesn't know "groq-" yet).

- [ ] **Step 3: Add the routing branch**

In `backends.py`, inside `route()` (currently lines 10-27), add the new
branch right after the `ane-` check:

```python
def route(model):
    """Engine for a model id. chatllm 1.7B GGUF -> chatllm (.cpp/Metal, persistent
    binding); q4-k-m -> femelo .cpp (Metal); other qwen3-asr -> transformers;
    whisper(-mlx) is the Apple-native live engine."""
    m = model.lower()
    if m.startswith("ane-"):  # ANE (Neural Engine) via the `speech` CLI
        return "ane"
    if m.startswith("groq-"):  # Groq's remote OpenAI-compatible whisper API
        return "groq"
    if model == "qwen3-asr-1.7b":
        return "chatllm"
    if "qwen3-asr" in m and "mlx-community" in m:  # MLX-native Qwen3-ASR (Metal, fast)
        return "qwen3mlx"
    if "q4-k-m" in m:
        return "qwen3cpp"
    if "qwen3-asr" in m:
        return "qwen3"
    if m == "firered" or "fire-red" in m:  # FireRedASR-AED via sherpa-onnx (CPU, batch)
        return "firered"
    return "whisper"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py::test_route_groq -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for missing API key**

Append to `tests/test_backends.py`:

```python
def test_groq_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        backends.groq_backend("groq-whisper-large-v3")
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_backends.py::test_groq_backend_requires_api_key -v`
Expected: FAIL — `AttributeError: module 'backends' has no attribute 'groq_backend'`

- [ ] **Step 7: Implement `groq_backend()`, `_groq_can_call()`, `_groq_warn_quota()`**

Insert this whole block into `backends.py` right after `_takes_path()` ends
(after the current line 191 `return _run`, i.e. right before the blank lines
leading into `def make_backend(model, language=None):` at line 194):

```python
# Groq's OpenAI-compatible transcription endpoint. whisper-large-v3 = most
# accurate; turbo = faster, still solid. Free tier caps at 20 requests/min
# per model — _GROQ_RPM_SAFE leaves a margin so we skip client-side instead
# of burning quota on a call that would just come back 429.
_GROQ_MODELS = {
    "groq-whisper-large-v3": "whisper-large-v3",
    "groq-whisper-large-v3-turbo": "whisper-large-v3-turbo",
}
_GROQ_RPM_SAFE = 18
_GROQ_STATE = {"calls": {}, "lock": None}  # model -> deque of recent call times
_GROQ_QUOTA_WARN_REQUESTS = 5
_GROQ_QUOTA_WARN_TOKENS = 500


def _groq_can_call(model):
    """Client-side RPM guard: True (and records the attempt) if this model has
    made fewer than _GROQ_RPM_SAFE calls in the trailing 60s; False if a call
    right now would risk a 429. Skipping is safe — the audio isn't lost, it's
    just not sent to Groq for this window (re-transcribe can retry later)."""
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415
    from collections import deque  # noqa: PLC0415
    if _GROQ_STATE["lock"] is None:
        _GROQ_STATE["lock"] = threading.Lock()
    with _GROQ_STATE["lock"]:
        now = time.monotonic()
        dq = _GROQ_STATE["calls"].setdefault(model, deque())
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= _GROQ_RPM_SAFE:
            return False
        dq.append(now)
        return True


def _groq_warn_quota(resp):
    """Low-quota heads-up only. Groq puts remaining RPM/TPM in response
    headers on every call; we stay quiet unless it's actually running low, so
    a live session's stderr isn't spammed once per utterance."""
    import sys  # noqa: PLC0415
    try:
        rem_req = resp.headers.get("x-ratelimit-remaining-requests")
        if rem_req is not None and float(rem_req) < _GROQ_QUOTA_WARN_REQUESTS:
            print(f"groq quota low: {rem_req} requests remaining", file=sys.stderr)
        rem_tok = resp.headers.get("x-ratelimit-remaining-tokens")
        if rem_tok is not None and float(rem_tok) < _GROQ_QUOTA_WARN_TOKENS:
            print(f"groq quota low: {rem_tok} tokens remaining", file=sys.stderr)
    except (TypeError, ValueError):
        pass


def groq_backend(model, language=None):
    """Remote Whisper via Groq's OpenAI-compatible transcription endpoint.
    callable(pcm_bytes | audio_path) -> [{start, end, text}] (seconds) — same
    contract as every other backend. Network call: fail-soft on any error
    (log + return []), matching mlx_whisper_backend's "one bad window can't
    kill a session" rule. Raises at construction time (not per-call) if
    GROQ_API_KEY is missing — that's a config problem, not a transient one."""
    import os  # noqa: PLC0415
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set — put it in .env or export it before "
            "selecting a groq-* model")
    groq_model = _GROQ_MODELS.get(model, model)

    def _run(audio):
        import sys  # noqa: PLC0415

        import httpx  # noqa: PLC0415

        import recorder  # noqa: PLC0415
        pcm = _pcm_bytes(audio)
        if len(pcm) < 3200:  # <0.1s -> not worth a call
            return []
        if not _groq_can_call(model):
            print(f"groq {model}: near RPM limit, skipping window "
                  f"(audio kept for re-transcribe)", file=sys.stderr)
            return []
        wav = recorder.pcm_to_wav(pcm, sample_rate=16000, channels=1)
        data = {"model": groq_model, "response_format": "verbose_json"}
        if language:
            data["language"] = language
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", wav, "audio/wav")},
                data=data, timeout=20.0)
        except httpx.HTTPError as e:
            print(f"groq request failed: {e}", file=sys.stderr)
            return []
        if resp.status_code != 200:
            print(f"groq error {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
            return []
        _groq_warn_quota(resp)
        try:
            segments = resp.json().get("segments", [])
        except ValueError:
            return []
        return [{"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in segments
                if s.get("no_speech_prob", 0) < 0.6
                and s.get("compression_ratio", 0) < 2.4
                and s.get("avg_logprob", 0) > -1.0]

    return _run
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_backends.py::test_groq_backend_requires_api_key -v`
Expected: PASS

- [ ] **Step 9: Write the failing test for a successful transcription**

Append to `tests/test_backends.py`:

```python
class _FakeResp:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json


def test_groq_backend_maps_segments_and_filters_hallucinations(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    seen = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        seen["url"] = url
        seen["data"] = data
        return _FakeResp(json_data={"segments": [
            {"start": 0.0, "end": 1.2, "text": "你好",
             "no_speech_prob": 0.1, "compression_ratio": 1.5, "avg_logprob": -0.3},
            {"start": 1.2, "end": 2.0, "text": "靜音噪音",
             "no_speech_prob": 0.9, "compression_ratio": 1.2, "avg_logprob": -0.1},
        ]})

    monkeypatch.setattr(httpx, "post", fake_post)
    run = backends.groq_backend("groq-whisper-large-v3")
    out = run(b"\x00\x01" * 4000)  # 8000 bytes, > 3200 floor
    assert out == [{"start": 0.0, "end": 1.2, "text": "你好"}]
    assert seen["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert seen["data"]["model"] == "whisper-large-v3"
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/test_backends.py::test_groq_backend_maps_segments_and_filters_hallucinations -v`
Expected: PASS

- [ ] **Step 11: Write the failing tests for short audio, HTTP errors, and RPM throttling**

Append to `tests/test_backends.py`:

```python
def test_groq_backend_skips_tiny_audio_without_calling_api(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()

    def fail_post(*a, **k):
        raise AssertionError("should not call the API for tiny audio")

    monkeypatch.setattr(httpx, "post", fail_post)
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01") == []  # 2 bytes, well under the 3200 floor


def test_groq_backend_non_200_returns_empty_not_raise(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: _FakeResp(status_code=429, text="rate limited"))
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01" * 4000) == []


def test_groq_backend_network_error_returns_empty_not_raise(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()

    def raise_timeout(*a, **k):
        raise httpx.ConnectTimeout("boom")

    monkeypatch.setattr(httpx, "post", raise_timeout)
    run = backends.groq_backend("groq-whisper-large-v3")
    assert run(b"\x00\x01" * 4000) == []


def test_groq_rpm_guard_skips_past_the_safe_margin(monkeypatch):
    import httpx
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backends._GROQ_STATE["calls"].clear()
    calls = {"n": 0}

    def counting_post(*a, **k):
        calls["n"] += 1
        return _FakeResp(json_data={"segments": []})

    monkeypatch.setattr(httpx, "post", counting_post)
    run = backends.groq_backend("groq-whisper-large-v3")
    for _ in range(25):  # well past _GROQ_RPM_SAFE (18)
        run(b"\x00\x01" * 4000)
    assert calls["n"] == backends._GROQ_RPM_SAFE
```

- [ ] **Step 12: Run tests to verify they fail (or pass) as expected before the fix**

Run: `pytest tests/test_backends.py -k groq -v`
Expected: `test_groq_backend_skips_tiny_audio_without_calling_api`,
`test_groq_backend_non_200_returns_empty_not_raise`,
`test_groq_backend_network_error_returns_empty_not_raise` should already PASS
(the implementation from Step 7 already handles these paths) —
`test_groq_rpm_guard_skips_past_the_safe_margin` should already PASS too,
since `_groq_can_call` was implemented in Step 7. This step is a
confirmation run, not a red step — if anything fails, fix the Step 7
implementation before moving on.

- [ ] **Step 13: Wire `groq` into `make_backend()`**

In `backends.py`, inside `make_backend()` (currently around lines 194-232),
add the new branch. The relevant tail of the function currently reads:

```python
    if r == "firered":
        return _takes_path(firered_batch_backend(model, language))
    import asr  # noqa: PLC0415
    return asr.mlx_whisper_backend(model, language)  # bytes, .pcm or container
```

Change it to:

```python
    if r == "firered":
        return _takes_path(firered_batch_backend(model, language))
    if r == "groq":
        return groq_backend(model, language)
    import asr  # noqa: PLC0415
    return asr.mlx_whisper_backend(model, language)  # bytes, .pcm or container
```

- [ ] **Step 14: Write the failing integration test for `make_backend`**

Append to `tests/test_backends.py`:

```python
def test_make_backend_dispatches_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    run = backends.make_backend("groq-whisper-large-v3-turbo")
    assert callable(run)
```

- [ ] **Step 15: Run test to verify it passes**

Run: `pytest tests/test_backends.py::test_make_backend_dispatches_groq -v`
Expected: PASS

- [ ] **Step 16: Run the full backends test suite**

Run: `pytest tests/test_backends.py -v`
Expected: all PASS, no regressions in the pre-existing tests.

- [ ] **Step 17: Commit**

```bash
git add backends.py tests/test_backends.py
git commit -m "feat(backends): add Groq remote whisper backend with RPM throttling"
```

---

### Task 2: `.env` loader in `app.py`

**Files:**
- Modify: `app.py:1-46` (add `_load_dotenv()`, call it next to `_ensure_tool_path()`)
- Test: `tests/test_app.py`

**Interfaces:**
- Produces: `app._load_dotenv(path: str | None = None) -> None` — reads
  `KEY=VALUE` lines from `path` (default: `.env` next to `app.py`) into
  `os.environ`, skipping blanks/`#`-comments and any key already present in
  the environment.
- Consumes: none from Task 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py`:

```python
def test_load_dotenv_sets_missing_keys_without_overriding(tmp_path, monkeypatch):
    import app
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=from-dotenv\nALREADY_SET=from-dotenv\n# comment\n\n")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    app._load_dotenv(str(env_file))
    assert os.environ["GROQ_API_KEY"] == "from-dotenv"
    assert os.environ["ALREADY_SET"] == "from-shell"  # .env never overrides
```

This test needs `import os` at the top of `tests/test_app.py` if not already
present — check first with `grep -n "^import os" tests/test_app.py`; add it
if missing.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_load_dotenv_sets_missing_keys_without_overriding -v`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_load_dotenv'`

- [ ] **Step 3: Implement `_load_dotenv()`**

In `app.py`, right after `_ensure_tool_path()` (currently lines 33-42) and
its call at line 45, add:

```python
def _load_dotenv(path=None):
    """Read .env (repo root, gitignored) into os.environ for local secrets
    (e.g. GROQ_API_KEY) — never overrides a value already set externally."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


_ensure_tool_path()
_load_dotenv()
```

Replace the existing standalone `_ensure_tool_path()` call (line 45) with
the two-line `_ensure_tool_path()` / `_load_dotenv()` pair above (function
definition goes directly below `_ensure_tool_path`'s own definition, call
goes where the old single call was).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py::test_load_dotenv_sets_missing_keys_without_overriding -v`
Expected: PASS

- [ ] **Step 5: Run the full app test suite**

Run: `pytest tests/test_app.py -v`
Expected: all PASS — confirms `_load_dotenv()` at import/startup time doesn't
break anything (it's a no-op whenever `.env` doesn't exist at the check path,
and in this repo's real `.env` it only ever adds `GROQ_API_KEY`, never
touches a key already set).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat(app): load GROQ_API_KEY from .env at startup"
```

---

### Task 3: Expose Groq models in the live and re-transcribe dropdowns

**Files:**
- Modify: `app.py:726` (`_LIVE_BODY`'s `<select id=model>`), `app.py:2668-2689`
  (`<select id=remodel>` inside the meeting-detail page body) — line numbers
  are pre-Task-2; Task 2 adds ~10 lines near the top of `app.py`, shifting
  everything after it. Use the exact HTML block quoted in each step below to
  locate the edit, not the line number.
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: none new — `body.model` (live `/models` POST, already
  `ModelIn.live: str`) and the re-transcribe `model` field
  (`/meetings/{mid}/transcribe/start`) already pass whatever string the
  dropdown sends straight into `backends.make_backend(model, language)` with
  no allowlist, so no other code changes are needed for these new ids to
  work end-to-end once Task 1 is done.

- [ ] **Step 1: Write the failing test for the live page dropdown**

Append to `tests/test_app.py`:

```python
def test_live_page_offers_groq_models(tmp_path):
    c, _ = make_client(tmp_path)
    html = c.get("/live").text
    assert "groq-whisper-large-v3-turbo" in html
    assert "groq-whisper-large-v3" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_live_page_offers_groq_models -v`
Expected: FAIL — neither id is in the page yet.

- [ ] **Step 3: Add the Groq optgroup to `_LIVE_BODY`**

In `app.py`, the live model `<select>` currently reads (lines 725-742):

```python
      <label class=fld>即時模型
        <select id=model>
          <optgroup label="🔧 .cpp · Metal">
          <option value="qwen3-asr-0.6b-q4-k-m">Qwen3-ASR 0.6B(預設·快)</option>
          <option value="qwen3-asr-1.7b">Qwen3-ASR 1.7B(chatllm·慢·備用)</option>
          </optgroup>
          <optgroup label="⚡ MLX · Metal/GPU">
          <option value="mlx-community/Qwen3-ASR-1.7B-8bit">Qwen3-ASR 1.7B(準·快)</option>
          <option value="mlx-community/whisper-small-mlx-q4">whisper small-q4(快·省)</option>
          <option value="mlx-community/whisper-large-v3-turbo-q4">whisper turbo-q4(較準)</option>
          <option value="mlx-community/whisper-large-v3-turbo">whisper turbo(最準·較吃)</option>
          <option value="mlx-community/whisper-base-mlx-q4">whisper base-q4(更快)</option>
          <option value="mlx-community/whisper-tiny-mlx-q4">whisper tiny-q4(最省)</option>
          </optgroup>
          <optgroup label="🐢 transformers · 慢">
          <option value="Qwen/Qwen3-ASR-0.6B">Qwen3-ASR 0.6B</option>
          </optgroup>
        </select></label>
```

Change the closing part (right before `</select></label>`) to add a new
optgroup:

```python
      <label class=fld>即時模型
        <select id=model>
          <optgroup label="🔧 .cpp · Metal">
          <option value="qwen3-asr-0.6b-q4-k-m">Qwen3-ASR 0.6B(預設·快)</option>
          <option value="qwen3-asr-1.7b">Qwen3-ASR 1.7B(chatllm·慢·備用)</option>
          </optgroup>
          <optgroup label="⚡ MLX · Metal/GPU">
          <option value="mlx-community/Qwen3-ASR-1.7B-8bit">Qwen3-ASR 1.7B(準·快)</option>
          <option value="mlx-community/whisper-small-mlx-q4">whisper small-q4(快·省)</option>
          <option value="mlx-community/whisper-large-v3-turbo-q4">whisper turbo-q4(較準)</option>
          <option value="mlx-community/whisper-large-v3-turbo">whisper turbo(最準·較吃)</option>
          <option value="mlx-community/whisper-base-mlx-q4">whisper base-q4(更快)</option>
          <option value="mlx-community/whisper-tiny-mlx-q4">whisper tiny-q4(最省)</option>
          </optgroup>
          <optgroup label="🐢 transformers · 慢">
          <option value="Qwen/Qwen3-ASR-0.6B">Qwen3-ASR 0.6B</option>
          </optgroup>
          <optgroup label="☁️ Groq API · 遠端">
          <option value="groq-whisper-large-v3-turbo">Groq whisper turbo(快·遠端)</option>
          <option value="groq-whisper-large-v3">Groq whisper large-v3(最準·遠端)</option>
          </optgroup>
        </select></label>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py::test_live_page_offers_groq_models -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for the re-transcribe dropdown**

Append to `tests/test_app.py`:

```python
def test_meeting_detail_offers_groq_remodel(tmp_path):
    c, store = make_client(tmp_path)
    mid = store.create_meeting("m", 1.0, "zh-TW")
    html = c.get(f"/m/{mid}").text
    assert "groq-whisper-large-v3-turbo" in html
    assert "groq-whisper-large-v3" in html
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_meeting_detail_offers_groq_remodel -v`
Expected: FAIL — neither id is in the page yet.

- [ ] **Step 7: Add the Groq optgroup to the re-transcribe `<select>`**

In `app.py`, the re-transcribe select currently reads (lines 2668-2689):

```python
        "<select id=remodel>"
        + ("<optgroup label='🧠 NPU · ANE 省電'>"
           "<option value='ane-qwen3-0.6b-hybrid'>Qwen3-ASR 0.6B(省電·快)</option>"
           "</optgroup>" if ane_on else "") +
        "<optgroup label='⚡ MLX · Metal/GPU'>"
        "<option value='mlx-community/whisper-large-v3-turbo-q4'>whisper turbo-q4(準·省)</option>"
        "<option value='mlx-community/whisper-large-v3-mlx'>whisper large-v3(最準·吃)</option>"
        "<option value='mlx-community/whisper-small-mlx-q4'>whisper small-q4(快)</option>"
        "<option value='mlx-community/Qwen3-ASR-1.7B-8bit'>Qwen3-ASR 1.7B(準·快)</option>"
        "</optgroup>"
        "<optgroup label='🎯 最高準度 · CPU（慢）'>"
        "<option value='firered'>FireRedASR-AED-L(最準·CPU慢·首次下載1GB)</option>"
        "</optgroup>"
        "<optgroup label='🔧 .cpp · Metal'>"
        "<option value='qwen3-asr-0.6b-q4-k-m'>Qwen3-ASR 0.6B(快)</option>"
        "<option value='qwen3-asr-1.7b'>Qwen3-ASR 1.7B(chatllm·慢·備用)</option>"
        "</optgroup>"
        "<optgroup label='🐢 transformers · 慢'>"
        "<option value='Qwen/Qwen3-ASR-0.6B'>Qwen3-ASR 0.6B</option>"
        "<option value='Qwen/Qwen3-ASR-1.7B'>Qwen3-ASR 1.7B(很慢)</option>"
        "</optgroup>"
        "</select>"
```

Add a new optgroup right before the closing `"</select>"` line:

```python
        "<select id=remodel>"
        + ("<optgroup label='🧠 NPU · ANE 省電'>"
           "<option value='ane-qwen3-0.6b-hybrid'>Qwen3-ASR 0.6B(省電·快)</option>"
           "</optgroup>" if ane_on else "") +
        "<optgroup label='⚡ MLX · Metal/GPU'>"
        "<option value='mlx-community/whisper-large-v3-turbo-q4'>whisper turbo-q4(準·省)</option>"
        "<option value='mlx-community/whisper-large-v3-mlx'>whisper large-v3(最準·吃)</option>"
        "<option value='mlx-community/whisper-small-mlx-q4'>whisper small-q4(快)</option>"
        "<option value='mlx-community/Qwen3-ASR-1.7B-8bit'>Qwen3-ASR 1.7B(準·快)</option>"
        "</optgroup>"
        "<optgroup label='🎯 最高準度 · CPU（慢）'>"
        "<option value='firered'>FireRedASR-AED-L(最準·CPU慢·首次下載1GB)</option>"
        "</optgroup>"
        "<optgroup label='🔧 .cpp · Metal'>"
        "<option value='qwen3-asr-0.6b-q4-k-m'>Qwen3-ASR 0.6B(快)</option>"
        "<option value='qwen3-asr-1.7b'>Qwen3-ASR 1.7B(chatllm·慢·備用)</option>"
        "</optgroup>"
        "<optgroup label='🐢 transformers · 慢'>"
        "<option value='Qwen/Qwen3-ASR-0.6B'>Qwen3-ASR 0.6B</option>"
        "<option value='Qwen/Qwen3-ASR-1.7B'>Qwen3-ASR 1.7B(很慢)</option>"
        "</optgroup>"
        "<optgroup label='☁️ Groq API · 遠端'>"
        "<option value='groq-whisper-large-v3-turbo'>Groq whisper turbo(快·遠端)</option>"
        "<option value='groq-whisper-large-v3'>Groq whisper large-v3(最準·遠端)</option>"
        "</optgroup>"
        "</select>"
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_app.py::test_meeting_detail_offers_groq_remodel -v`
Expected: PASS

- [ ] **Step 9: Run the full test suite**

Run: `pytest -q`
Expected: all PASS, no regressions anywhere in the app.

- [ ] **Step 10: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat(app): expose Groq remote whisper models in live + re-transcribe dropdowns"
```

---

## Note on the approved spec

The approved spec (`docs/superpowers/specs/2026-08-06-groq-remote-whisper-design.md`)
says the Groq optgroup should appear in "live 頁、上傳頁、重新辨識" (three
places). Checking `app.py`, the upload form (`/`, the `<form
action="/ingest">` block) has **no model `<select>` at all** — auto-upload
always resolves its ASR model automatically via `modelprofile.py`'s hardware
detection, with no per-upload user choice. There is no third dropdown to add
Groq to. This plan wires the two dropdowns that actually exist (live,
re-transcribe); the upload path already benefits from Groq indirectly if a
user manually re-transcribes an uploaded meeting afterward.
