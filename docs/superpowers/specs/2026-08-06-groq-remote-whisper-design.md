# Groq 遠端 Whisper Backend — Design

Date: 2026-08-06
Status: Approved (design), pending implementation plan

## Goal

在既有 `backends.py` 統一 ASR backend protocol 裡加一個新引擎:透過 Groq API 打
`whisper-large-v3` / `whisper-large-v3-turbo`。跟現有本地引擎(mlx-whisper /
Qwen3-ASR / FireRed)走同一套 `make_backend(model, language)` factory,同一個
`callable(pcm_bytes | audio_path) -> [{start, end, text}]`(秒) contract,所以
live 即時辨識、上傳頁、重新辨識三處都能直接選它,不需要另開一套流程。

## Non-goals

- 不做語者分離/聲紋(Groq 只回文字,現有 diarization 管線不變)。
- 不做本地模型下載管理 UI(`_SUPPORTED` 清單)——Groq 沒有權重要下載。
- 不做完整 quota dashboard / 用量統計 UI,只做 stderr 層級的低額度警告。
- 不改 `live.py` 的 VAD / AdaptiveBackend 邏輯——沿用既有降級/fallback 機制,
  Groq 只是 chain 裡新增的一個 tier。
- 不做跨呼叫的音訊緩衝合併(會打亂 live 逐句對應時間軸,風險大於收益)。

## Groq API 摘要(已查證官方文件)

- Endpoint: `POST https://api.groq.com/openai/v1/audio/transcriptions`
- 認證: `Authorization: Bearer $GROQ_API_KEY`
- 支援格式含 wav,音訊會被降到 16kHz mono(跟我們內部格式一致,免轉檔)
- `response_format=verbose_json` 回傳 segments,欄位跟 whisper 標準一致:
  `start`, `end`, `text`, `avg_logprob`, `compression_ratio`, `no_speech_prob`
- 免費層限制(依 model,whisper-large-v3 / turbo 都是):
  RPM 20、RPD 2000、ASH(音訊秒/小時)7200、ASD 28800
- **Minimum billed duration: 10 秒/次** ——短於 10 秒的音訊一樣算 10 秒
- 檔案大小上限 25MB(免費層),遠大於我們單次窗口

RPM=20 是 live 場景最緊的限制:活躍對話很容易單分鐘打超過 20 次,需要節流。
ASH/ASD 相對寬裕,batch 端每窗 ~30 秒天然超過 10 秒門檻,不需額外處理。

## 元件設計

### 1. Model id / routing

新增兩個 model id:
- `groq-whisper-large-v3`(最準,標示「準」)
- `groq-whisper-large-v3-turbo`(較快,標示「快」)

`backends.route()` 加一支:
```python
if m.startswith("groq-"):
    return "groq"
```
`make_backend()` 對應分支呼叫 `groq_backend(model, language)`,回傳值直接是
input-agnostic 的 callable(不需要 `_takes_bytes`/`_takes_path` 額外包一層,
backend 內部自己吃 bytes 也吃 path,跟 `qwen3_mlx_backend` 同款寫法)。

### 2. `groq_backend(model, language=None)`(新函式,backends.py)

行為:
- 建構時(呼叫這個 factory 的當下,不是每次呼叫時)檢查
  `os.environ.get("GROQ_API_KEY")`,沒有就直接 `raise RuntimeError`,清楚訊息
  ——這是設定問題,不該讓每個 live window 靜默失敗才發現。
- 呼叫時輸入正常化:bytes 直接用;path 走 `_pcm_bytes()`(既有 helper)轉成
  16k mono PCM bytes。
- 音訊太短(<3200 bytes,~0.1s)直接 `return []`,不浪費一次配額。
- **RPM 節流**:模組層級維護一個 per-model 的 sliding-window 計數器(過去 60
  秒的呼叫時間戳,`collections.deque`)。呼叫前檢查:過去 60 秒次數已經逼近
  上限(留安全邊界,例如 >=18 次/60s)就直接 `return []` + log stderr,不真
  的送出去等 429。這面跟現有 `ane_live_backend` 的 busy-skip 哲學一致——
  「skip 掉,音訊還在,之後重新辨識救得回來」。
- pcm bytes -> wav(復用 `recorder.pcm_to_wav`,16k mono)-> multipart POST,
  `model` 參數用 model id 去掉 `groq-` 前綴,`response_format=verbose_json`,
  `language`(若有指定)一併帶上。
- httpx timeout 20s。
- 錯誤處理(網路例外、timeout、非 2xx):log stderr(含 status code /
  Retry-After 若有),`return []`——絕不 raise,跟其他 backend 一致的
  fail-soft 原則,一個爛窗口不能弄死整個 live session。
- 成功回應:讀 `x-ratelimit-remaining-*` 系列 header(哪些存在就讀哪些,
  Groq 可能只給 requests 維度不給 audio-seconds 維度),數值低於安全門檻才
  印一行 stderr 警告;平常不印,不洗版 log。
- segments 映射成 `[{start, end, text}]`,套用跟 `mlx_whisper_backend` 一樣
  的過濾條件(`no_speech_prob<0.6`, `compression_ratio<2.4`,
  `avg_logprob>-1.0`)——欄位名一致,邏輯直接重用同樣的門檻常數。

### 3. `.env` 載入

app.py 啟動時加一個小函式:讀 repo 根目錄 `.env`(存在才讀),逐行
`KEY=VALUE` set 進 `os.environ`(僅當該 KEY 尚未存在於環境變數時,不覆蓋外部
已設定的值)。不引入 `python-dotenv`,幾行純手刻即可。

### 4. UI 三處 `<select>` 都加 Groq optgroup

- live 頁 model dropdown
- 上傳頁 model dropdown
- 重新辨識(re-transcribe)dropdown

optgroup label:`☁️ Groq API · 遠端`,兩個 option(`groq-whisper-large-v3`
標「最準·遠端」,`groq-whisper-large-v3-turbo` 標「快·遠端」)。不進
`_SUPPORTED`(那份是本地下載管理清單)。

## 測試

TDD:mock httpx(`respx` 或手動 monkeypatch),覆蓋:
- 正常 verbose_json 回應 -> 正確映射 segments
- 空/靜音回應(no text)-> `[]`
- 429 -> log + `[]`,不 raise
- timeout -> log + `[]`,不 raise
- RPM 節流:模擬過去 60 秒已達門檻次數 -> 直接 `[]`,不觸發實際 HTTP call
- 缺 `GROQ_API_KEY` -> `make_backend()` 建構時 raise

跟現有 `backends.py` 測試檔案風格一致(如果尚未有 `test_backends.py`,新建一個)。
