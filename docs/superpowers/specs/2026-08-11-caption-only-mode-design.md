# 純字幕限定模式（Caption-Only）— Design

Date: 2026-08-11
Status: Approved (design), pending implementation plan

## Goal

浮動面板（floatpanel）新增一個「字幕限定」開關：開啟後只即時顯示字幕，**完全不建立
會議、不存逐字稿、不留任何音檔**。效能不足（辨識引擎跟不上、視窗被丟）時文字直接
遺失，不做任何補救——這是「不留音檔備份」的自然後果，不是要另外設計丟棄機制。

跟現有已上線的「純字幕」錄音模式（`live_mode=transcribe`）不同：那個模式仍然會建立
`meetings` 資料列並把逐字稿寫進 DB，只是不留音檔。這次要做的是更徹底的、完全
ephemeral 的第二種模式。

僅限浮動面板。web `/live` 頁面不動。

## Non-goals

- 不支援語者辨識（diarization）。固定用 side label（我/對方），不追求跨會議認人。
- 不支援斷線重連後的音訊補傳（`/native/backfill`）——沒有音檔要補。
- 不新增任何「限制佇列大小、主動丟棄舊字幕」之類的機制——遺失純粹是引擎跟不上
  的自然結果，不特別處理。
- 不改動既有 `live_mode=transcribe`（會存 DB 的純字幕模式），兩者並存、互不影響。
- 不改 web `/live` 頁面、不改設定頁既有的「錄音模式」三選一（`both`/`record`/
  `transcribe`）下拉選單。
- 不支援跨程序重啟後恢復這個模式的 session（重啟後自然變成一個新 ephemeral session，
  符合「文字可丟失」精神）。

## 現況（已核實，避免重做）

- `live_mode=transcribe`（設定頁「純字幕（不留錄音檔）」）已上線：`audio_files` 換成
  `NullSink`（`live_session.py:77`），但 `live_session.make_store_emit`
  （`live_session.py:285`）仍然對每個 final 呼叫 `store.add_transcript`/
  `store.extend_transcript`，`mid` 仍是一筆真的 `meetings` DB row。
- 浮動面板顯示字幕**主要**靠 `/ws/native-capture` 這條 websocket 直接推送
  `{"type":"interim"/"final", ...}`（`app.py` 的 `_push`/`_push_q`，
  `live_session.make_store_emit` 的 `push` 參數），Swift 端 `task.receive` 迴圈
  （`main.swift:632` 附近）直接吃這些訊息設 `self.captions`——**不是**單靠
  `/live/state` 的 HTTP 輪詢。這代表 caption-only 模式完全不用碰 DB 也能讓字幕顯示
  正常運作。
- `/live/state`（`app.py:3052`）、`/live/stop`、`/live/pause` 都只操作
  `live_active`/`live_stop`/`live_paused`（純 in-memory dict/set，key 是任意
  `mid`），不強制要求 `mid` 對應真的 DB row——只有讀 `store.get_meeting(mid)` 那幾行
  在查不到時會回傳 `None`，需要 fallback。

## 設計

### 1. 新 `mode` 值：`"caption"`

`/ws/native-capture` 現有的三種 `source`（mic/system/both）不受影響，這次只加
`mode` 的第三種值 `"caption"`（現有兩種是空字串預設走 `store.get_setting` 和
`"transcribe"`）。floatpanel 是唯一會送這個值的呼叫端，透過 URL query
`?mode=caption` 明確指定——不影響 `store.get_setting("live_mode", "both")` 這個
web /live 共用的既有設定。

### 2. Pseudo-mid：完全不建 `meetings` row

`app.py` 的 `ws_native_capture` 目前開頭就是：
```python
if session and session.isdigit() and store.get_meeting(int(session)) is not None:
    mid = int(session)
    ...
else:
    mid = store.create_meeting(title, t0, "zh-TW")
    conn_offset_ms = 0
```
`mode == "caption"` 時整段跳過，改用一個模組層級遞減計數器（例如從 `-1` 開始每次
減一）配出一個**負整數** pseudo-mid——SQLite 的 `meetings.id` 是自增正整數，負數
保證不會撞到任何真實 id。`conn_offset_ms` 固定為 0（沒有「接續同一場會議」這回事，
每個 caption-only session 都是全新的）。

`live_active[mid] = 1`、`native_sessions[mid] = {...}` 等既有 dict 原樣沿用——它們
本來就不查 DB，key 是負數也完全正常運作。`/live/stop`、`/live/pause` 因此對
caption-only session 一樣有效，不用改。

### 3. `live_session.make_caption_only_emit()`（新函式，`live_session.py`）

跟 `make_store_emit` 平行的新函式，一樣做「同說話者短暫停頓合併成一行」的邏輯，但
`prev`（上一行）的來源從 `store.last_live_row(mid, track)` 換成一個**函式內
closure 捕捉的 plain dict**（`{track: {"speaker":.., "text":.., "start_ms":..,
"end_ms":..}}`），不落地、不查 DB。合併判斷條件（`_MERGE_GAP_MS`/`_MERGE_MAX_MS`/
`_MERGE_MAX_CHARS`）原封不動照抄，維持跟現有模式一致的斷行體驗。

`push` 呼叫方式與現有 `make_store_emit` 完全相同（`{"type":"final"/"interim", ...}`
送過 websocket）——這是 floatpanel 顯示字幕的唯一資料來源，不受影響。

### 4. `flush_sessions` 加 `persist=True` 參數

`live_session.flush_sessions(sessions, tracks, mid, conn_offset_ms, store, skip=(),
persist=True)`。`persist=False` 時，收尾那段「把最後一句 partial utterance 寫進
store」的邏輯直接跳過，改成只是把 events 丟掉（沒有地方推播了，因為此時 websocket
多半已經斷線在做收尾）。`ws_native_capture` 在 `mode == "caption"` 時傳
`persist=False`。

### 5. `/live/state` 的 pseudo-mid fallback

`m = store.get_meeting(mid)` 查不到（負數 mid）時，改查一個小型模組層級 dict
`_caption_sessions: dict[int, dict]`（`ws_native_capture` 建立 session 時順手塞入
`{"title": "字幕限定", "created_at": t0}`，`_run()` 的 `finally` 區塊清掉）。
`title`/`started_at` 從這個 fallback 取；`caption`/`captions`
（`store.recent_transcripts`/`store.last_caption_end_ms`）對 pseudo-mid 就讓它們
保持空——反正 floatpanel 的字幕顯示不靠這個欄位，只有 elapsed 計時器和標題文字需要
這個 fallback。

### 6. Floatpanel UI（`main.swift`）

- Source picker（`Picker("來源", ...)`)旁加一個 icon-only 切換鈕（風格比照現有的
  暫停鈕），只在 `!m.recording` 時可切換。Tooltip：「只顯示即時字幕，不錄音、不留
  任何逐字稿或會議記錄」。
- `@Published var captionOnly = false`（`Model`），`startNativeRelay()` 組
  websocket URL 時，開啟時附加 `&mode=caption`。
- 開啟時跳過斷線緩衝/`/native/backfill` 的呼叫路徑——沒有音檔要補傳，補傳邏輯對
  caption-only 沒有意義。
- `transcriptView`（面板內的逐字稿捲動區）維持現況顯示，不用特別隱藏：反正
  `m.transcripts` 只是本地記憶體陣列，離開面板/停止就消失，跟「不留紀錄」的精神
  不衝突。

## 測試

TDD，Python 側優先：
- `tests/test_live_session.py`：`make_caption_only_emit` 的合併邏輯（同說話者短
  暫停合併、超過 gap/長度上限另起一行、`push` 收到正確的 merged text）。
- `tests/test_app.py`：`mode=caption` 的 `/ws/native-capture` 連線不會呼叫
  `store.create_meeting`（用 monkeypatch 斷言未呼叫）、`/live/state` 對 pseudo-mid
  回傳可用的 title/started_at 不炸掉。

Swift 側：repo 目前沒有任何 XCTest 基礎設施（`Package.swift` 只有
`executableTarget`），這次不新增測試框架，維持手動驗證（build + 真機測試新開關）。
