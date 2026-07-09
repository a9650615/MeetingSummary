# MeetingSummary · 本地會議錄音 / 逐字稿 / 摘要

Apple Silicon (M 系列) 上**完全本地、隱私優先**的會議助手：錄音 → 即時逐字稿 → 會後精校 → 說話者分群 → 摘要。語音不離開本機（服務只綁 `127.0.0.1`）。主語言 zh-TW，支援中英混講。

## 快速開始

### 原生 App（一般使用者）
1. 到 [Releases](https://github.com/a9650615/MeetingSummary/releases/latest) 下載 `MeetingSummary-vX.Y.Z.zip`。
2. 解壓，首次右鍵 **打開**（自簽章、非 Apple 公證，跳一次 Gatekeeper 警告即可，之後正常雙擊）。
3. 第一次啟動會自動下載已裝好相依（含 mlx 加速）的 Python 執行環境，免裝 Xcode/cmake；依需求下載模型。瀏覽器會開進度頁，裝好自動導向 App（或直接開原生錄音面板）。
4. 開始錄音時系統會跳「麥克風」「螢幕與系統音訊錄製」授權，兩個都要允許——沒有系統音訊錄製權限，抓對方聲音會是全靜音。

App 體積小（~70KB launcher）；相依與模型都在**首次啟動時才下載**（Apple Silicon 全套約 1–1.5GB，一次性）。

### 開發 / 原始碼啟動
```bash
./supervise.sh              # 起服務 + 開會偵測,綁 127.0.0.1:8765
# 或裸跑:
.venv/bin/python -m app     # MEETING_PORT 可改 (預設 8765, 8000 留給開發)
```
打開 http://127.0.0.1:8765 。

## 功能
- **Live 即時逐字稿**：瀏覽器麥克風/系統音 → 16kHz PCM over WebSocket → 本地 ASR。兩段式（whisper 即時稿 + 精準稿）。來源可選 我 / 對方 / 混合 / 分軌。
- **會後精校**：用更準的模型重跑整場逐字稿，視窗化進度。
- **說話者分群（聲紋）**：sherpa-onnx（pyannote-3-0 切割 + 3D-Speaker 聲紋），離線。模型**首次分群自動下載**到 `models/`，免手動設定。
- **摘要**：mlx-lm（Qwen2.5）。
- **回放**：雙軌統一播放、可點逐字稿跳播、播放高亮跟讀。
- **標題 / 說話者命名**：標題可編輯；點說話者名可整場改名。
- **開會偵測通知**：偵測麥克風被佔用即跳通知（Notion/Granola 風格）。
- **PWA**：可安裝。
- **線上更新**：比對 GitHub Releases；模型管理頁有「檢查更新 → 更新並重啟」。

## 模型 & 加速 runtime
模型管理頁（⚙️）可下載/刪除模型、檢視快取、釋放閒置記憶體、編譯加速 runtime。

| 用途 | 預設 | 備選 |
|------|------|------|
| Live ASR | whisper small-q4 (mlx) | whisper turbo/base/tiny、Qwen3-ASR |
| 精校 ASR | whisper turbo-q4 | Qwen3-ASR 0.6B/1.7B |
| 摘要 | Qwen2.5-3B-4bit (mlx) | — |
| 分群 | sherpa pyannote-3-0 + 3D-Speaker（自動下載） | `SHERPA_SEG_MODEL` / `SHERPA_EMB_MODEL` 覆寫 |

**加速 runtime（.cpp · Metal，選用）**
- `chatllm`（Qwen3-ASR 1.7B GGUF）：設定頁一鍵安裝會**先下載 CI 預編譯版**（可攜 arm64 bundle，免 cmake）；下載不到才從原始碼編譯（自動 `brew install cmake`）。
- `femelo`（Qwen3-ASR 0.6B GGUF）：需 **python≥3.11**（找不到會自動 `brew install python@3.13`），裝在獨立 venv。

設定頁一鍵安裝；缺的環境會自動以 brew 補齊，失敗則於該列顯示明確錯誤（⚠）。

## 隱私
- 服務只綁 `127.0.0.1`（loopback），不對外。
- 錄音、逐字稿、模型全在本機 `data/` 與 `models/`。
- 唯一外連：依需求下載模型 / 檢查 GitHub Releases 更新。

## 版本
版本見 [`VERSION`](VERSION)、變更見 [`CHANGELOG.md`](CHANGELOG.md)（App 內模型管理頁也看得到）。
