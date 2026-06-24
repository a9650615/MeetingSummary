# 變更紀錄

依語意化版本（0.x.x 為 1.0 前的快速迭代）。

## 0.1.5
- 「模型管理」重新設計為「設定」頁：分三區（系統·更新 / 模型 / 加速 runtime）+ 區段導覽 + 版本顯示，更新/釋放記憶體整併到系統區。

## 0.1.4
- 加速 runtime 加上「清除」：移除該 runtime 的編譯目錄（femelo→`.venv-qwen314`、chatllm→`chatllm.cpp`），狀態回到未安裝；重新編譯可復原，模型權重保留。

## 0.1.3
- 統一按鈕大小：移除首頁 Live 按鈕的特例尺寸，表單列改 `align-items:end`，輸入框與按鈕底邊對齊，不再有大有小。
- 更新改為**手動決定**：啟動不再自動套用更新，只在模型管理頁「檢查更新 → 更新並重啟」由使用者決定。

## 0.1.2
- 修正說話者分群：sherpa 模型（pyannote-3-0 切割 + 3D-Speaker 聲紋）首次分群**自動下載**到 `models/`，乾淨機器免手動設定。
- 移除 community-1 onnx 選項（與 sherpa 不相容，缺 `sample_rate` metadata）。
- 模型下載 / runtime 編譯失敗時，於模型管理頁顯示明確錯誤（不再只進伺服器 log）。
- 新增 README 與 App 內變更紀錄。

## 0.1.1
- CD：tag `v*` 於 macOS runner 編譯 `.app` 並附到 Release。
- 手動「檢查更新」：模型管理頁比對 GitHub Releases → 更新並重啟。
- 共用 `updater.py`（bootstrap 與 App 共用）。

## 0.1.0
- 首個版本：Live 即時逐字稿（兩段式）、會後精校、說話者分群、摘要、雙軌回放。
- 原生 `.app`（首次啟動安裝相依 + 依需求下載模型）、PWA、UI 內關閉。
- 開會偵測通知；標題 / 說話者命名；辨識語言指定。
- 加速 runtime：femelo（0.6B）、chatllm（1.7B），Metal。
- 服務綁 `127.0.0.1:8765`（隱私優先）。
