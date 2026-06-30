---
name: release
description: >-
  Record shipped work and batch-publish versions of MeetingSummary. Two modes:
  (A) after EVERY change, commit + add one terse user-facing line under
  "## 未發佈" in CHANGELOG (no version, no tag); (B) cut a release — promote
  未發佈 to vX.Y.Z, bump VERSION, run pytest, tag (triggers macOS CD) — which
  happens AUTOMATICALLY once enough has accumulated, on a critical fix, or when
  the user says 發佈/出版/ship. Invoke proactively; never hand-run git/tag steps.
---

# Release MeetingSummary

Pre-1.0 semver (`0.x.x`). A tag push `v*` triggers `.github/workflows/release.yml`
(macOS): checks `VERSION == tag`, builds the `.app` + prebuilt helpers, attaches
them to the GitHub Release. **Releasing = tagging.** Plain commits to `main` do
NOT release — they accumulate.

## Don't release per change. Batch. (Don't make the user ask.)

The old habit (bump+tag every fix) spammed versions. Instead:

### Mode A — record a change (EVERY change, automatically)
1. Commit + push `main`. Commit message MAY carry technical detail (it's for the
   repo). Author stays `a9650615` — never override.
2. Prepend ONE terse, **user-facing** line under a `## 未發佈` heading at the top of
   `CHANGELOG.md` (create the heading if absent). No VERSION change, no tag.

### Mode B — cut a release (decide this YOURSELF; don't wait to be reminded)
Release when ANY holds:
- `## 未發佈` has **~5+ lines**, or
- a **notable feature** just landed (worth its own version), or
- a **critical / blocking fix** users need now → release immediately, don't batch, or
- the user says 發佈 / 出版 / 上版 / ship.

Then run the gated flow.

## Release flow (Mode B — in order; stop on any failure)
1. **Version.** Read `VERSION`. Default = **patch**; **minor** for a notable feature.
   Never reuse/skip a number.
2. **Promote 未發佈** → rename the heading to `## X.Y.Z`; tidy bullets (merge dups,
   keep terse — see voice).
3. **Write `VERSION`** = `X.Y.Z` (one line + trailing newline).
4. **Gate: tests pass** — `.venv/bin/python -m pytest -q`. Red → STOP + fix. Never
   tag a red build.
5. **Commit** (`release: vX.Y.Z`), **push main**, then `git tag vX.Y.Z && git push
   origin vX.Y.Z`. `VERSION` must equal the tag (sans `v`) or CD fails its check.
6. **Verify CD** (~10 min; chatllm build is slow — poll, don't assume failure):
   `gh run list … release.yml` → `gh run watch <id> --exit-status`.
7. **Confirm assets**: `gh release view vX.Y.Z … --json assets` — expect
   `MeetingSummary-vX.Y.Z.zip` (+ usually the prebuilt helper tarballs, best-effort).

## CHANGELOG voice — for USERS, not engineers
Shown in-app (設定 → 更新紀錄). Keep it short — it's not a commit log.
- **One short line per user-visible change.** Lead with what they'll notice. zh-TW.
- **NO engineering detail**: no file/function/API names (ScreenCaptureKit, asyncio,
  setsid, Info.plist…), line counts, internal mechanism, or "(根因: …)". That goes
  in the commit, not here.
- `### 新增 / ### 改善 / ### 修正` headings only when a version has several items.
- Internal-only work (refactors with no user effect): omit, or one line —
  「內部結構整理，提升穩定性」.
- Good: 「修正點 App 後偶爾打不開」. Bad: 「bootstrap setsid() 脫離 .app process
  group，避免 launcher 結束時被 reap」.

## Notes
- End-user update is manual (設定 → 檢查更新 → 更新並重啟); tagging only publishes.
- One tag per version; if a tag exists, take the next patch.
