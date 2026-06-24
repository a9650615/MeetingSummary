---
name: release
description: >-
  Cut and publish a new version of MeetingSummary. Use whenever the user wants to
  release / ship / publish a version, bump the version, tag a release, or "出新版 /
  發佈 / 發版 / 上版". Runs the full gated flow: bump VERSION, write the zh-TW
  CHANGELOG entry, run pytest, commit + push main, tag vX.Y.Z (triggers the
  macOS CD that builds the .app + prebuilt chatllm and attaches them to the
  GitHub Release), then verify the release assets. Proactively invoke this — do
  NOT hand-run the git/tag steps — whenever a chunk of work is done and the user
  signals it should go out.
---

# Release MeetingSummary

Pre-1.0 semver (`0.x.x`). Tag push to `v*` triggers `.github/workflows/release.yml`
on a macOS runner: it checks `VERSION == tag`, builds the `.app`, builds + packages
the prebuilt `chatllm-runtime-arm64.tar.gz`, and attaches both to the GitHub Release.

## Checklist (do in order; stop on any failure)

1. **Pick the version.** Read `VERSION`. Default bump = **patch** (`0.1.6 → 0.1.7`).
   Bump minor only if the user says so or the change is a notable feature set.
   Never reuse or skip a number. Call it `X.Y.Z` below.

2. **Draft the CHANGELOG entry.** `git log $(git describe --tags --abbrev=0)..HEAD
   --oneline` to see what's shipped since the last tag. Add a new section at the
   **top** of `CHANGELOG.md`, directly under the intro line:
   ```
   ## X.Y.Z
   - <zh-TW bullet, user-facing, what changed + why it matters>
   ```
   Match the existing voice: concise zh-TW, technical terms kept, one bullet per
   real change. Don't list internal churn.

3. **Write `VERSION`** = `X.Y.Z` (single line, trailing newline).

4. **Gate: tests must pass.** `.venv/bin/python -m pytest -q`. If anything fails,
   STOP and fix before releasing — never tag a red build.

5. **Commit.** Stage all, commit with a `type(scope): summary` subject + a short
   body of the real changes. The git author is already `a9650615`
   (`13378242+a9650615@users.noreply.github.com`) — **never override it**; the repo
   must show only that contributor.

6. **Push main, then tag.**
   ```
   git push origin main
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
   `VERSION` must equal the tag (without the `v`) or CD fails its own check.

7. **Verify CD.** The release job can take **~10 min** (the chatllm cmake build is
   the slow step). Watch it:
   ```
   rid=$(gh run list -R a9650615/MeetingSummary --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')
   gh run watch "$rid" -R a9650615/MeetingSummary --exit-status
   ```
   If `gh run watch` times out, poll `gh run view "$rid" --json status,conclusion`
   instead of assuming failure — the long step is normal.

8. **Confirm assets.** The release must carry the installer zip and (best-effort)
   the chatllm runtime:
   ```
   gh release view vX.Y.Z -R a9650615/MeetingSummary --json assets -q '.assets[]|"\(.name) \(.size)b"'
   ```
   Expect `MeetingSummary-vX.Y.Z.zip` and usually `chatllm-runtime-arm64.tar.gz`.
   The chatllm prebuild is best-effort in CI — if it's missing, the release still
   stands (users fall back to source build); note it to the user.

## Notes
- Updating is manual for end users (設定 → 檢查更新 → 更新並重啟). Releasing here only
  publishes; it does not push updates to anyone.
- Don't bump/tag for docs-only or experiment commits unless the user asks.
- One tag per version; if a tag already exists, pick the next patch.
