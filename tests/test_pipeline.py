from pipeline import run_pipeline
from store import Store


def test_pipeline_transcribes_then_summarizes(tmp_path):
    store = Store(tmp_path / "m.db")
    asr_backend = lambda p: [{"start": 0.0, "end": 1.0, "text": "討論預算"}]
    summary_backend = lambda prompt: "會議記錄"

    result = run_pipeline(
        "meeting.m4a", store=store, title="季度會議", lang="zh-TW",
        kind="minutes", asr_backend=asr_backend, summary_backend=summary_backend,
    )

    mid = result["meeting_id"]
    assert result["summary"] == "會議記錄"
    assert store.list_transcripts(mid)[0]["text"] == "討論預算"
    assert store.list_summaries(mid)[0]["text"] == "會議記錄"
    assert store.get_meeting(mid)["status"] == "finalized"  # batch run is done
