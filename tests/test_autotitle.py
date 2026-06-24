from app import _auto_title, _is_default_title


def test_is_default_title():
    assert _is_default_title("")
    assert _is_default_title(None)
    assert _is_default_title("未命名")
    assert _is_default_title("實測")
    assert _is_default_title("錄音 2026-06-24 14:30")
    assert not _is_default_title("Q3 產品規劃會議")


def test_auto_title_strips_and_caps():
    # backend echoes a noisy title -> cleaned to bare text
    out = _auto_title("摘要內容…", backend=lambda p: '「產品藍圖討論」。\n其他多餘行')
    assert out == "產品藍圖討論"


def test_auto_title_empty_summary_or_failure():
    assert _auto_title("", backend=lambda p: "x") is None
    assert _auto_title("有內容", backend=lambda p: (_ for _ in ()).throw(RuntimeError())) is None
