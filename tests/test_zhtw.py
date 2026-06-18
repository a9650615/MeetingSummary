import zhtw


def test_converts_simplified_to_traditional_tw():
    zhtw.configure(True)
    assert zhtw.to_tw("会议决定") == "會議決定"


def test_disabled_is_identity():
    zhtw.configure(False)
    assert zhtw.to_tw("会议决定") == "会议决定"
    zhtw.configure(True)  # restore for other tests


def test_handles_empty():
    assert zhtw.to_tw("") == ""
