"""Summarizer: wraps a pluggable LLM backend (mlx-lm / Qwen2.5).
Long transcripts overflow the context window, so they go through map-reduce
(spec B3). Backend = callable(prompt) -> summary text.

ponytail: char count is a cheap proxy for token budget; swap for a real
tokenizer estimate if chunk sizing ever misbehaves on CJK."""

_INSTRUCTION = {
    "minutes": "請將以下會議逐字稿整理成會議記錄,輸出:會議重點、決議事項、"
               "待辦行動(含負責人與期限)。",
    "bullets": "請將以下會議逐字稿整理成條列式重點。",
    "actions": "請只從以下會議逐字稿擷取三類並分區塊輸出:\n"
               "【行動項目】每條格式「- [負責人,逐字稿沒明說是誰就寫 未指定] 事項 "
               "(期限,沒提到就寫 未定)」\n"
               "【決議】條列已拍板的決定\n"
               "【待解問題】條列尚未解決或待追蹤的問題\n"
               "某類若無內容寫「無」。除這三區塊外不要加其他敘述。",
}


# Small models invent plausible names/owners/deadlines (e.g. 小明/小紅/小米) that
# were never said. Hard rule up front + explicit fallbacks cut most of it; the
# deterministic _ground() pass catches the rest.
_GUARD = ("嚴格規則:只能根據下方逐字稿的內容,絕對不可杜撰任何人名、數字、日期、期限或"
          "未提及的事項。**逐字稿裡沒出現過的人名一律不准寫**(例如不可憑空寫出 小明/"
          "小米 這種沒講過的名字),不知道負責人就寫「未指定」,沒提到期限/時間就寫「未定」。"
          "忽略明顯與會議無關、亂碼、或像影片片頭/字幕台詞的內容(例如『優優獨播劇場』"
          "之類辨識雜訊),不要納入摘要。寧可少寫,也不要編造。\n")

_GROUND_FALLBACK = {"未指定", "未定", "待定", "無", "未提及", "tbd", "n/a", "-", "—"}


def _ground(out, transcript):
    """Deterministic anti-fabrication backstop. A value the summary assigns to a
    負責人 / 期限 / 時間 / 日期 field that appears NOWHERE in the transcript was
    invented by the model (the 小米 / 當天 case) -> replace with the safe fallback.
    A name/time actually said in the meeting is in the transcript, so it's kept."""
    import re  # noqa: PLC0415
    t = transcript or ""

    def made_up(val):
        v = val.strip(" 　()（）[]「」『』,，。、;；")
        return bool(v) and v.lower() not in _GROUND_FALLBACK and v not in t

    def label(fallback):
        def f(m):
            return m.group("pre") + fallback if made_up(m.group("v")) else m.group(0)
        return f

    out = re.sub(r"(?P<pre>負責人[:：]\s*)(?P<v>[^\s,，。;；]+)", label("未指定"), out)
    out = re.sub(r"(?P<pre>(?:期限|截止|時間|日期|舉行時間)[:：]\s*)(?P<v>[^\s,，。;；]+)",
                 label("未定"), out)
    # actions bracket owner: "- [小米] 事項"
    out = re.sub(r"(?P<pre>^\s*[-*]\s*\[)(?P<v>[^\]]+)\]",
                 lambda m: m.group(0) if not made_up(m.group("v"))
                 else m.group("pre") + "未指定]", out, flags=re.M)
    return out


def build_prompt(text, *, kind, lang, notes=""):
    ref = (f"\n\n使用者現場筆記（可信參考，優先採用其中的人名、日期、決議）:\n{notes.strip()}"
           if notes and notes.strip() else "")
    return f"{_GUARD}{_INSTRUCTION[kind]}\n輸出語言:{lang}{ref}\n\n逐字稿:\n{text}"


def _dedup_lines(out):
    """Collapse a runaway LLM loop — consecutive lines with identical content (a
    numbered/bulleted list that repeats the same item 18x, see report). Compare
    after stripping the leading list marker (the "29." / "30." / "-" differs but
    the content is the same). Universal guard regardless of model/penalty."""
    import re  # noqa: PLC0415
    out_lines, prev = [], None
    for ln in out.splitlines():
        key = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", ln).strip()
        if key and key == prev:
            continue
        out_lines.append(ln)
        prev = key
    return "\n".join(out_lines)


def _post(out, lang):
    """LLM often emits 簡體 even for a zh-TW meeting -> normalize to 繁體(台灣).
    Also collapse degenerate repeated lines (a loop the penalty didn't catch)."""
    out = _dedup_lines(out)
    if (lang or "").lower().startswith("zh"):
        import zhtw  # noqa: PLC0415
        return zhtw.to_tw(out)
    return out


def _chunk(text, max_chars):
    chunks, cur, size = [], [], 0
    for line in text.splitlines():
        if cur and size + len(line) > max_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def summarize(text, *, kind, lang, backend, max_chars=24000, notes=""):
    if not (text or "").strip():
        return "（無逐字稿，無法產生摘要）"
    # Notes are user-provided ground truth: pass to _ground as valid source so a
    # name/date the user typed isn't scrubbed as "fabricated".
    ground = text + ("\n" + notes if notes else "")
    if len(text) <= max_chars:
        out = _ground(backend(build_prompt(text, kind=kind, lang=lang, notes=notes)), ground)
        return _post(out, lang)
    # map: summarize each chunk; reduce: summarize the joined chunk summaries.
    partials = [
        _ground(backend(build_prompt(c, kind=kind, lang=lang, notes=notes)), c + "\n" + notes)
        for c in _chunk(text, max_chars)
    ]
    return summarize("\n".join(partials), kind=kind, lang=lang,
                     backend=backend, max_chars=max_chars, notes=notes)


def mlx_lm_backend(model="mlx-community/Qwen2.5-14B-Instruct-4bit", max_tokens=1024):
    """Real backend — Apple Silicon only, imported lazily. 7B is the default
    for very long single-pass inputs; pick by input length upstream (spec §6)."""
    from mlx_lm import generate, load  # noqa: PLC0415
    from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: PLC0415

    model_obj, tokenizer = load(model)
    # Greedy (temp 0) keeps the summary factual — sampling is what invents 小明/期限.
    # repetition_penalty stops the model looping the same line/phrase. The DEFAULT
    # repetition_context_size is only 20 tokens — shorter than one list item, so by
    # the time a line repeats the earlier identical tokens have rolled out of the
    # window and aren't penalized → it loops a whole line forever (see the 18x
    # "中午前將完成 Bug…" report). Widen the window to cover many lines + raise the
    # penalty; frequency_penalty additionally scales with how often a token recurs.
    sampler = make_sampler(temp=0.0)
    logits_processors = make_logits_processors(
        repetition_penalty=1.5, repetition_context_size=512,
        frequency_penalty=0.4, frequency_context_size=512)

    def _run(prompt):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return generate(model_obj, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, logits_processors=logits_processors)

    return _run
