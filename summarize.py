"""Summarizer: wraps a pluggable LLM backend (mlx-lm / Qwen2.5).
Long transcripts overflow the context window, so they go through map-reduce
(spec B3). Backend = callable(prompt) -> summary text.

ponytail: char count is a cheap proxy for token budget; swap for a real
tokenizer estimate if chunk sizing ever misbehaves on CJK."""

# Kept deliberately terse. A 4-bit model's instruction-following degrades as the
# prompt grows, and the rules that matter (owner = the speaker who said it, no
# invented names, no placeholder deadline) compete with every extra clause for
# attention. Each line below earned its place by fixing an observed failure; the
# deterministic passes (_ground / _drop_empty_deadlines / _drop_meta) are what
# make the short version safe.
_INSTRUCTION = {
    # 待辦行動 is NOT asked for here — it's assembled per speaker by
    # _actions_by_speaker(), which is the only way it comes out complete.
    "minutes": "請將以下會議逐字稿整理成會議記錄,輸出兩個區塊:\n"
               "【會議重點】條列本次談到的主題,每條開頭用「- 」;不要用人名開頭,"
               "也不要寫「某某提到/表示/回應」這種句子。\n"
               "【決議事項】只列已經拍板的決定,沒有就寫「無」。\n"
               "不要輸出其他區塊。用逐字稿的原話,別換講法。",
    "bullets": "請將以下會議逐字稿整理成條列式重點。",
    "actions": "只擷取三段:\n"
               "【行動項目】一行一件「- [名字] 事情」,沒說是誰寫 [未指定];"
               "有明講期限才在後面加「(期限)」\n"
               "【決議】已拍板的決定\n"
               "【待解問題】未解決或待追蹤的\n"
               "某段無內容寫「無」。不要其他敘述。",
}


# Small models invent plausible names/owners/deadlines (小明/小紅/小米, and on
# meeting 187 a "Qwen (負責)" — the model's own name) and credit one speaker's
# work to another. These rules plus the enumerated speaker roster cut most of it.
_GUARD = ("嚴格規則:只能根據下方逐字稿的內容,絕對不可杜撰任何人名、數字、日期、期限或"
          "未提及的事項。**逐字稿裡沒出現過的人名一律不准寫**,不知道負責人就寫「未指定」。"
          "**負責人只能是說出那句話的說話者本人**:逐字稿每行開頭「說話者:」就是講者,"
          "誰說「我會做X」,X 的負責人就是誰,不可以安到別人頭上。"
          "**期限只有逐字稿明講日期或時間才寫**,沒講到就完全不要寫期限這個欄位"
          "(不要寫「期限:未定」之類的佔位字)。"
          "忽略明顯與會議無關、亂碼、或像影片片頭/字幕台詞的內容(例如『優優獨播劇場』"
          "之類辨識雜訊),不要納入摘要。寧可少寫,也不要編造。"
          "結尾不要加任何自我說明或聲明。\n")

_GROUND_FALLBACK = {"未指定", "未定", "待定", "無", "未提及", "tbd", "n/a", "-", "—"}

_EMPTY_DEADLINE = "未定|待定|未提及|未指定|不明|無|TBD|tbd|N/A|n/a|-|—|\\?"


def _speakers(transcript):
    """Speaker labels from the "說話者: 內容" transcript fed to the summarizer.
    This is the closed set of people who can own an action item — anything else
    the model writes as an owner (a model name, a topic, an invented person) is
    fabrication, regardless of whether the string happens to appear elsewhere in
    the transcript."""
    import re  # noqa: PLC0415
    return {m.group(1).strip()
            for m in re.finditer(r"^[ \t]*([^\s:：][^:：\n]{0,15}?)[:：][ \t]",
                                 transcript or "", re.M)}


def _ground(out, transcript):
    """Deterministic anti-fabrication backstop. A value the summary assigns to a
    負責人 / 期限 / 時間 / 日期 field that appears NOWHERE in the transcript was
    invented by the model (the 小米 / 當天 case) -> replace with the safe fallback.
    A name/time actually said in the meeting is in the transcript, so it's kept —
    an owner who was named but never spoke (「這個給 Michael 處理」) stays."""
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
    # owner form the two patterns above miss — "- Qwen (負責)" in the 決議事項 list,
    # where 「Qwen」 is the summarizer model naming itself as a participant.
    out = re.sub(r"(?P<v>[^\s,，。;；:：、(（]+)(?P<post>\s*[(（]\s*負責[^)）]*[)）])",
                 lambda m: m.group(0) if not made_up(m.group("v"))
                 else "未指定" + m.group("post"), out)
    return out


def _drop_empty_deadlines(out):
    """A deadline nobody stated is noise, not information — drop the whole field
    instead of printing 「期限:未定」 on every line (user report). Runs after
    _ground(), so a fabricated date has already been demoted to 未定 and gets
    dropped here too."""
    import re  # noqa: PLC0415
    # Whole-line first — otherwise stripping the field out of "- 期限：未定" leaves
    # an orphan "-" bullet that no longer matches the line pattern.
    out = "\n".join(ln for ln in out.splitlines()
                    if not re.fullmatch(r"\s*[-*\d.)]*\s*(?:期限|截止|時間)[:：]?\s*(?:"
                                        + _EMPTY_DEADLINE + r")\s*", ln))
    # "…批準 (期限: 未定)" then "…批準；期限：未定"
    out = re.sub(r"[ \t]*[（(]\s*(?:期限|截止|時間|日期)[:：]?\s*(?:" + _EMPTY_DEADLINE
                 + r")\s*[)）]", "", out)
    return re.sub(r"[ \t]*[;；,，、|·]?[ \t]*(?:期限|截止日?|完成時間)[:：]\s*(?:"
                  + _EMPTY_DEADLINE + r")\s*(?=$|[\n|])", "", out, flags=re.M)


# A first-person commitment cue. Only a gate for "is this person an actor at all"
# — once a speaker qualifies, the extractor sees ALL of their lines, so an item
# phrased without a cue (「那早上會追蹤一下這邊的進度」) is still picked up.
_COMMIT_CUE = r"我(?:們)?(?:今天|等下|接下來|之後|明天|下午|早上)?(?:也|再|先|還)*(?:會|要|來|去)|" \
              r"(?:今天|明天|下午|早上|等下|接下來)(?:也|再|先|還)*會"

_ACTIONS_ONE = ("這是 {name} 在會議中說的話。列出 {name} 說自己要做的事,一行一件,"
                "格式「- 事情」,用原話,不要寫名字,不要編造。"
                "只要他講了「我會/我要/今天會」就算一件,照原話寫,"
                "就算內容聽起來不具體也要寫出來。\n"
                "輸出語言:{lang}\n\n{lines}")


def _commit_speakers(transcript):
    """{speaker: [their lines]} for speakers who committed to doing something.

    Asking one call for the whole 待辦行動 list drops items nondeterministically —
    over six runs on the same transcript the 7B-4bit model lost a different
    speaker almost every time (Pei's second item, then Chester entirely, then
    Pei entirely). Iterating speakers in code makes completeness structural, and
    feeding each extractor only that speaker's own words makes mis-attribution
    impossible rather than merely discouraged."""
    import re  # noqa: PLC0415
    said, actors = {}, set()
    for m in re.finditer(r"^[ \t]*([^\s:：][^:：\n]{0,15}?)[:：][ \t]*(.*)$",
                         transcript or "", re.M):
        who, line = m.group(1).strip(), m.group(2).strip()
        if not line:
            continue
        said.setdefault(who, []).append(line)
        if re.search(_COMMIT_CUE, line):
            actors.add(who)
    return {w: said[w] for w in said if w in actors}


_FIRST_PERSON = (r"^(?:好[,，]?\s*)?我(?:們)?(?:今天|等下|接下來|之後|明天|下午|早上)?"
                 r"(?:也|再|先|還)*(?:會|要)[,，]?\s*")


def _third_person(item):
    """"我今天也會根據…" reads wrong in a third-person action list. Applies to the
    extractor's output too, not just the verbatim fallback — the extractor is told
    to keep the speaker's own words, so it echoes the 我 as well."""
    import re  # noqa: PLC0415
    return re.sub(_FIRST_PERSON, "", item).strip()


def _cue_lines(lines):
    """The speaker's own commitment sentences, verbatim — the fallback when the
    extractor returns 「無」 for someone the cue gate already proved committed to
    something. Chester's 「我今天也會根據昨天討論還有測試結果,繼續訓練那個量」 is
    vague ASR, so the model judged it not concrete and dropped him entirely.
    Quoting him is worse prose than the model's phrasing but infinitely better
    than silently losing an attendee's work."""
    import re  # noqa: PLC0415
    out = []
    for ln in lines:
        for s in re.split(r"(?<=[。;；!?])", ln):
            s = s.strip().rstrip("。")
            if s and re.search(_COMMIT_CUE, s) and not re.fullmatch(r"謝謝|以上|好", s):
                out.append(s)
    return out


def _actions_by_speaker(transcript, *, lang, backend, max_chars=24000):
    """The 【待辦行動】 block, one focused extraction per committing speaker."""
    out = []
    for name, lines in _commit_speakers(transcript).items():
        prompt = _ACTIONS_ONE.format(name=name, lang=lang,
                                     lines="\n".join(lines)[:max_chars])
        try:
            got = _post(backend(prompt), lang)
        except Exception:
            got = ""  # best-effort: one speaker's failure can't sink the summary
        items = []
        for ln in got.splitlines():
            item = ln.strip().lstrip("-*•").strip()
            if item and item.lower() not in _GROUND_FALLBACK:
                items.append(item)
        out += [f"- {name}: {_third_person(i).rstrip('。')}"
                for i in (items or _cue_lines(lines))]
    return out


def _drop_meta(out):
    """Strip the model talking about itself instead of the meeting: the compliance
    note it tacks on the end ("以上內容均根據提供的逐字稿整理,不涉及杜撰…") and
    the empty roll-call bullet for someone who never said anything
    ("- Hank: 未指定（無實質發言）")."""
    import re  # noqa: PLC0415
    nothing = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*(?:\*\*)?[^:：\n]{0,20}(?:\*\*)?[:：]?\s*"
                         r"[（(]?(?:未指定|無|沒有|未)[^\n]{0,12}"
                         r"(?:無實質發言|沒有(?:明確)?(?:提到|發言|說明)|未發言|無內容)"
                         r"[^\n]{0,4}$")
    keep = [ln for ln in out.splitlines()
            if not (re.search(r"逐字稿|以上(?:內容|資訊)", ln)
                    and re.search(r"杜撰|捏造|虛構|編造|未(?:加以)?添加", ln))
            and not nothing.match(ln)]
    return "\n".join(keep).rstrip()


def build_prompt(text, *, kind, lang, notes=""):
    ref = (f"\n\n使用者現場筆記（可信參考，優先採用其中的人名、日期、決議）:\n{notes.strip()}"
           if notes and notes.strip() else "")
    who = sorted(_speakers(text))
    # The owner candidates, stated up front — the model attributes far better when
    # they're enumerated than when it has to infer them from the line prefixes.
    roster = f"\n說話者:{'、'.join(who)}" if who else ""
    return f"{_GUARD}{_INSTRUCTION[kind]}\n輸出語言:{lang}{roster}{ref}\n\n逐字稿:\n{text}"


_CORRECT = (
    "校正這份語音辨識逐字稿的同音字、錯別字、斷詞:\n"
    "- 人名若與名單某人同音或近音,改成名單的寫法\n"
    "- 不改數字、日期、金額、時間,和任何你不確定的字\n"
    "- 不增不刪內容\n"
    "- 輸入 {n} 行就輸出 {n} 行,保持「說話者: 內容」一行對一行。"
    "就算前後兩行同一人、句子像被切斷,也不可合併或拆行\n"
    "- 只輸出逐字稿,不要說明或程式碼框\n")


def build_correction_prompt(text, *, roster, lang):
    names = "、".join(roster) if roster else "(無)"
    # State the expected line count — the model merged two consecutive same-speaker
    # lines on meeting 187, which cost the summarizer one action item.
    head = _CORRECT.replace("{n}", str(len(_line_labels(text)) or len(text.splitlines())))
    return f"{head}\n已知與會者名單:{names}\n輸出語言:{lang}\n\n逐字稿:\n{text}"


def _line_labels(text):
    """The leading "說話者:" label of every labelled line, in order."""
    import re  # noqa: PLC0415
    return [m.group(1).strip() for m in
            re.finditer(r"^[ \t]*([^\s:：][^:：\n]{0,15}?)[:：][ \t]", text or "", re.M)]


def _lines_aligned(src, out):
    """Correction must stay line-for-line. On meeting 187 the model concatenated
    two of Pei's consecutive lines into one — no words lost, but the line boundary
    was, and the summarizer only picked up the head of the merged line, silently
    dropping his second action item. Same speaker sequence in, same sequence out,
    or the correction is rejected and the raw chunk is used."""
    return _line_labels(src) == _line_labels(out)


def correct_transcript(text, *, roster, lang, backend, max_chars=24000):
    """Conservative ASR-error correction of the text FED TO the summarizer:
    homophones/typos + align mentioned person names to the voiceprint roster.
    Never invents; numbers/dates/amounts left alone. Best-effort — any backend
    error returns that piece unchanged so the summary still runs. Non-destructive:
    the stored transcript is untouched, only the summary's input is cleaned."""
    if not (text or "").strip():
        return text

    def _one(t):
        if not t.strip():
            return t
        try:
            out = backend(build_correction_prompt(t, roster=roster, lang=lang)).strip()
        except Exception:
            return t
        # dedup=False: this is per-line transcript correction, not summary output.
        # Two speakers genuinely repeating a short line ("我: 好" / "對方: 好") are
        # real content — collapsing them would violate the prompt's no-delete rule.
        out = _post(out, lang, dedup=False)
        return out if out and _lines_aligned(t, out) else t

    if len(text) <= max_chars:
        return _one(text)
    return "\n".join(_one(c) for c in _chunk(text, max_chars))


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


def _post(out, lang, dedup=True):
    """LLM often emits 簡體 even for a zh-TW meeting -> normalize to 繁體(台灣).
    dedup (summary output only) collapses degenerate repeated lines (a loop the
    penalty didn't catch); the transcript-correction path passes dedup=False so
    real repeated utterances aren't deleted."""
    if dedup:
        out = _dedup_lines(out)
        out = _drop_empty_deadlines(_drop_meta(out))
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


def _summary_body(text, *, kind, lang, backend, max_chars, notes):
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
    return _summary_body("\n".join(partials), kind=kind, lang=lang, backend=backend,
                         max_chars=max_chars, notes=notes)


def summarize(text, *, kind, lang, backend, max_chars=24000, notes=""):
    if not (text or "").strip():
        return "（無逐字稿，無法產生摘要）"
    out = _summary_body(text, kind=kind, lang=lang, backend=backend,
                        max_chars=max_chars, notes=notes)
    if kind != "minutes":
        return out
    # 待辦行動 is built per speaker from the ORIGINAL transcript — never from the
    # map-reduce partials, which have already lost per-line attribution.
    items = _actions_by_speaker(text, lang=lang, backend=backend, max_chars=max_chars)
    return out + "\n\n【待辦行動】\n" + ("\n".join(items) if items else "無")


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
    # A MILD penalty only. 1.5 was catastrophic: over a long zh summary the real
    # names/terms recur constantly (王小強…王小強), so a strong penalty over a
    # 512-token window forbade repeating them and forced the model to substitute
    # invented alternatives — the "瞎掰" report — plus garbled late-generation
    # nonsense (「一侓」). 1.15 keeps output grounded and structured; _dedup_lines()
    # is the real backstop for literal repeated-line loops, so the penalty doesn't
    # have to be.
    logits_processors = make_logits_processors(
        repetition_penalty=1.15, repetition_context_size=256,
        frequency_penalty=0.1, frequency_context_size=256)

    def _run(prompt):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return generate(model_obj, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, logits_processors=logits_processors)

    return _run
