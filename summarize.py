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
               "【行動項目】每條格式「- [負責人] 事項 (期限,無則寫 待定)」\n"
               "【決議】條列已拍板的決定\n"
               "【待解問題】條列尚未解決或待追蹤的問題\n"
               "某類若無內容寫「無」。除這三區塊外不要加其他敘述。",
}


# Small models invent plausible names/owners/deadlines (e.g. 小明/小紅) that were
# never said. Hard rule up front + explicit fallbacks cut most of it.
_GUARD = ("嚴格規則:只能根據下方逐字稿的內容,絕對不可杜撰任何人名、數字、日期、期限或"
          "未提及的事項。逐字稿沒提到負責人就寫「未指定」,沒提到期限就寫「未定」。"
          "忽略明顯與會議無關、亂碼、或像影片片頭/字幕台詞的內容(例如『優優獨播劇場』"
          "之類辨識雜訊),不要納入摘要。寧可少寫,也不要編造。\n")


def build_prompt(text, *, kind, lang):
    return f"{_GUARD}{_INSTRUCTION[kind]}\n輸出語言:{lang}\n\n逐字稿:\n{text}"


def _post(out, lang):
    """LLM often emits 簡體 even for a zh-TW meeting -> normalize to 繁體(台灣)."""
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


def summarize(text, *, kind, lang, backend, max_chars=24000):
    if len(text) <= max_chars:
        return _post(backend(build_prompt(text, kind=kind, lang=lang)), lang)
    # map: summarize each chunk; reduce: summarize the joined chunk summaries.
    partials = [
        backend(build_prompt(c, kind=kind, lang=lang))
        for c in _chunk(text, max_chars)
    ]
    return summarize("\n".join(partials), kind=kind, lang=lang,
                     backend=backend, max_chars=max_chars)


def mlx_lm_backend(model="mlx-community/Qwen2.5-14B-Instruct-4bit", max_tokens=1024):
    """Real backend — Apple Silicon only, imported lazily. 7B is the default
    for very long single-pass inputs; pick by input length upstream (spec §6)."""
    from mlx_lm import generate, load  # noqa: PLC0415
    from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: PLC0415

    model_obj, tokenizer = load(model)
    # Greedy (temp 0) keeps the summary factual — sampling is what invents 小明/期限.
    # repetition_penalty stops the model from looping the same line/phrase.
    sampler = make_sampler(temp=0.0)
    logits_processors = make_logits_processors(repetition_penalty=1.3)

    def _run(prompt):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return generate(model_obj, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, logits_processors=logits_processors)

    return _run
