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


def build_prompt(text, *, kind, lang):
    return f"{_INSTRUCTION[kind]}\n輸出語言:{lang}\n\n逐字稿:\n{text}"


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
        return backend(build_prompt(text, kind=kind, lang=lang))
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

    model_obj, tokenizer = load(model)

    def _run(prompt):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return generate(model_obj, tokenizer, prompt=text, max_tokens=max_tokens)

    return _run
