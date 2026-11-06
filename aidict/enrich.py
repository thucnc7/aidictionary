"""Dịch + giải thích thuật ngữ AI sang tiếng Việt bằng LLM (API tương thích OpenAI). Kết quả cache trong data/enriched.jsonl."""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = """Bạn là biên tập viên từ điển thuật ngữ AI Anh–Việt, phục vụ người Việt đọc sách và bài báo kỹ thuật tiếng Anh trên Kindle.
Đầu vào là một mảng JSON các thuật ngữ, mỗi thuật ngữ có định nghĩa tiếng Anh từ nguồn (có thể rỗng) và đôi khi có câu dịch tham khảo (references_vi).
Trả về DUY NHẤT một mảng JSON, mỗi thuật ngữ một phần tử, mỗi phần tử có đúng các khóa:
term, pos, vi, keep_en, short_vi, explain_vi, example_en, example_vi, related, expansion, aliases.
- term: chép nguyên chuỗi đầu vào.
- pos: một trong n, v, adj, adv, phrase, abbr.
- vi: 1–3 cách gọi tiếng Việt phổ biến nhất trong cộng đồng AI Việt Nam, ưu tiên cách dịch trong references_vi nếu có.
- keep_en: true nếu người Việt thường giữ nguyên thuật ngữ tiếng Anh khi nói và viết (ví dụ transformer, token, prompt).
- short_vi: một câu tối đa 150 ký tự, định nghĩa ngắn gọn bằng tiếng Việt. Câu này hiện trong popup Kindle nên phải tự đủ nghĩa.
- explain_vi: 2–5 câu giải thích khái niệm cho người mới: nó là gì, dùng để làm gì, khác gì với khái niệm dễ nhầm.
- example_en: một câu tiếng Anh tự nhiên, giống câu gặp trong sách hay paper AI. example_vi: bản dịch tiếng Việt của câu đó.
- related: 0–5 thuật ngữ tiếng Anh liên quan.
- expansion: dạng viết đầy đủ nếu term là từ viết tắt, ngược lại "".
- aliases: các cách viết khác hoặc từ viết tắt tiếng Anh của term (ví dụ "RAG" cho "retrieval-augmented generation"); có thể rỗng.
Bám sát định nghĩa nguồn. Không bịa số liệu, tên người hay mốc thời gian. Viết tiếng Việt có dấu, văn phong từ điển, không dùng markdown."""

FIELDS = {"term": str, "pos": str, "vi": list, "keep_en": bool, "short_vi": str, "explain_vi": str,
          "example_en": str, "example_vi": str, "related": list, "expansion": str, "aliases": list}


def build_messages(batch: list[dict], pairs: list[tuple[str, str]]) -> list[dict]:
    payload = []
    for t in batch:
        rx = re.compile(r"\b" + re.escape(t["term"]) + r"\b", re.I)
        refs = [f"{en[:200]} ⟶ {vi[:200]}" for en, vi in pairs if rx.search(en)][:3]
        payload.append({"term": t["term"], "definition_en": " ".join(t["definitions_en"])[:800], "references_vi": refs})
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _valid(d) -> bool:
    if not isinstance(d, dict) or any(not isinstance(d.get(k), typ) for k, typ in FIELDS.items()):
        return False
    return all(isinstance(x, str) for k in ("vi", "related", "aliases") for x in d[k]) and bool(d["short_vi"].strip())


def parse_response(text: str, expected: list[str]) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    want, out = set(expected), []
    for d in data:
        if isinstance(d, dict):
            for k in ("vi", "related", "aliases"):  # model đôi khi trả chuỗi thay vì mảng
                if isinstance(d.get(k), str):
                    d[k] = [d[k]] if d[k].strip() else []
        if _valid(d) and d["term"] in want:
            want.discard(d["term"])  # LLM trả trùng một term thì giữ bản đầu
            out.append({k: d[k] for k in FIELDS})
    return out


def content_of(body: str) -> str:
    if body.lstrip().startswith("data:"):
        parts = []
        for line in body.splitlines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            choices = json.loads(line[5:]).get("choices") or [{}]
            parts.append(choices[0].get("delta", {}).get("content") or "")
        return "".join(parts)
    return json.loads(body)["choices"][0]["message"]["content"]


def call_llm(messages: list[dict]) -> str:
    missing = [k for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Thiếu biến môi trường {', '.join(missing)} (xem mục Tự build trong README).")
    body = json.dumps({"model": os.environ["LLM_MODEL"], "messages": messages,
                       "temperature": 0.2, "stream": False}).encode()
    req = urllib.request.Request(f"{os.environ['LLM_BASE_URL'].rstrip('/')}/chat/completions", data=body, headers={
        "Authorization": f"Bearer {os.environ['LLM_API_KEY']}", "Content-Type": "application/json"})
    for wait in (5, 15, 45, None):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return content_of(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code < 500 or wait is None:
                raise
        except (urllib.error.URLError, TimeoutError):
            if wait is None:
                raise
        time.sleep(wait)


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(terms_path: str, out_path: str, pairs_path: str, batch_size: int = 8, limit: int | None = None,
        call=call_llm, workers: int = 1) -> int:
    terms = _read_jsonl(terms_path)
    pairs = [tuple(p) for p in _read_jsonl(pairs_path)]
    done = {e["term"] for e in _read_jsonl(out_path)}
    todo = [t for t in terms if t["term"] not in done][:limit]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    def work(batch):
        names = [t["term"] for t in batch]
        try:
            items = parse_response(call(build_messages(batch, pairs)), names)
        except Exception as e:  # term thiếu sẽ được làm lại ở lần chạy sau
            print(f"cảnh báo: batch {names[0]}… lỗi: {e}", flush=True)
            return []
        if len(items) < len(names):
            print(f"cảnh báo: batch {names[0]}… thiếu {len(names) - len(items)}/{len(names)}", flush=True)
        return items

    written = 0
    ex = ThreadPoolExecutor(workers)
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for fut in as_completed([ex.submit(work, b) for b in batches]):
                items = fut.result()
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in items)
                f.flush()
                written += len(items)
                done |= {e["term"] for e in items}
                print(f"{len(done)}/{len(terms)}", flush=True)
    finally:  # Ctrl+C: huỷ các batch chưa chạy thay vì gọi LLM rồi bỏ kết quả
        ex.shutdown(wait=False, cancel_futures=True)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    n = run("data/terms.jsonl", "data/enriched.jsonl", "data/cheatsheet_pairs.jsonl", a.batch_size, a.limit,
            workers=a.workers)
    print(f"mới: {n}")
