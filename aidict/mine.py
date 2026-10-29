"""Quét thuật ngữ AI mới từ abstract arXiv, bảng thuật ngữ D2L và gợi ý của LLM -> data/mined_terms.txt."""
import json
import os
import re
import time
import urllib.error
import xml.etree.ElementTree as ET
from collections import Counter

from aidict.base import load_base
from aidict.enrich import call_llm
from aidict.sources import D2L_URL, HF_URL, fetch, norm_key, parse_d2l_glossary, parse_hf_glossary

ARXIV = ("https://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending"
         "&start={start}&max_results={n}")
CATS = ["cs.CL", "cs.LG", "cs.AI", "cs.CV"]
STOP = set("""a an the of and or to in on for with by from at as is are was were be been being this that these those
we our us it its they their there here which who whom whose what when where how than then also such both either via
into onto over under between across within without about against among per can could may might must should would will
do does did not no nor only more most less least very much many few other another each every all any some same new
using use used based propose proposed show shows shown present presents paper work study approach method methods
result results however while thus therefore furthermore moreover often first second third one two three up""".split())
CATEGORIES = [
    "LLM pretraining, fine-tuning and alignment", "inference efficiency, quantization and model serving",
    "reasoning models and test-time compute", "AI agents, tool use and planning", "retrieval-augmented generation and search",
    "multimodal and vision-language models", "diffusion, flow matching and generative media",
    "reinforcement learning", "evaluation, benchmarks and metrics", "AI safety, alignment and interpretability",
    "training data, data curation and synthetic data", "model architectures, attention variants and state space models",
    "distributed training, GPUs and AI hardware", "classic machine learning and statistics",
    "natural language processing fundamentals", "computer vision fundamentals",
]
FILTER_PROMPT = """You are building an English–Vietnamese dictionary of AI/ML terminology for people reading AI papers and books.
Below is a JSON array of candidate phrases mined from recent arXiv abstracts and AI textbooks.
Return ONLY a JSON array of strings: the candidates that are established technical terms a reader would look up
(concepts, methods, architectures, training or inference techniques, metrics, well-known benchmarks and models).
Write each kept term in canonical dictionary form: singular, lowercase unless it is an acronym or proper name (LLM, LoRA, BERT).
For a candidate written as "long form (ACRONYM)", keep that exact shape.
Drop generic academic phrases (experimental results, proposed framework, state-of-the-art performance), fragments,
paper-specific names that are not widely known, and plain English words with no special AI meaning."""
BRAINSTORM_PROMPT = """List 40 English technical terms about {cat} that often appear in AI papers and books from 2023 to 2026,
favouring newer terms a reader might not find in an older dictionary. Canonical dictionary form: singular, lowercase unless
an acronym or proper name. Return ONLY a JSON array of strings."""


def _initials(words: list[str], skip_stop: bool) -> str:
    return "".join(p[0] for w in words for p in w.split("-") if p and not (skip_stop and p.lower() in STOP)).lower()


def acronym_pairs(texts: list[str]) -> dict[str, str]:
    """Cặp "dạng đầy đủ (VIẾT TẮT)" trong văn bản; khớp chữ cái đầu (kiểu Schwartz–Hearst, đơn giản hoá)."""
    seen: dict[str, Counter] = {}
    for text in texts:
        for m in re.finditer(r"\(([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*?)(s?)\)", text):
            acr, plural = m.group(1), m.group(2)
            letters = re.sub(r"[^a-z]", "", acr.lower())
            words = [w.strip(",'\"") for w in re.split(r"[.;:!?()\[\]]", text[:m.start()])[-1].split()]
            for k in range(1, min(len(words), 10) + 1):
                suffix = words[-k:]
                if letters in (_initials(suffix, False), _initials(suffix, True)):
                    long = " ".join(suffix).lower()
                    if plural and long.endswith("s"):
                        long = long[:-1]
                    seen.setdefault(acr, Counter())[long] += 1
                    break
    return {acr: c.most_common(1)[0][0] for acr, c in seen.items()}


def candidate_ngrams(texts: list[str], known: set[str], base_words: set[str], min_df: int = 4) -> list[tuple[str, int]]:
    df = Counter()
    for text in texts:
        toks = re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", text.lower())
        grams = set()
        for n in range(1, 5):
            for i in range(len(toks) - n + 1):
                g = toks[i:i + n]
                if g[0] in STOP or g[-1] in STOP:
                    continue
                if n == 1 and (len(g[0]) < 4 or g[0] in base_words):
                    continue
                grams.add(" ".join(g))
        df.update(grams - known)
    return sorted(((g, c) for g, c in df.items() if c >= min_df), key=lambda x: -x[1])


def parse_term_list(text: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in data if isinstance(s, str) and s.strip()] if isinstance(data, list) else []


def fetch_arxiv(cat: str, total: int, page: int = 200, get=fetch, sleep=time.sleep) -> list[str]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for start in range(0, total, page):
        entries = []
        for attempt in range(6):
            try:
                entries = ET.fromstring(get(ARXIV.format(cat=cat, start=start, n=page))).findall("a:entry", ns)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503):
                    raise
                sleep(60 * 2 ** attempt)  # bị chặn tạm vì gọi dày: chờ 1, 2, 4… phút
                continue
            if entries:
                break
            sleep(10)  # arXiv đôi khi trả trang rỗng khi quá tải
        out += [" ".join((e.findtext(f"a:{k}", "", ns) or "").split()) for e in entries for k in ("title", "summary")]
        sleep(5)  # arXiv API yêu cầu tối thiểu 3 giây giữa các request
    return out


def _ask(prompt: str, payload, call) -> list[str]:
    try:
        return parse_term_list(call([{"role": "system", "content": prompt},
                                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]))
    except Exception as e:
        print(f"cảnh báo: LLM lỗi: {e}", flush=True)
        return []


def main(per_cat: int = 1500, call=call_llm):
    os.makedirs("data/raw", exist_ok=True)
    texts = []
    for c in CATS:  # cache từng chuyên mục, lỗi giữa chừng không phải tải lại phần đã có
        cache = f"data/raw/arxiv-{c}.jsonl"
        if not os.path.exists(cache):
            got = fetch_arxiv(c, per_cat)
            with open(cache, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in got)
        with open(cache, encoding="utf-8") as f:
            texts += [json.loads(line) for line in f]
        print(f"arxiv {c}: tổng {len(texts) // 2} abstract", flush=True)

    with open("data/terms.jsonl", encoding="utf-8") as f:
        terms = [json.loads(line) for line in f]
    with open("data/enriched.jsonl", encoding="utf-8") as f:
        aliases = [a for line in f for a in json.loads(line)["aliases"]]
    known = {norm_key(t["term"]) for t in terms} | {norm_key(a) for a in aliases}
    known |= {k[:-1] for k in known if k.endswith("s")} | {k + "s" for k in known}
    base_words = {w.lower() for w in load_base("data/raw/vi-extract.jsonl.gz")}

    acr = {a: l for a, l in acronym_pairs(texts).items()
           if a.lower() not in known and l not in known and sum(f"({a}" in t for t in texts) >= 3}
    grams = [g for g, _ in candidate_ngrams(texts, known, base_words)[:2000]]
    d2l = [en for en, _ in parse_d2l_glossary(fetch(D2L_URL)) if norm_key(en) not in known]
    hf = {norm_key(t["term"]) for t in parse_hf_glossary(fetch(HF_URL))}
    corpus = "\n".join(texts).lower()
    ideas = [t for c in CATEGORIES for t in _ask(BRAINSTORM_PROMPT.format(cat=c), [], call)]
    ideas = [t for t in dict.fromkeys(ideas) if norm_key(t) not in known and t.lower() in corpus]  # bỏ từ LLM bịa
    cands = list(dict.fromkeys([f"{l} ({a})" for a, l in acr.items()] + d2l + ideas + grams))
    print(f"ứng viên: viết tắt={len(acr)} d2l={len(d2l)} LLM gợi ý={len(ideas)} n-gram={len(grams)} tổng={len(cands)}", flush=True)

    kept = []
    for i in range(0, len(cands), 150):
        kept += _ask(FILTER_PROMPT, cands[i:i + 150], call)
        print(f"lọc {min(i + 150, len(cands))}/{len(cands)}: giữ {len(kept)}", flush=True)
    lines, seen = [], set(known) | hf
    for t in kept:
        m = re.match(r"(.+?)\s*\(([A-Za-z0-9-]+)\)$", t)
        term, hint = (m.group(1), f"Also abbreviated as {m.group(2)}.") if m else (t, "")
        if norm_key(term) not in seen:
            seen.add(norm_key(term))
            lines.append(f"{term}\t{hint}" if hint else term)
    with open("data/mined_terms.txt", "w", encoding="utf-8") as f:
        f.write("# Thuật ngữ quét từ arXiv, bảng thuật ngữ D2L và gợi ý LLM (python -m aidict.mine). Sửa tay thoải mái.\n")
        f.writelines(line + "\n" for line in sorted(lines, key=str.lower))
    print(f"mined_terms.txt: {len(lines)} thuật ngữ mới")


if __name__ == "__main__":
    main()
