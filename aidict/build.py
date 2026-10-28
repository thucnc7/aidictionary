"""Gộp từ điển nền + thuật ngữ AI, sinh dạng biến tố, dựng HTML từng mục."""
import csv
import html
import json
import os

from lemminflect import getAllInflections, getAllInflectionsOOV

UPOS = {"n": "NOUN", "noun": "NOUN", "v": "VERB", "verb": "VERB", "adj": "ADJ", "adv": "ADV"}
POS_VI = {"noun": "danh từ", "n": "danh từ", "verb": "động từ", "v": "động từ", "adj": "tính từ",
          "adv": "phó từ", "phrase": "cụm từ", "abbr": "viết tắt"}
x = html.escape


def forms(word: str, pos: str, oov: bool = False) -> set[str]:
    if word.isupper() and len(word) >= 2:
        return {word + "s"}
    upos = UPOS.get(pos)
    if not upos:
        return set()
    head, _, last = word.rpartition(" ")
    infl = getAllInflections(last, upos=upos) or (getAllInflectionsOOV(last, upos=upos) if oov else {})
    prefix = head + " " if head else ""
    return {prefix + f for fs in infl.values() for f in fs} - {word}


def variants(term: str) -> set[str]:
    out = {term[:1].upper() + term[1:]}
    if "-" in term:
        out |= {term.replace("-", " "), term.replace("-", "")}
    if " " in term:
        out.add(term.replace(" ", "-"))
    return out - {term}


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_enriched(path: str, overrides_path: str) -> dict[str, dict]:
    d = {e["term"]: e for e in _read_jsonl(path)}
    for o in _read_jsonl(overrides_path):
        d[o["term"]] = {**d.get(o["term"], {}), **o}
    return d


def render_ai(e: dict, definitions_en: list[str]) -> str:
    p = [f"<p><b>[AI]</b> {x(e['short_vi'])}</p>"]
    if e["vi"]:
        keep = " — thường giữ nguyên tiếng Anh" if e["keep_en"] else ""
        p.append(f"<p><i>Tiếng Việt:</i> {x('; '.join(e['vi']))}{keep}</p>")
    if e["expansion"]:
        p.append(f"<p><i>Viết tắt của:</i> {x(e['expansion'])}</p>")
    p.append(f"<p>{x(e['explain_vi'])}</p>")
    p.append(f"<p><i>Ví dụ:</i> {x(e['example_en'])}<br/>→ {x(e['example_vi'])}</p>")
    if e["related"]:
        p.append(f"<p><i>Liên quan:</i> {x(', '.join(e['related']))}</p>")
    if definitions_en:
        d = definitions_en[0]
        p.append(f"<p><i>EN:</i> {x(d[:300] + '…' if len(d) > 300 else d)}</p>")
    return "".join(p)


def render_base(b: dict) -> str:
    p = [f"<p>/{x(b['ipa'])}/</p>"] if b["ipa"] else []
    p += [f"<p><i>{x(POS_VI.get(s['pos'], s['pos']))}</i> {x(s['gloss'])}</p>" for s in b["senses"][:10]]
    return "".join(p)


def assemble(base: dict, terms: list[dict], enriched: dict[str, dict]) -> list[tuple[str, list[str], str]]:
    base_lc = {w.lower(): w for w in base}  # ponytail: hai từ chỉ khác hoa/thường (Bush/bush) thì lấy từ sau
    ai, missing = {}, 0
    for t in terms:
        e = enriched.get(t["term"])
        if not e:
            missing += 1
            continue
        # từ viết tắt viết hoa không gộp vào từ thường (GAN ≠ gan)
        hw = t["term"] if t["term"].isupper() else base_lc.get(t["term"].lower(), t["term"])
        ai[hw] = (e, t)
    if missing:
        print(f"{missing} thuật ngữ chưa có bản dịch, bỏ qua")
    heads = set(base) | set(ai)
    out = []
    for hw in sorted(heads):
        alts, parts = set(), []
        if hw in ai:
            e, t = ai[hw]
            for w in [hw, *e["aliases"]]:
                for f in {w} | forms(w, e["pos"], oov=True):
                    alts |= {f} | variants(f)
            parts.append(render_ai(e, t["definitions_en"]))
        if hw in base:
            alts |= set(base[hw]["forms"])
            for s in base[hw]["senses"]:
                alts |= forms(hw, s["pos"])
            parts.append(("<hr/><p><b>Nghĩa thông thường</b></p>" if parts else "") + render_base(base[hw]))
        # ponytail: chưa rõ Kindle có phân biệt hoa/thường khi tra tên phụ không; kiểm tra trên máy thật (Task 5)
        alts -= heads  # không lấy headword của mục khác (embedding, rag…); cũng loại chính hw
        out.append((hw, sorted(alts), "".join(parts)))
    return out


def export_review(enriched: dict, path: str):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "vi", "keep_en", "short_vi"])
        for e in enriched.values():
            w.writerow([e["term"], "; ".join(e["vi"]), e["keep_en"], e["short_vi"]])
