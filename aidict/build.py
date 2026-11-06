"""Gộp từ điển nền + thuật ngữ AI, sinh dạng biến tố, dựng HTML từng mục."""
import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys

from lemminflect import getAllInflections, getAllInflectionsOOV

from aidict.base import load_base

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


def assemble(base: dict, terms: list[dict], enriched: dict[str, dict],
             no_merge: frozenset | set = frozenset()) -> list[tuple[str, list[str], str]]:
    base_lc = {}
    for w in base:  # hai từ chỉ khác hoa/thường (act/ACT) thì ưu tiên bản chữ thường
        if w.lower() not in base_lc or w == w.lower():
            base_lc[w.lower()] = w
    ai, missing = {}, 0
    for t in terms:
        e = enriched.get(t["term"])
        if not e:
            missing += 1
            continue
        s = t["term"]
        # viết tắt (GAN), chữ hoa giữa từ (ReCoRD, IoU) và tên riêng trong no_merge (Pro, Gemini)
        # thành mục riêng phân biệt hoa/thường, không chiếm từ thường (gan, record, pro)
        if s.isupper() or s[1:] != s[1:].lower() or s in no_merge or s in base:
            hw = s
        else:
            hw = base_lc.get(s.lower(), s)
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


TITLE = "AI Dictionary Anh–Việt"
DESC = ("Từ điển Anh–Việt kèm 2.010 thuật ngữ AI có giải thích tiếng Việt. Nguồn: Wiktionary tiếng Việt "
        "(CC BY-SA, qua kaikki.org); Google ML Glossary (CC BY 4.0); Wikipedia Glossary of AI (CC BY-SA); "
        "Hugging Face Transformers glossary (Apache 2.0); bảng thuật ngữ D2L tiếng Việt; Stanford cheatsheets (MIT); "
        "thuật ngữ quét từ arXiv. Phần tiếng Việt do LLM sinh. Phát hành theo CC BY-SA 4.0: "
        "https://github.com/thucnc7/aidictionary")


def write_opf(entries: list[tuple[str, list[str], str]], raw_dir: str) -> str:
    """PyGlossary sinh OPF + XHTML dạng từ điển Kindle; kindlegen chạy riêng ở run_kindlegen."""
    from pyglossary.glossary_v2 import Glossary
    Glossary.init()
    glos = Glossary()
    for hw, alts, body in entries:
        glos.addEntry(glos.newEntry([hw, *alts], body, defiFormat="h"))
    glos.setInfo("title", TITLE)
    glos.setInfo("description", DESC)
    glos.sourceLangName = "English"
    glos.targetLangName = "Vietnamese"
    shutil.rmtree(raw_dir, ignore_errors=True)
    glos.write(raw_dir, formatName="Mobi", kindlegen_path="")
    return os.path.join(raw_dir, "OEBPS", "content.opf")


def run_kindlegen(opf: str, kindlegen: str, out_path: str, log_path: str) -> str:
    # cùng tham số PyGlossary dùng; tự chạy để giữ log và báo lỗi (PyGlossary chỉ log, không raise)
    r = subprocess.run([kindlegen, opf, "-gen_ff_mobi7", "-dont_append_source", "-o", "content.mobi"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(r.stdout + r.stderr)
    mobi = os.path.join(os.path.dirname(opf), "content.mobi")
    if not os.path.exists(mobi):
        sys.exit(f"kindlegen không tạo được file .mobi (exit {r.returncode}), xem {log_path}")
    shutil.copyfile(mobi, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-base", type=int, help="chỉ giữ N từ nền đầu tiên để build thử nhanh")
    a = ap.parse_args()
    kindlegen = os.environ.get("KINDLEGEN") or os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Amazon", "Kindle Previewer 3", "lib", "fc", "bin", "kindlegen.exe")
    if not os.path.exists(kindlegen):
        sys.exit(f"Không thấy kindlegen tại {kindlegen}. Cài Kindle Previewer 3 hoặc đặt biến KINDLEGEN.")
    base = load_base("data/raw/vi-extract.jsonl.gz")
    if a.limit_base:
        base = dict(list(base.items())[:a.limit_base])
    terms = _read_jsonl("data/terms.jsonl")
    enriched = load_enriched("data/enriched.jsonl", "data/overrides.jsonl")
    with open("data/no_merge_terms.txt", encoding="utf-8") as f:
        no_merge = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    os.makedirs("build", exist_ok=True)
    export_review(enriched, "build/review.csv")
    entries = assemble(base, terms, enriched, no_merge)
    print(f"{len(entries)} mục, đang chạy kindlegen...", flush=True)
    opf = write_opf(entries, "build/aidict")
    print(run_kindlegen(opf, kindlegen, "build/aidict.mobi", "build/kindlegen.log"))


if __name__ == "__main__":
    main()
