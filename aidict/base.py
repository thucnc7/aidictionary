"""Từ điển nền Anh–Việt từ Wiktionary tiếng Việt (bản trích xuất kaikki.org)."""
import gzip
import json
import re

_STUB = re.compile(r"^[^.]{0,80}?(số nhiều|ngôi thứ ba|quá khứ|phân từ|so sánh)[^.]{0,60}?\bcủa ([\w'-]+)\.?$", re.I)


def load_base(path: str) -> dict[str, dict]:
    base: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("lang_code") != "en":
                continue
            e = base.setdefault(d["word"], {"ipa": "", "senses": [], "forms": []})
            if not e["ipa"]:
                e["ipa"] = next((s["ipa"].strip("/") for s in d.get("sounds", []) if s.get("ipa")), "")
            for s in d.get("senses", []):
                if s.get("glosses"):
                    e["senses"].append({"pos": d.get("pos", ""), "gloss": " ".join(s["glosses"])})
            for fm in d.get("forms", []):
                if "romanization" not in fm.get("tags", []) and fm["form"] not in e["forms"]:
                    e["forms"].append(fm["form"])
    # Mục chỉ là dạng biến tố ("Số nhiều của bias") chuyển thành tên phụ của từ gốc,
    # để Kindle tra "biases" ra nghĩa đầy đủ của "bias" thay vì một dòng trỏ sang.
    for w, e in base.items():
        real = []
        for s in e["senses"]:
            m = _STUB.match(s["gloss"])
            lemma = m and m.group(2)
            if lemma and lemma != w and lemma in base:
                if w not in base[lemma]["forms"]:
                    base[lemma]["forms"].append(w)
            else:
                real.append(s)
        e["senses"] = real
    return {w: e for w, e in base.items() if e["senses"]}
