"""Từ điển nền Anh–Việt từ Wiktionary tiếng Việt (bản trích xuất kaikki.org)."""
import gzip
import json


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
    return {w: e for w, e in base.items() if e["senses"]}
