"""Thu thập thuật ngữ AI: Google ML Glossary, Wikipedia, cheatsheet Stanford (vi), danh sách thủ công."""
import json
import re
import urllib.request
from html.parser import HTMLParser

UA = "aidict/0.1 (personal dictionary build)"
GOOGLE_URL = "https://developers.google.com/machine-learning/glossary"
WIKI_URL = ("https://en.wikipedia.org/w/api.php?action=parse&page=Glossary_of_artificial_intelligence"
            "&prop=text&format=json&formatversion=2")
CHEAT_URL = "https://raw.githubusercontent.com/shervinea/cheatsheet-translation/master/vi/{}.md"
CHEAT_NAMES = ["cs-221-logic-models", "cs-229-deep-learning", "cs-229-linear-algebra",
               "cs-229-machine-learning-tips-and-tricks", "cs-229-probability", "cs-229-supervised-learning",
               "cs-229-unsupervised-learning", "cs-230-convolutional-neural-networks",
               "cs-230-deep-learning-tips-and-tricks", "cs-230-recurrent-neural-networks"]
_PAREN = re.compile(r"\s*\(.*\)$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


class _Google(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items, self.para, self.in_h2, self.in_icon = [], None, False, False

    def _close_para(self):
        if self.para is not None and self.items:
            text = _norm("".join(self.para))
            if text and not text.startswith(("See ", "Click the icon")):
                self.items[-1]["paras"].append(text)
        self.para = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "h2" and "data-text" in a and "hide-from-toc" in cls:
            self._close_para()
            self.items.append({"term": a["data-text"].strip(), "paras": [], "tags": []})
            self.in_h2 = True
        elif tag == "div" and "glossary-icon" in cls:
            self.in_icon = True
        elif tag == "p":
            self._close_para()
            self.para = []
        elif tag == "br" and self.para is not None:
            self.para.append(" ")

    def handle_endtag(self, tag):
        if tag == "h2":
            self.in_h2 = False
        elif tag == "div":
            self.in_icon = False
        elif tag == "p":
            self._close_para()

    def handle_data(self, data):
        if self.in_h2 or not self.items:
            return
        if self.in_icon:
            if data.strip():
                self.items[-1]["tags"].append(data.strip().lstrip("#"))
        elif self.para is not None:
            self.para.append(data)


def parse_google(html: str) -> list[dict]:
    p = _Google()
    p.feed(html)
    p.close()
    p._close_para()
    return [{"term": i["term"], "definition_en": " ".join(i["paras"]), "tags": i["tags"], "source": "google"}
            for i in p.items]


class _Wiki(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items, self.mode, self.buf, self.skip = [], None, [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "sup"):
            self.skip += 1
        elif tag in ("dt", "dd"):
            self.mode, self.buf = tag, []
        elif tag == "br" and self.mode:
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if tag in ("style", "sup"):
            self.skip -= 1
        elif tag == "dt" and self.mode == "dt":
            self.items.append({"term": _norm("".join(self.buf)), "defs": []})
            self.mode = None
        elif tag == "dd" and self.mode == "dd":
            if self.items:
                self.items[-1]["defs"].append(_norm("".join(self.buf)))
            self.mode = None
        elif tag in ("p", "li") and self.mode:
            self.buf.append(" ")

    def handle_data(self, data):
        if self.mode and not self.skip:
            self.buf.append(data)


def parse_wikipedia(html: str) -> list[dict]:
    p = _Wiki()
    p.feed(html)
    p.close()
    return [{"term": i["term"], "definition_en": " ".join(d for d in i["defs"] if d), "tags": [], "source": "wikipedia"}
            for i in p.items if i["term"]]


def parse_cheatsheet(md: str) -> list[tuple[str, str]]:
    pairs = re.findall(r"\*\*\d+\. (.+?)\*\*\s*\n\s*&#10230;[ \t]*(.*)", md)
    return [(en.strip(), vi.strip()) for en, vi in pairs if vi.strip()]


def load_manual(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            term, _, hint = line.partition("\t")
            out.append({"term": term.strip(), "definition_en": hint.strip(), "tags": [], "source": "manual"})
    return out


def norm_key(term: str) -> str:
    return _norm(_PAREN.sub("", term)).lower()


def merge_terms(lists: list[list[dict]]) -> list[dict]:
    out: dict[str, dict] = {}
    for lst in lists:
        for r in lst:
            k = norm_key(r["term"])
            if not k:
                continue
            m = out.setdefault(k, {"term": _norm(_PAREN.sub("", r["term"])), "definitions_en": [], "tags": [], "sources": []})
            if r["definition_en"] and r["definition_en"] not in m["definitions_en"]:
                m["definitions_en"].append(r["definition_en"])
            m["tags"] += [t for t in r["tags"] if t not in m["tags"]]
            if r["source"] not in m["sources"]:
                m["sources"].append(r["source"])
    return list(out.values())


def main():
    g = parse_google(fetch(GOOGLE_URL))
    w = parse_wikipedia(json.loads(fetch(WIKI_URL))["parse"]["text"])
    m = load_manual("data/manual_terms.txt")
    pairs = [p for n in CHEAT_NAMES for p in parse_cheatsheet(fetch(CHEAT_URL.format(n)))]
    terms = merge_terms([g, w, m])
    with open("data/terms.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in terms)
    with open("data/cheatsheet_pairs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(list(p), ensure_ascii=False) + "\n" for p in pairs)
    print(f"google={len(g)} wikipedia={len(w)} manual={len(m)} merged={len(terms)} pairs={len(pairs)}")


if __name__ == "__main__":
    main()
