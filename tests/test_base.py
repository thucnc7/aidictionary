import gzip, json
from aidict.base import load_base

def test_load_base(tmp_path):
    rows = [
        {"word": "transformer", "lang_code": "en", "pos": "noun", "senses": [{"glosses": ["Máy biến thế."]}], "sounds": [{"ipa": "/trænsˈfɔːmə/"}]},
        {"word": "transformer", "lang_code": "en", "pos": "verb", "senses": [{}]},
        {"word": "emptyword", "lang_code": "en", "pos": "noun", "senses": [{}]},
        {"word": "trở thành", "lang_code": "vi", "pos": "verb", "senses": [{"glosses": ["x"]}]},
        {"word": "attention", "lang_code": "en", "pos": "noun", "senses": [{"glosses": ["Sự chú ý."]}], "forms": [{"form": "attentions"}]},
    ]
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    base = load_base(str(p))
    assert set(base) == {"transformer", "attention"}
    assert base["transformer"] == {"ipa": "trænsˈfɔːmə", "senses": [{"pos": "noun", "gloss": "Máy biến thế."}], "forms": []}
    assert base["attention"]["forms"] == ["attentions"]
