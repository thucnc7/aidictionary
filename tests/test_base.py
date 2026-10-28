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

def test_inflection_stub_entries_fold_into_lemma(tmp_path):
    rows = [
        {"word": "bias", "lang_code": "en", "pos": "noun", "senses": [{"glosses": ["Thành kiến."]}]},
        {"word": "biases", "lang_code": "en", "pos": "noun", "senses": [{"glosses": ["Số nhiều của bias"]}, {"glosses": ["Dạng trần thuật hiện tại đơn ở ngôi thứ ba số ít của bias"]}]},
        {"word": "learn", "lang_code": "en", "pos": "verb", "senses": [{"glosses": ["Học."]}]},
        {"word": "learning", "lang_code": "en", "pos": "noun", "senses": [{"glosses": ["Dạng phân từ hiện tại và danh động từ (gerund) của learn."]}, {"glosses": ["Sự học."]}]},
        {"word": "colour", "lang_code": "en", "pos": "noun", "senses": [{"glosses": ["Quá khứ của ông."]}]},
    ]
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    base = load_base(str(p))
    assert "biases" not in base and "biases" in base["bias"]["forms"]
    assert base["learning"]["senses"] == [{"pos": "noun", "gloss": "Sự học."}]
    assert "learning" in base["learn"]["forms"]
    assert base["colour"]["senses"] == [{"pos": "noun", "gloss": "Quá khứ của ông."}]  # "ông" không phải từ tiếng Anh trong base
