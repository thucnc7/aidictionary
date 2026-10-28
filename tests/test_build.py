from aidict.build import assemble, forms

def S(gloss, pos="noun"): return {"ipa": "", "senses": [{"pos": pos, "gloss": gloss}], "forms": []}
BASE = {"transformer": S("Máy biến thế."), "rag": S("Giẻ rách."), "gan": S("Lá gan."),
        "embed": S("Gắn vào.", "verb"), "embedding": S("Dạng -ing của embed.", "verb")}
def E(term, **kw):
    return {"term": term, "pos": "n", "vi": ["x"], "keep_en": False, "short_vi": "ngắn", "explain_vi": "gt",
            "example_en": "ex", "example_vi": "vd", "related": [], "expansion": "", "aliases": [], **kw}
def T(term, defs=()): return {"term": term, "definitions_en": list(defs), "tags": [], "sources": ["google"]}
TERMS = [T("Transformer", ["A model <b> & co"]), T("retrieval-augmented generation"), T("embedding"), T("GAN")]
ENR = {"Transformer": E("Transformer"), "retrieval-augmented generation": E("retrieval-augmented generation", aliases=["RAG"]),
       "embedding": E("embedding"), "GAN": E("GAN", pos="abbr")}

def entries(): return {hw: (alts, html) for hw, alts, html in assemble(BASE, TERMS, ENR)}

def test_ai_sense_first_base_kept():
    alts, html = entries()["transformer"]
    assert html.index("[AI]") < html.index("Nghĩa thông thường") < html.index("Máy biến thế")
    assert {"transformers", "Transformers", "Transformer"} <= set(alts)

def test_acronym_does_not_steal_base_word():
    e = entries()
    assert "RAG" in e["retrieval-augmented generation"][0]
    assert "[AI]" not in e["rag"][1] and "[AI]" not in e["gan"][1]
    assert "[AI]" in e["GAN"][1]

def test_inflection_colliding_headword_dropped():
    e = entries()
    assert "embedding" not in e["embed"][0]
    assert "embeddings" in e["embedding"][0]
    assert e["embedding"][1].startswith("<p><b>[AI]</b>")

def test_render_escapes_html():
    assert "A model &lt;b&gt; &amp; co" in entries()["transformer"][1]

def test_exact_case_base_word_preferred():
    base = {"act": S("Hành động."), "ACT": S("Kỳ thi ACT.")}  # thứ tự như Wiktionary: act trước ACT
    e = {hw: html for hw, _, html in assemble(base, [T("act")], {"act": E("act")})}
    assert "[AI]" in e["act"] and "[AI]" not in e["ACT"]

def test_mixed_case_and_no_merge_terms_stay_separate():
    base = {"record": S("Hồ sơ."), "pro": S("Tán thành.")}
    e = {hw: html for hw, _, html in assemble(base, [T("ReCoRD"), T("Pro")], {"ReCoRD": E("ReCoRD"), "Pro": E("Pro")},
                                              no_merge={"Pro"})}
    assert "[AI]" not in e["record"] and "[AI]" in e["ReCoRD"]
    assert "[AI]" not in e["pro"] and "[AI]" in e["Pro"]

def test_forms():
    assert {"queries"} <= forms("query", "n")
    assert {"fine-tuned", "fine-tuning", "fine-tunes"} <= forms("fine-tune", "v", oov=True)
    assert {"neural networks"} <= forms("neural network", "n")
    assert forms("LLM", "n") == {"LLMs"}
