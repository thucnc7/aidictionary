from aidict.mine import acronym_pairs, ask_all, candidate_ngrams, fetch_arxiv, parse_term_list, singular, stable_batches
from aidict.sources import parse_d2l_glossary, parse_hf_glossary

def test_parse_d2l_glossary():
    md = ("## A\n| English | Tiếng Việt | Thảo luận tại |\n|---|---|---|\n"
          "| accuracy                | độ chính xác  |          |\n"
          "| adversarial learning    | học đối kháng | [https://git.io/x](https://git.io/x) |\n")
    assert parse_d2l_glossary(md) == [("accuracy", "độ chính xác"), ("adversarial learning", "học đối kháng")]

def test_parse_hf_glossary():
    md = ("# Glossary\n\nIntro text.\n\n## A\n\n### attention mask\n\nThe attention mask is an optional argument.\n\n"
          '<Youtube id="M6adb1j2jPI"/>\n\nIt tells the model which [tokens](#token) to attend to.\n\n'
          "### connectionist temporal classification (CTC)\n\nAn algorithm that `aligns` sequences.\n")
    assert parse_hf_glossary(md) == [
        {"term": "attention mask", "definition_en": "The attention mask is an optional argument. It tells the model which tokens to attend to.",
         "tags": [], "source": "huggingface"},
        {"term": "connectionist temporal classification (CTC)", "definition_en": "An algorithm that aligns sequences.",
         "tags": [], "source": "huggingface"}]

def test_acronym_pairs():
    texts = ["We study retrieval-augmented generation (RAG) for QA.",
             "Large language models (LLMs) are strong. Mixture-of-Experts (MoE) layers route tokens.",
             "Our method (OURS) beats the baseline."]
    got = acronym_pairs(texts)
    assert got["RAG"] == "retrieval-augmented generation"
    assert got["LLM"] == "large language model"
    assert got["MoE"] == "mixture-of-experts"
    assert "OURS" not in got  # chữ cái đầu không khớp

def test_candidate_ngrams():
    texts = ["We apply speculative decoding to speed up inference."] * 4 + ["The results of speculative decoding."]
    got = dict(candidate_ngrams(texts, known={"inference"}, base_words={"speculative", "decoding", "speed"}, min_df=4))
    assert got["speculative decoding"] == 5
    assert "inference" not in got        # đã có trong từ điển AI
    assert "speculative" not in got      # từ đơn có sẵn trong từ điển nền
    assert not any(k.startswith(("we ", "the ")) or k.endswith(" to") for k in got)

def test_fetch_arxiv_backs_off_on_429():
    import urllib.error
    atom = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Speculative  decoding</title>'
            '<summary>We study it.</summary></entry></feed>')
    calls, sleeps = [], []
    def get(url):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return atom
    assert fetch_arxiv("cs.CL", 1, page=1, get=get, sleep=sleeps.append) == ["Speculative decoding", "We study it."]
    assert len(calls) == 2 and sleeps[0] >= 60

def test_ask_all_caches_results(tmp_path):
    import json
    calls = []
    def echo(messages):
        calls.append(1)
        return json.dumps(json.loads(messages[-1]["content"]))
    jobs = [("p", ["a"]), ("p", ["b"])]
    cache = str(tmp_path / "cache.jsonl")
    assert ask_all(jobs, echo, cache) == [["a"], ["b"]]
    assert ask_all(jobs, echo, cache) == [["a"], ["b"]]
    assert len(calls) == 2  # lần chạy sau lấy từ cache, không gọi LLM lại

def test_stable_batches_change_only_one_bucket():
    cands = [f"term {i}" for i in range(300)]
    before = stable_batches(cands, 20)
    after = stable_batches(cands + ["brand new term"], 20)
    assert sum(b != a for b, a in zip(before, after)) == 1
    assert sorted(t for b in before for t in b) == sorted(cands)

def test_parse_term_list():
    assert parse_term_list('Here:\n```json\n["KV cache", "LoRA", 3, ""]\n```') == ["KV cache", "LoRA"]
    assert parse_term_list("không có") == []

def test_singular_headwords():
    assert singular("verifiers") == "verifier"
    assert singular("coding agents") == "coding agent"
    assert singular("frontier LLMs") == "frontier LLM"
    assert singular("internal representations") == "internal representation"
    for keep in ("time series", "physics", "loss", "training data", "bias", "Qwen3-VL", "ethics"):
        assert singular(keep) == keep
