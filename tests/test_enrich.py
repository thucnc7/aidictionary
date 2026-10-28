import json
import time
import pytest
from aidict.enrich import content_of, parse_response, run

ITEM = {"term": "overfitting", "pos": "n", "vi": ["quá khớp"], "keep_en": True,
        "short_vi": "Mô hình học thuộc dữ liệu huấn luyện nên dự đoán kém trên dữ liệu mới.",
        "explain_vi": "...", "example_en": "...", "example_vi": "...", "related": ["regularization"],
        "expansion": "", "aliases": []}

def test_parse_response_fenced():
    text = "```json\n" + json.dumps([ITEM], ensure_ascii=False) + "\n```"
    assert parse_response(text, ["overfitting"]) == [ITEM]

def test_parse_response_rejects_bad():
    missing = {k: v for k, v in ITEM.items() if k != "short_vi"}
    other = {**ITEM, "term": "dropout"}
    assert parse_response(json.dumps([missing, other]), ["overfitting"]) == []
    assert parse_response("xin lỗi, tôi không thể", ["overfitting"]) == []

def test_parse_response_coerces_string_lists():
    # model đôi khi trả "vi": "đóng bao" thay vì ["đóng bao"]
    got = parse_response(json.dumps([{**ITEM, "vi": "quá khớp", "aliases": ""}]), ["overfitting"])
    assert got == [{**ITEM, "vi": ["quá khớp"], "aliases": []}]

def _files(tmp_path):
    terms, out, pairs = tmp_path / "terms.jsonl", tmp_path / "enriched.jsonl", tmp_path / "pairs.jsonl"
    terms.write_text("".join(json.dumps({"term": t, "definitions_en": [], "tags": [], "sources": ["manual"]}) + "\n"
                             for t in ["overfitting", "dropout"]), encoding="utf-8")
    pairs.write_text("", encoding="utf-8")
    return str(terms), str(out), str(pairs)

def test_run_resumes(tmp_path):
    calls = []
    def fake(messages):
        calls.append(messages)
        return json.dumps([ITEM])  # chỉ trả overfitting, thiếu dropout
    t, o, p = _files(tmp_path)
    assert run(t, o, p, call=fake) == 1
    assert run(t, o, p, call=fake) == 0
    assert "dropout" in calls[1][-1]["content"] and "overfitting" not in calls[1][-1]["content"]

def test_run_parallel_writes_all(tmp_path):
    def echo(messages):
        [t] = json.loads(messages[-1]["content"])
        return json.dumps([{**ITEM, "term": t["term"]}])
    t, o, p = _files(tmp_path)
    assert run(t, o, p, batch_size=1, workers=2, call=echo) == 2
    assert run(t, o, p, batch_size=1, workers=2, call=echo) == 0

def test_run_interrupt_cancels_pending_batches(tmp_path):
    calls = []
    def interrupted(messages):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        time.sleep(0.3)
        return "[]"
    terms = tmp_path / "terms.jsonl"
    terms.write_text("".join(json.dumps({"term": f"t{i}", "definitions_en": [], "tags": [], "sources": []}) + "\n"
                             for i in range(5)), encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(KeyboardInterrupt):
        run(str(terms), str(tmp_path / "out.jsonl"), str(tmp_path / "pairs.jsonl"), batch_size=1, workers=1, call=interrupted)
    time.sleep(1)
    assert len(calls) <= 2  # các batch còn chờ bị huỷ, không gọi LLM tiếp

def test_run_survives_errors(tmp_path):
    def boom(messages): raise OSError("network down")
    t, o, p = _files(tmp_path)
    assert run(t, o, p, call=boom) == 0

def test_content_of_json_and_sse():
    assert content_of('{"choices":[{"message":{"content":"xin chào"}}]}') == "xin chào"
    sse = ('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
           'data: {"choices":[{"delta":{"content":"xin "}}]}\n\n'
           'data: {"choices":[{"delta":{"content":"chào"}}]}\n\ndata: [DONE]\n')
    assert content_of(sse) == "xin chào"
