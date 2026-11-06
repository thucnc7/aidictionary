# AI Dictionary Anh–Việt cho Kindle

Từ điển Anh–Việt dùng trên máy đọc sách Kindle, dành cho người đọc paper và sách về trí tuệ nhân tạo. Ngoài khoảng 100.000 từ tiếng Anh thông dụng, từ điển có thêm **2.010 thuật ngữ AI**. Mỗi thuật ngữ có bản dịch, giải thích khái niệm bằng tiếng Việt, câu ví dụ song ngữ và các thuật ngữ liên quan.

Khi đọc sách trên Kindle, chạm giữ vào một từ như `attention`, `KV cache` hay `sycophancy` là thấy ngay nghĩa chuyên ngành AI. Nghĩa thông thường của từ (nếu có) hiện tiếp bên dưới.

## Ví dụ một mục từ

> **sycophancy** *(danh từ)*: thói xu nịnh, sự xu nịnh
>
> Xu hướng của mô hình AI chiều theo hoặc tâng bốc người dùng, nói điều họ muốn nghe thay vì điều đúng.
>
> Một mô hình có sycophancy sẽ đồng tình với quan điểm của người dùng, khen quá mức hoặc đổi câu trả lời đúng thành sai khi người dùng phản bác, dù lẽ ra phải giữ lập trường. Hiện tượng này khác với ảo giác (hallucination): ở đây mô hình có thể biết đáp án đúng nhưng vẫn chiều lòng người dùng.
>
> *Ví dụ:* Reducing sycophancy is important so that the assistant does not simply agree with incorrect claims from the user.
> → Giảm thói xu nịnh là điều quan trọng để trợ lý không chỉ đơn thuần đồng tình với những nhận định sai của người dùng.
>
> *Liên quan:* RLHF, alignment, reward hacking, hallucination, preference data

## Cài lên Kindle

1. Tải file `aidict.mobi` ở mục [Releases](../../releases).
2. Cắm Kindle vào máy tính, chép file vào thư mục `documents/dictionaries/`. Nếu thư mục chưa có thì chép vào `documents/`.
3. Rút cáp, trên Kindle vào **Cài đặt → Ngôn ngữ và từ điển → Từ điển → Tiếng Anh** rồi chọn **AI Dictionary Anh–Việt**.

Tra được cả dạng biến đổi của từ (`models` → `model`, `embeddings` → `embedding`), từ viết tắt (`RAG`, `LoRA`, `GRPO`) và cụm nhiều chữ (`gradient descent`, `mixture of experts`).

## Có gì trong từ điển

| Thành phần | Số lượng | Nguồn |
|---|---|---|
| Từ tiếng Anh thông dụng | ~100.000 | Wiktionary tiếng Việt (bản trích xuất của [kaikki.org](https://kaikki.org)) |
| Thuật ngữ ML cơ bản | 739 | [Google Machine Learning Glossary](https://developers.google.com/machine-learning/glossary) |
| Thuật ngữ AI tổng quát | 399 | [Wikipedia: Glossary of artificial intelligence](https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence) |
| Thuật ngữ Transformers | 51 | [Hugging Face Transformers glossary](https://huggingface.co/docs/transformers/glossary) |
| Thuật ngữ thời LLM bổ sung tay | 61 | `data/manual_terms.txt` |
| Thuật ngữ mới từ paper và sách | 854 | Quét từ 6.400 abstract arXiv 2026 (cs.CL, cs.LG, cs.AI, cs.CV) và bảng thuật ngữ sách [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn) |

Một số thuật ngữ có ở nhiều nguồn nên tổng sau khi gộp là 2.010. Cách dịch tham khảo bảng thuật ngữ D2L tiếng Việt và bản dịch tiếng Việt của [cheatsheet Stanford CS 221/229/230](https://github.com/shervinea/cheatsheet-translation).

Phần tiếng Việt (bản dịch, giải thích, ví dụ) do LLM viết:

- 776 thuật ngữ đầu: Gemini 3.8 Flash
- 350 thuật ngữ tiếp theo: Nemotron 3 Ultra (chất lượng thấp hơn, nên rà lại)
- 884 thuật ngữ mới từ paper và sách: Claude Sonnet 5.5

LLM có thể dịch sai. Nếu gặp lỗi, bạn sửa theo hướng dẫn ở mục [Sửa bản dịch](#sửa-bản-dịch) hoặc mở issue.

## Tự build lại

Cần Python 3 (đã chạy thử trên 3.14) và [Kindle Previewer 3](https://www.amazon.com/Kindle-Previewer/b?node=21381691011) (có sẵn `kindlegen` bên trong).

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # Linux/macOS: .venv/bin/pip
mkdir -p data/raw
curl -L https://kaikki.org/dictionary/downloads/vi/vi-extract.jsonl.gz -o data/raw/vi-extract.jsonl.gz
```

Pipeline gồm 4 bước. Bước nào cũng ghi ra file trung gian trong `data/`, nên có thể chạy lại riêng từng bước:

| Bước | Lệnh | Việc làm |
|---|---|---|
| 1. Quét thuật ngữ mới | `python -X utf8 -m aidict.mine` | Tải abstract arXiv, lọc cụm từ và viết tắt hay gặp, nhờ LLM giữ lại thuật ngữ thật → `data/mined_terms.txt` |
| 2. Thu thập | `python -X utf8 -m aidict.sources` | Gộp các glossary và danh sách ở bước 1 → `data/terms.jsonl` |
| 3. Dịch và giải thích | `python -X utf8 -m aidict.enrich` | Gọi LLM cho các thuật ngữ chưa có trong `data/enriched.jsonl`. Có thể dừng giữa chừng rồi chạy tiếp |
| 4. Build | `python -X utf8 -m aidict.build` | Gộp với từ điển nền, sinh dạng biến đổi của từ → `build/aidict.mobi` |

Dữ liệu đã dịch đều có sẵn trong repo. Nếu chỉ muốn build lại file `.mobi` thì chạy bước 4 là đủ, không cần LLM.

Bước 1 và 3 gọi LLM qua một API tương thích OpenAI, cấu hình bằng biến môi trường:

```bash
export LLM_BASE_URL=https://your-gateway/v1      # endpoint /chat/completions tương thích OpenAI
export LLM_MODEL=ag/gemini-3.8-flash-high
export NINEROUTER_API_KEY=...                     # hoặc ghi key vào ~/.config/9router_key
```

Nếu `kindlegen` không nằm ở đường dẫn mặc định của Kindle Previewer trên Windows, đặt biến `KINDLEGEN` trỏ tới file đó. Trên Windows nhớ chạy Python với `-X utf8`, nếu không console sẽ báo lỗi mã hoá ký tự tiếng Việt.

Chạy test: `python -X utf8 -m pytest -q`

## Sửa bản dịch

- **Sửa nghĩa một thuật ngữ:** mở `build/review.csv` (sinh ra sau mỗi lần build) để rà, rồi ghi phần sửa vào `data/overrides.jsonl`, mỗi dòng một mục. Chỉ cần ghi các trường muốn đổi, ví dụ `{"term": "attention", "vi": ["cơ chế chú ý"]}`. Sau đó build lại.
- **Bỏ hoặc thêm thuật ngữ:** sửa `data/mined_terms.txt` hoặc `data/manual_terms.txt`, chạy lại bước 2, 3 và 4.
- **Tên riêng trùng từ thông dụng** (ví dụ tên sản phẩm `Gemini`, `Flax`): thêm vào `data/no_merge_terms.txt` để chúng không đè lên nghĩa thường của từ.

## Cấu trúc thư mục

```
aidict/
  base.py      đọc từ điển nền Wiktionary tiếng Việt
  sources.py   tải và phân tích các glossary
  mine.py      quét thuật ngữ mới từ arXiv và sách
  enrich.py    gọi LLM để dịch và giải thích
  build.py     dựng HTML và gọi kindlegen ra .mobi
data/          thuật ngữ, bản dịch đã sinh, danh sách sửa tay
tests/         test cho từng bước (pytest)
```

## Giấy phép dữ liệu

File từ điển dùng nội dung từ Wiktionary và Wikipedia (CC BY-SA), Google ML Glossary (CC BY 4.0), Hugging Face Transformers (Apache 2.0), cheatsheet Stanford (MIT) và bảng thuật ngữ D2L tiếng Việt. Vì có nội dung CC BY-SA, khi chia sẻ lại file từ điển hoặc dữ liệu trong `data/` cần ghi nguồn và giữ cùng giấy phép CC BY-SA 4.0.
