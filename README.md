<div align="center">

# AI Dictionary Anh–Việt cho Kindle

**Tra nghĩa thuật ngữ trí tuệ nhân tạo ngay trên Kindle khi đọc paper và sách AI tiếng Anh**

[![Phiên bản](https://img.shields.io/github/v/release/thucnc7/aidictionary?label=phi%C3%AAn%20b%E1%BA%A3n)](https://github.com/thucnc7/aidictionary/releases/latest)
[![Lượt tải](https://img.shields.io/github/downloads/thucnc7/aidictionary/total?label=l%C6%B0%E1%BB%A3t%20t%E1%BA%A3i)](https://github.com/thucnc7/aidictionary/releases)
[![Dữ liệu: CC BY-SA 4.0](https://img.shields.io/badge/d%E1%BB%AF%20li%E1%BB%87u-CC%20BY--SA%204.0-lightgrey)](#giấy-phép)
[![Mã nguồn: MIT](https://img.shields.io/badge/m%C3%A3%20ngu%E1%BB%93n-MIT-blue)](LICENSE)

2.010 thuật ngữ AI · dịch, giải thích và ví dụ tiếng Việt · kèm ~100.000 từ tiếng Anh thông dụng

**[⬇ Tải từ điển (aidict.mobi, 12 MB)](https://github.com/thucnc7/aidictionary/releases/latest/download/aidict.mobi)**

</div>

---

Từ điển Anh–Việt phổ thông trên Kindle thường không có nghĩa AI: `transformer` chỉ là "vật làm biến đổi" hay "máy biến thế", còn `overfitting`, `fine-tuning` hay `KV cache` thì không có. Từ điển này bổ sung lớp thuật ngữ AI lên trên một từ điển Anh–Việt đầy đủ. Khi chạm giữ vào một từ, Kindle hiện nghĩa chuyên ngành AI trước, nghĩa thông thường (nếu có) ngay bên dưới.

Mỗi thuật ngữ gồm:

- bản dịch tiếng Việt, hoặc giữ nguyên tiếng Anh nếu giới AI Việt Nam quen dùng như vậy (`KV cache`, `prompt`)
- định nghĩa ngắn một câu và đoạn giải thích khái niệm
- câu ví dụ song ngữ
- dạng đầy đủ của từ viết tắt và các thuật ngữ liên quan

> **sycophancy** *(danh từ)*: thói xu nịnh, sự xu nịnh
>
> Xu hướng của mô hình AI chiều theo hoặc tâng bốc người dùng, nói điều họ muốn nghe thay vì điều đúng.
>
> Một mô hình có sycophancy sẽ đồng tình với quan điểm của người dùng, khen quá mức hoặc đổi câu trả lời đúng thành sai khi người dùng phản bác. Hiện tượng này khác với ảo giác (hallucination): ở đây mô hình có thể biết đáp án đúng nhưng vẫn chiều lòng người dùng.
>
> *Ví dụ:* Reducing sycophancy is important so that the assistant does not simply agree with incorrect claims from the user.
> → Giảm thói xu nịnh là điều quan trọng để trợ lý không chỉ đơn thuần đồng tình với những nhận định sai của người dùng.
>
> *Liên quan:* RLHF, alignment, reward hacking, hallucination, preference data

Có hai cách dùng repo này:

| | Dành cho | Cần chuẩn bị |
|---|---|---|
| [**Cách 1: Tải về dùng ngay**](#cách-1-tải-về-dùng-ngay) | Người chỉ muốn có từ điển trên Kindle | Kindle và cáp USB |
| [**Cách 2: Tự build từ điển riêng**](#cách-2-tự-build-từ-điển-riêng) | Người muốn thêm thuật ngữ, sửa bản dịch hoặc dùng LLM khác | Python, Kindle Previewer, API key LLM (chỉ khi dịch thuật ngữ mới) |

## Cách 1: Tải về dùng ngay

1. Tải file **[aidict.mobi](https://github.com/thucnc7/aidictionary/releases/latest/download/aidict.mobi)**. Các bản cũ hơn xem ở mục [Releases](https://github.com/thucnc7/aidictionary/releases).
2. Cắm Kindle vào máy tính bằng cáp USB, rồi chép file vào thư mục `documents/dictionaries/` trên Kindle. Nếu thư mục chưa có thì chép vào `documents/`.
3. Rút cáp. Mở một sách tiếng Anh, chạm giữ vào một từ, rồi chạm vào tên từ điển trong khung tra cứu và chọn **AI Dictionary Anh–Việt**.
   Hoặc vào **Cài đặt → Ngôn ngữ và từ điển → Từ điển → Tiếng Anh** để đặt làm từ điển mặc định.

Tra được cả dạng biến đổi của từ (`models` → `model`, `embeddings` → `embedding`), từ viết tắt (`RAG`, `LoRA`, `GRPO`) và cụm nhiều chữ (`gradient descent`, `mixture of experts`).

> [!NOTE]
> - Dùng được trên mọi máy Kindle e-ink (Basic, Paperwhite, Oasis, Scribe, Colorsoft).
> - Không dùng được trong ứng dụng Kindle cho điện thoại hoặc máy tính, vì các ứng dụng này không cho thêm từ điển ngoài.
> - Kindle đời mới kết nối với máy tính dạng MTP. Trên macOS cần thêm ứng dụng như [OpenMTP](https://openmtp.ganeshrvel.com) để chép file.

## Cách 2: Tự build từ điển riêng

Phù hợp khi bạn muốn sửa bản dịch, thêm thuật ngữ cho lĩnh vực của mình, hoặc quét thuật ngữ mới từ arXiv. Toàn bộ thuật ngữ đã dịch có sẵn trong `data/`, nên phần lớn thay đổi không cần gọi LLM. Khi thêm thuật ngữ mới, LLM chỉ được gọi cho những thuật ngữ chưa có bản dịch.

### Cài đặt

Cần Python 3 (đã chạy thử trên 3.14) và [Kindle Previewer 3](https://www.amazon.com/Kindle-Previewer/b?node=21381691011) cho Windows hoặc macOS. Kindle Previewer có sẵn công cụ `kindlegen` để đóng gói file `.mobi`.

```bash
git clone https://github.com/thucnc7/aidictionary.git
cd aidictionary
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Từ điển nền Anh–Việt (Wiktionary tiếng Việt, ~34 MB)
mkdir -p data/raw
curl -L https://kaikki.org/dictionary/downloads/vi/vi-extract.jsonl.gz -o data/raw/vi-extract.jsonl.gz
```

Trên macOS, hoặc khi cài Kindle Previewer vào chỗ khác, hãy chỉ đường dẫn tới `kindlegen`:

```bash
export KINDLEGEN="/Applications/Kindle Previewer 3.app/Contents/lib/fc/bin/kindlegen"
```

### Build

```bash
python -X utf8 -m aidict.build
```

Kết quả nằm trong thư mục `build/`:

- `aidict.mobi`: file từ điển để chép vào Kindle
- `review.csv`: bảng toàn bộ thuật ngữ AI, mở bằng Excel để rà bản dịch

Tham số `-X utf8` giúp Windows không báo lỗi mã hoá ký tự tiếng Việt. Muốn build thử nhanh thì thêm `--limit-base 2000`: chỉ lấy 2.000 từ nền nhưng vẫn đủ thuật ngữ AI.

### Tuỳ biến

| Muốn | Sửa | Rồi chạy |
|---|---|---|
| Sửa bản dịch của một thuật ngữ | Thêm một dòng vào `data/overrides.jsonl`, chỉ cần ghi các trường muốn đổi, ví dụ `{"term": "attention", "vi": ["cơ chế chú ý"]}` | `build` |
| Thêm thuật ngữ | Thêm vào `data/manual_terms.txt`, mỗi dòng một thuật ngữ. Có thể thêm gợi ý nghĩa tiếng Anh sau một dấu Tab | `sources` → `enrich` → `build` |
| Bỏ thuật ngữ | Xoá dòng tương ứng trong `data/mined_terms.txt` hoặc `data/manual_terms.txt` | `sources` → `build` |
| Giữ nguyên nghĩa thường của một từ bị tên riêng đè lên (ví dụ `Gemini`, `Flax`) | Thêm vào `data/no_merge_terms.txt` | `build` |
| Quét thuật ngữ mới từ arXiv | Chọn chuyên mục trong `CATS` ở `aidict/mine.py` | `mine` → `sources` → `enrich` → `build` |
| Đổi tên hoặc mô tả từ điển | `TITLE`, `DESC` trong `aidict/build.py` | `build` |

Mỗi bước là một lệnh `python -X utf8 -m aidict.<bước>`:

| Bước | Việc làm | Ghi ra |
|---|---|---|
| `mine` | Tải abstract mới nhất trên arXiv, lọc cụm từ và từ viết tắt hay gặp, rồi nhờ LLM giữ lại thuật ngữ thật | `data/mined_terms.txt` |
| `sources` | Tải các glossary (Google, Wikipedia, Hugging Face, D2L, cheatsheet Stanford) rồi gộp với danh sách thuật ngữ trong `data/` | `data/terms.jsonl` |
| `enrich` | Gọi LLM để dịch và giải thích những thuật ngữ chưa có bản dịch. Có thể dừng giữa chừng rồi chạy tiếp | `data/enriched.jsonl` |
| `build` | Gộp với từ điển nền, sinh dạng biến đổi của từ, đóng gói bằng kindlegen | `build/aidict.mobi` |

### Cấu hình LLM

Bước `mine` và `enrich` gọi LLM qua API dạng `/chat/completions` tương thích OpenAI, nên dùng được OpenAI, OpenRouter, Gemini, Ollama, LiteLLM và các dịch vụ tương tự:

```bash
export LLM_BASE_URL=https://api.openai.com/v1    # địa chỉ API, phần trước /chat/completions
export LLM_MODEL=<tên model>
export LLM_API_KEY=<API key>
```

Thêm khoảng 100 thuật ngữ mới tốn khoảng 13 lượt gọi (8 thuật ngữ mỗi lượt). Dùng `--workers` để chỉnh số lượt gọi song song (mặc định 4) và `--limit` để dịch thử vài thuật ngữ trước.

### Chạy test

```bash
python -X utf8 -m pytest -q
```

### Cấu trúc thư mục

```
aidict/
  base.py      đọc từ điển nền Wiktionary tiếng Việt
  sources.py   tải và phân tích các glossary
  mine.py      quét thuật ngữ mới từ arXiv và sách
  enrich.py    gọi LLM để dịch và giải thích
  build.py     dựng HTML và đóng gói .mobi
data/          danh sách thuật ngữ, bản dịch đã sinh, các file sửa tay
tests/         test cho từng bước (pytest)
```

## Trong từ điển có gì

| Thành phần | Số lượng | Nguồn |
|---|---|---|
| Từ tiếng Anh thông dụng | ~100.000 | Wiktionary tiếng Việt, bản trích xuất của [kaikki.org](https://kaikki.org) |
| Thuật ngữ học máy cơ bản | 739 | [Google Machine Learning Glossary](https://developers.google.com/machine-learning/glossary) |
| Thuật ngữ AI tổng quát | 399 | [Wikipedia: Glossary of artificial intelligence](https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence) |
| Thuật ngữ Transformers | 51 | [Hugging Face Transformers glossary](https://huggingface.co/docs/transformers/glossary) |
| Thuật ngữ thời LLM chọn tay | 61 | `data/manual_terms.txt` |
| Thuật ngữ mới từ paper và sách | 854 | 6.400 abstract arXiv năm 2026 (cs.CL, cs.LG, cs.AI, cs.CV) và bảng thuật ngữ sách [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn) |

Một số thuật ngữ có mặt ở nhiều nguồn, nên sau khi gộp còn 2.010. Cách dịch có tham khảo bảng thuật ngữ D2L tiếng Việt và bản dịch tiếng Việt của [cheatsheet Stanford CS 221/229/230](https://github.com/shervinea/cheatsheet-translation).

## Chất lượng bản dịch và góp ý

Phần tiếng Việt (bản dịch, giải thích, ví dụ) do LLM viết: Gemini 3.8 Flash dịch 776 thuật ngữ, Claude Sonnet 5.5 dịch 1.234 thuật ngữ còn lại. LLM vẫn có thể sai, nhất là với các thuật ngữ ít gặp.

Nếu thấy bản dịch sai hoặc thiếu một thuật ngữ quan trọng, bạn có thể:

- **mở [issue](https://github.com/thucnc7/aidictionary/issues)**, ghi rõ thuật ngữ, chỗ sai và cách dịch đề xuất;
- **gửi pull request**: thêm dòng sửa vào `data/overrides.jsonl`, hoặc thêm thuật ngữ vào `data/manual_terms.txt`.

## Giấy phép

- **Mã nguồn** (thư mục `aidict/`, `tests/`): [MIT](LICENSE).
- **Dữ liệu** (thư mục `data/`) **và file từ điển** `aidict.mobi`: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.vi), vì có dùng nội dung từ Wiktionary và Wikipedia. Khi chia sẻ lại hoặc làm từ điển mới dựa trên dữ liệu này, bạn cần ghi nguồn và giữ cùng giấy phép.

Nguồn dữ liệu và giấy phép gốc:

- Wiktionary tiếng Việt qua kaikki.org: CC BY-SA
- Wikipedia Glossary of AI: CC BY-SA
- Google ML Glossary: CC BY 4.0
- Hugging Face Transformers: Apache 2.0
- Cheatsheet Stanford (shervinea): MIT
- Bảng thuật ngữ D2L tiếng Việt: nhóm dịch Đắm mình vào Học Sâu
