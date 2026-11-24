
# Phát hiện văn bản sinh bởi AI / Đồng sáng tác (MGT vs. HWT)

Dự án thực nghiệm phân loại văn bản gồm hai giai đoạn:
1. **Giai đoạn 1 (DAIGT V2):** Huấn luyện và so sánh các mô hình học máy truyền thống (TF-IDF) và học sâu (Transformer) để đánh giá hiệu năng và thời gian huấn luyện.
2. **Giai đoạn 2 (MixSet):** Xây dựng pipeline phân loại ở cấp độ câu kết hợp **ELECTRA** (`google/electra-small-discriminator`), **Majority Voting**, **Mean Pooling** và **SVM** (RBF kernel + cân bằng lớp + tinh chỉnh ngưỡng quyết định)[cite: 1].

---

## Cấu trúc tài liệu & mã nguồn

### 1. Báo cáo / Tiểu luận (Word)
- `<ten_file_bao_cao>.docx`: Báo cáo/tiểu luận hoàn chỉnh trình bày cơ sở phương pháp, công thức, thực nghiệm và đánh giá kết quả.

### 2. Bộ dữ liệu (2 Dataset)
- `<ten_dataset_1>`: Bộ dữ liệu **DAIGT V2** (dùng cho bước đối sánh mô hình ban đầu).
- `<ten_dataset_2>`: Bộ dữ liệu **MixSet** (ví dụ `Mixset_train.json` / `Mixset_test.json` hoặc file CSV/JSON tương ứng, dùng cho pipeline cấp độ câu)[cite: 1].

### 3. Các file mã nguồn (5 File code)
- `<file_code_1>`: Tải / chuẩn hóa dữ liệu và tiền xử lý.
- `<file_code_2>`: Huấn luyện và đánh giá các mô hình truyền thống trên DAIGT V2.
- `<file_code_3>`: Huấn luyện và đánh giá các mô hình Deep Learning / Transformer trên DAIGT V2.
- `<file_code_4>`: Vẽ biểu đồ / tổng hợp kết quả so sánh các mô hình.
- `MixSet__17_11.ipynb`: Pipeline xử lý cấp câu trên MixSet (chia dữ liệu cấp tài liệu, undersampling, trích đặc trưng/voting với ELECTRA, huấn luyện SVM và tối ưu ngưỡng)[cite: 1].

---

## Thứ tự chạy đề xuất

1. **Chuẩn bị dữ liệu & môi trường:** Cài đặt các thư viện cần thiết (`torch`, `transformers`, `datasets`, `scikit-learn`, `pandas`, `numpy`, `nltk`, `matplotlib`, `kagglehub`) và chuẩn bị 2 bộ dữ liệu[cite: 1].
2. **Chạy thực nghiệm bước 1 (DAIGT V2):** Chạy các file code huấn luyện mô hình trên DAIGT V2, sau đó chạy file vẽ biểu đồ so sánh độ chính xác và thời gian huấn luyện.
3. **Chạy pipeline bước 2 (MixSet):** Chạy `MixSet__17_11.ipynb` (trên GPU như Colab T4 sẽ nhanh hơn) để đánh giá phân loại cấp độ câu[cite: 1].
4. **Đối chiếu báo cáo:** Đối chiếu kết quả đầu ra với số liệu trong file `.docx`.

---

## Lưu ý quan trọng
- **Quy ước nhãn:** `HWT` (Human-Written Text) → `0`; `MGT` (Machine-Generated / Mixed Text) → `1`[cite: 1, 2].
- **Đặc thù MixSet:** Một số mẫu `HWT` trong MixSet thuộc nhóm biến thể chỉnh sửa tự nhiên (`gpt4_humanize`) có liên quan đến mô hình nguồn (như Dolly, GPT-4), nên ranh giới phân loại phức tạp hơn dữ liệu thuần túy[cite: 2].
