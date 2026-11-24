# Nhận diện văn bản đồng sáng tác Người–AI trên MixSet với pipeline lai ELECTRA–SVM

> Tiểu luận học phần **Xử lý ngôn ngữ tự nhiên** – Ngành Khoa học dữ liệu, Trường Đại học Quy Nhơn (2026)

| | |
|---|---|
| **Sinh viên thực hiện** | Lê Thị Phụng – MSSV 4654110019 – Lớp Khoa học dữ liệu K46 |
| **Giảng viên hướng dẫn** | TS. Lê Văn Hùng |

---

## 1. Giới thiệu

Cùng với sự phát triển của các mô hình ngôn ngữ lớn (GPT-4, Claude, Llama…), bài toán phát hiện văn bản do AI tạo ra đã mở rộng sang **nhận diện văn bản đồng sáng tác Người–AI (mixtext)**, nơi ranh giới giữa văn bản người viết và máy viết trở nên mờ nhạt.

Dự án gồm **hai giai đoạn gắn kết**:

1. **Khảo sát nền tảng trên DAIGT V2** – so sánh các mô hình học máy truyền thống (Random Forest, SVM, XGBoost, LightGBM) và học sâu (RoBERTa, ELECTRA, DeBERTa-v3-small) trong bài toán phát hiện văn bản AI thuần túy.
2. **Pipeline lai trên MixSet** – xây dựng và tối ưu pipeline **ELECTRA + Majority Voting + SVM** để phân biệt văn bản con người (HWT) với văn bản có yếu tố đồng sáng tác/trộn (MGT), tập trung xử lý **mất cân bằng lớp** (MGT ≈ 2500 so với HWT ≈ 500).

## 2. Cấu trúc thư mục

```
.
├── DAIGT_V2_2710_1.ipynb                    # Khảo sát DAIGT V2 – thực nghiệm 1
├── DAIGT_V2_2710_2.ipynb                    # Khảo sát DAIGT V2 – thực nghiệm 2
├── MixSet__03_11.ipynb                      # MixSet – pipeline ban đầu
├── MixSet__17_11.ipynb                      # MixSet – pipeline tối ưu
├── create_comparision_image_among_models.*  # Vẽ biểu đồ so sánh các mô hình (Hình 4.1)
└── README.md
```



### Mô tả từng file

| File | Chức năng |
|---|---|
| `DAIGT_V2_2710_1` | Tải DAIGT V2 bằng `kagglehub`, lọc `RDizzl3_seven == True`, chia stratified 80/20. Huấn luyện **Random Forest, LightGBM** (TF-IDF n-grams) và **DeBERTa-v3-small** → *Bảng A*. |
| `DAIGT_V2_2710_2` | Huấn luyện **SVM (LinearSVC), Random Forest, XGBoost** (TF-IDF 15k) và fine-tune **RoBERTa-base, ELECTRA** → *Bảng B*. |
| `MixSet__03_11` | **Phiên bản ban đầu** trên MixSet: tách câu → ELECTRA (majority voting + mean-pooling) → SVM. Chưa dùng undersampling, ngưỡng mặc định 0.5. |
| `MixSet__17_11` | **Phiên bản tối ưu**: chia tách ở cấp tài liệu, undersampling trên tập train, SVM RBF + `class_weight='balanced'`, tinh chỉnh ngưỡng quyết định trên đường Precision–Recall. |
| `create_comparision_image_among_models` | Tổng hợp kết quả và tạo hình so sánh độ chính xác / thời gian huấn luyện giữa các mô hình trên DAIGT V2. |

## 3. Dữ liệu

| Tập dữ liệu | Nguồn | Ghi chú |
|---|---|---|
| **DAIGT V2** | Kaggle: `thedrcat/daigt-v2-train-dataset` (file `train_v2_drcat_02.csv`) | Lọc `RDizzl3_seven == True` → ≈ 20.450 mẫu. Nhãn: 0 = Human, 1 = AI. |
| **MixSet** | Hugging Face: `ONE-Lab/MixSet` | ≈ 3000 mẫu train (MGT ≈ 2500, HWT ≈ 500). Dùng cột `revised`. Nhãn: HWT → 0, MGT → 1. |

## 4. Phương pháp

### Giai đoạn 1 – DAIGT V2
- **Mô hình học máy truyền thống:** TF-IDF mức từ (`max_features=15000`, `stop_words='english'`) → Random Forest / LinearSVC / XGBoost / LightGBM.
- **Mô hình học sâu:** fine-tune RoBERTa-base, ELECTRA, DeBERTa-v3-small với token embeddings.

### Giai đoạn 2 – MixSet (pipeline lai)

```
Văn bản ─► Chia train/test cấp tài liệu ─► Tách câu (NLTK sent_tokenize)
        ─► ELECTRA-small-discriminator ─┬─► Logits token ─► Majority Voting ─► Nhãn câu
                                        └─► Mean-pooling hidden states ─► Embedding câu
        ─► SVM (RBF, class_weight='balanced', probability=True)
        ─► Threshold tuning (θ* ≈ 0.7959) ─► Nhãn cuối cùng
```

Các kỹ thuật chính:
- **Chống rò rỉ dữ liệu:** chia stratified ở cấp tài liệu *trước khi* tách câu.
- **Undersampling** chỉ trên tập train: giữ toàn bộ HWT, giảm MGT xuống tỷ lệ ≈ 1.5 : 1 (≈ 400 HWT + 600 MGT).
- **Trọng số lớp** `class_weight='balanced'` cho SVM.
- **Tinh chỉnh ngưỡng** trên đường Precision–Recall của lớp Human.

## 5. Kết quả

### 5.1. DAIGT V2

| Mô hình | Loại | Accuracy | Thời gian train |
|---|---|---|---|
| SVM (LinearSVC) | Truyền thống | 99.51% | 0.92 giây |
| **ELECTRA** | Deep Learning | **99.53%** | 16.96 phút |
| RoBERTa-base | Deep Learning | 98.93% | 18.39 phút |
| Random Forest | Truyền thống | 98.66% | 50.33 giây |
| XGBoost | Truyền thống | 97.19% | 2.29 phút |
| LightGBM | Truyền thống | 99.39% | 4.51 phút |
| DeBERTa-v3-small | Deep Learning | 98.68% | – |

### 5.2. MixSet – trước và sau tối ưu

| Chỉ số | Ban đầu (`MixSet__03_11`) | Tối ưu (`MixSet__17_11`) |
|---|---|---|
| Accuracy | 94.95% | 93.70% |
| Precision (Human) | 0.50 | **0.90** |
| Recall (Human) | – | 0.85 |
| F1 (Human) | 0.64 | **0.87** |
| Macro F1 | 0.81 | **0.91** |

**Nhận xét:** Accuracy giảm nhẹ nhưng độ cân bằng giữa hai lớp được cải thiện rõ rệt. Trên dữ liệu mất cân bằng, chỉ nhìn vào Accuracy là chưa đủ.

## 6. Hướng dẫn chạy

### Môi trường
- Thực nghiệm được thực hiện trên **Google Colab** (GPU T4 cho Transformer, CPU cho các mô hình truyền thống).
- Python 3.9+

### Cài đặt thư viện

```bash
pip install kagglehub scikit-learn xgboost lightgbm transformers datasets torch nltk pandas matplotlib
```

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

### Thứ tự chạy

1. Chạy `DAIGT_V2_2710_1` và `DAIGT_V2_2710_2` để lấy kết quả khảo sát trên DAIGT V2 *(cần cấu hình Kaggle API nếu tải dữ liệu bằng `kagglehub`)*.
2. Chạy `create_comparision_image_among_models` để tạo hình so sánh các mô hình.
3. Chạy `MixSet__03_11` để lấy kết quả của pipeline ban đầu.
4. Chạy `MixSet__17_11` để lấy kết quả của pipeline đã tối ưu.

> Các notebook tự động tải dữ liệu từ Kaggle / Hugging Face khi chạy.

## 7. Siêu tham số chính

| Thành phần | Giá trị |
|---|---|
| TF-IDF (DAIGT) | `max_features=15000`, `stop_words='english'` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| ELECTRA | `google/electra-small-discriminator`, `max_length=128` |
| SVM (MixSet) | `kernel='rbf'`, `C=1.0`, `class_weight='balanced'`, `probability=True` |
| Undersampling | Tỷ lệ MGT/HWT ≈ 1.5 trên tập train |
| Ngưỡng tối ưu (lớp Human) | ≈ 0.7959 |

## 8. Hạn chế và hướng phát triển

**Hạn chế**
- ELECTRA chưa được fine-tune chuyên sâu trên MixSet (chủ yếu dùng để trích xuất embedding và majority voting).
- Đánh giá chủ yếu ở mức câu, chưa khảo sát tổng hợp lên mức toàn văn bản.
- Undersampling làm mất một phần thông tin của lớp đa số.
- Chỉ thực nghiệm trên tiếng Anh; MixSet có quy mô còn nhỏ.
- Chưa so sánh trực tiếp với DetectGPT, Binoculars hay full fine-tuning.

**Hướng phát triển**
- Fine-tune hoàn toàn ELECTRA / DeBERTa-v3 / RoBERTa-large trên MixSet.
- Tổng hợp dự đoán từ mức câu lên mức văn bản (document-level aggregation).
- Thử SMOTE trên không gian embedding, focal loss, generative augmentation.
- Mở rộng sang tiếng Việt (có bước tách từ) và xây dựng bộ dữ liệu đồng sáng tác tiếng Việt.
- Tích hợp vào công cụ kiểm tra liêm chính học thuật cho phép tùy chỉnh ngưỡng quyết định.

## 9. Tài liệu tham khảo

1. Ahmad, A., et al. (2026). *A Comprehensive Survey on Detecting AI-Generated Text: Challenges, Methods, and Future Directions.*
2. Su, J., et al. (2025). *Beyond Pure Generation: Detecting Human–AI Co-Authored Text in the Era of Interactive Writing Tools.*
3. The DAIGT Team (2024). *DAIGT V2 Train Dataset.* Kaggle. https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
4. ONE-Lab (2024). *MixSet.* Hugging Face Datasets. https://huggingface.co/datasets/ONE-Lab/MixSet
5. Clark, K., et al. (2020). *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.*
6. Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
7. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
8. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.*
9. Mitchell, E., et al. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature.*
10. Hans, A., et al. (2024). *Binoculars: Zero-Shot Detection of LLM-Generated Text.*

## 10. Liên hệ

- Tác giả: Lê Thị Phụng
- Repository: https://github.com/Phung965275/Human-MGT-coauthoredtext-Detection
