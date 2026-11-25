# Detecting Human–AI Co-Authored Text on MixSet with a Hybrid ELECTRA–SVM Pipeline

Zip file represents for results in docs file and the below.

## 1. Introduction

With the rapid growth of large language models (GPT-4, Claude, Llama, etc.), AI-generated text detection has expanded to **detecting human–AI co-authored text (mixtext)**, where the boundary between human-written and machine-written text is increasingly blurred.

The project consists of **two connected stages**:

1. **Baseline study on DAIGT V2** – comparing traditional machine learning models (Random Forest, SVM, XGBoost, LightGBM) and deep learning models (RoBERTa, ELECTRA, DeBERTa-v3-small) on pure AI-generated text detection.
2. **Hybrid pipeline on MixSet** – building and optimizing an **ELECTRA + Majority Voting + SVM** pipeline to distinguish human-written text (HWT) from machine-generated/mixed text (MGT), with a focus on **class imbalance** (MGT ≈ 2,500 vs. HWT ≈ 500).

## 2. Repository Structure

```
.
├── DAIGT_V2_2710_1.ipynb                    # DAIGT V2 study – experiment 1
├── DAIGT_V2_2710_2.ipynb                    # DAIGT V2 study – experiment 2
├── MixSet__03_11.ipynb                      # MixSet – initial pipeline
├── MixSet__17_11.ipynb                      # MixSet – optimized pipeline
├── create_comparision_image_among_models.*  # Model comparison plots (Figure 4.1)
└── README.md
```



### File descriptions

| File | Purpose |
|---|---|
| `DAIGT_V2_2710_1` | Downloads DAIGT V2 via `kagglehub`, filters `RDizzl3_seven == True`, stratified 80/20 split. Trains **Random Forest, LightGBM** (TF-IDF n-grams) and **DeBERTa-v3-small** → *Table A*. |
| `DAIGT_V2_2710_2` | Trains **SVM (LinearSVC), Random Forest, XGBoost** (TF-IDF 15k) and fine-tunes **RoBERTa-base, ELECTRA** → *Table B*. |
| `MixSet__03_11` | **Initial version** on MixSet: sentence splitting → ELECTRA (majority voting + mean-pooling) → SVM. No undersampling, default threshold 0.5. |
| `MixSet__17_11` | **Optimized version**: document-level split, undersampling on the training set, RBF SVM + `class_weight='balanced'`, decision threshold tuning on the Precision–Recall curve. |
| `create_comparision_image_among_models` | Aggregates results and plots accuracy / training time comparisons across models on DAIGT V2. |

## 3. Data

| Dataset | Source | Notes |
|---|---|---|
| **DAIGT V2** | Kaggle: `thedrcat/daigt-v2-train-dataset` (`train_v2_drcat_02.csv`) | Filtered by `RDizzl3_seven == True` → ≈ 20,450 samples. Labels: 0 = Human, 1 = AI. |
| **MixSet** | Hugging Face: `ONE-Lab/MixSet` | ≈ 3,000 training samples (MGT ≈ 2,500, HWT ≈ 500). Uses the `revised` column. Labels: HWT → 0, MGT → 1. |

## 4. Methodology

### Stage 1 – DAIGT V2
- **Traditional ML models:** word-level TF-IDF (`max_features=15000`, `stop_words='english'`) → Random Forest / LinearSVC / XGBoost / LightGBM.
- **Deep learning models:** fine-tuning RoBERTa-base, ELECTRA and DeBERTa-v3-small with token embeddings.

### Stage 2 – MixSet (hybrid pipeline)

```
Text ─► Document-level train/test split ─► Sentence splitting (NLTK sent_tokenize)
     ─► ELECTRA-small-discriminator ─┬─► Token logits ─► Majority Voting ─► Sentence label
                                     └─► Mean-pooling of hidden states ─► Sentence embedding
     ─► SVM (RBF, class_weight='balanced', probability=True)
     ─► Threshold tuning (θ* ≈ 0.7959) ─► Final label
```

Key techniques:
- **Data leakage prevention:** stratified split at the document level *before* sentence splitting.
- **Undersampling** on the training set only: keep all HWT samples, reduce MGT to a ≈ 1.5 : 1 ratio (≈ 400 HWT + 600 MGT).
- **Class weighting** with `class_weight='balanced'` for the SVM.
- **Threshold tuning** on the Precision–Recall curve of the Human class.

## 5. Results

### 5.1. DAIGT V2

| Model | Type | Accuracy | Training time |
|---|---|---|---|
| SVM (LinearSVC) | Traditional | 99.51% | 0.92 s |
| **ELECTRA** | Deep Learning | **99.53%** | 16.96 min |
| RoBERTa-base | Deep Learning | 98.93% | 18.39 min |
| Random Forest | Traditional | 98.66% | 50.33 s |
| XGBoost | Traditional | 97.19% | 2.29 min |
| LightGBM | Traditional | 99.39% | 4.51 min |
| DeBERTa-v3-small | Deep Learning | 98.68% | – |

### 5.2. MixSet – before and after optimization

| Metric | Initial (`MixSet__03_11`) | Optimized (`MixSet__17_11`) |
|---|---|---|
| Accuracy | 94.95% | 93.70% |
| Precision (Human) | 0.50 | **0.90** |
| Recall (Human) | – | 0.85 |
| F1 (Human) | 0.64 | **0.87** |
| Macro F1 | 0.81 | **0.91** |

**Takeaway:** Accuracy drops slightly, but the balance between the two classes improves significantly. On imbalanced data, accuracy alone is not enough.

## 6. Usage

### Environment
- Experiments were run on **Google Colab** (T4 GPU for Transformers, CPU for traditional models).
- Python 3.9+

### Install dependencies

```bash
pip install kagglehub scikit-learn xgboost lightgbm transformers datasets torch nltk pandas matplotlib
```

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

### Running order

1. Run `DAIGT_V2_2710_1` and `DAIGT_V2_2710_2` for the DAIGT V2 study *(the Kaggle API must be configured if downloading with `kagglehub`)*.
2. Run `create_comparision_image_among_models` to generate the model comparison plots.
3. Run `MixSet__03_11` for the initial pipeline results.
4. Run `MixSet__17_11` for the optimized pipeline results.

> The notebooks automatically download data from Kaggle / Hugging Face when run.

## 7. Key Hyperparameters

| Component | Value |
|---|---|
| TF-IDF (DAIGT) | `max_features=15000`, `stop_words='english'` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| ELECTRA | `google/electra-small-discriminator`, `max_length=128` |
| SVM (MixSet) | `kernel='rbf'`, `C=1.0`, `class_weight='balanced'`, `probability=True` |
| Undersampling | MGT/HWT ratio ≈ 1.5 on the training set |
| Optimal threshold (Human class) | ≈ 0.7959 |

## 8. Limitations and Future Work

**Limitations**
- ELECTRA is not deeply fine-tuned on MixSet (mainly used for embedding extraction and majority voting).
- Evaluation is mostly at the sentence level; aggregation to the full-document level is not fully explored.
- Undersampling discards part of the majority-class information.
- Experiments use English only, and MixSet is relatively small.
- No direct comparison with DetectGPT, Binoculars, or full fine-tuning.

**Future work**
- Fully fine-tune ELECTRA / DeBERTa-v3 / RoBERTa-large on MixSet.
- Aggregate predictions from sentence level to document level.
- Try SMOTE in embedding space, focal loss, and generative augmentation.
- Extend to Vietnamese (with word segmentation) and build a Vietnamese co-authored text dataset.
- Integrate into an academic integrity tool with an adjustable decision threshold.

## 9. References

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

## 10. Contact

- Author: Le Thi Phung
- Repository: https://github.com/Phung965275/Human-MGT-coauthoredtext-Detection
