# 🧠 Benchmarking Classical and Transformer-Based NLP Models for Stress Detection in Mental Health Text

**CSCI-4364/6364 S26 — Machine Learning**  
George Washington University  

---

## 👥 Team Members

- Prathi Patil Saralebettu — G21906887  
- Quoc Nguyen — G30672640  
- Sinchana Manjula Prabhakara — G26662228  
- Sosna Achamyeleh — G44048835  

---

## 📌 Abstract

This project benchmarks classical NLP models (Logistic Regression and Naive Bayes with TF-IDF) against a fine-tuned DistilBERT transformer for binary stress detection in mental health social media text.

Using the **Dreaddit dataset (3,549 Reddit posts)**, all models were trained and evaluated under identical conditions across six random seeds.

### Key Results:
- DistilBERT: **Macro F1 = 0.793 ± 0.015**
- Logistic Regression: **0.756 ± 0.019**
- Naive Bayes: **0.748 ± 0.017**

DistilBERT significantly outperformed classical baselines, confirmed via McNemar’s test, Wilcoxon signed-rank test, and Nadeau-Bengio corrected t-test.

A key finding is that transformer models improve performance but introduce **overconfident error patterns**, which is critical in mental health applications.

---

## 🎯 Research Question

Does fine-tuning DistilBERT provide a statistically significant improvement over classical TF-IDF-based models for binary stress detection on a small (~3.5K samples) mental health dataset?

---

## 🧪 Models

### Classical Models
- Logistic Regression (TF-IDF unigrams + bigrams)
- Multinomial Naive Bayes (TF-IDF)

### Transformer Model
- DistilBERT (`distilbert-base-uncased`)
- Fine-tuned using Hugging Face Trainer

---

## 📊 Dataset

- **Dreaddit Dataset** (Turcan & McKeown, 2019)
- 3,549 Reddit posts
- Binary labels: stressed / not stressed
- Subreddits include anxiety, PTSD, stress, domestic violence, homelessness

---

## ⚙️ Methodology

### Preprocessing
- Lowercasing text
- Removing URLs, mentions, hashtags, punctuation
- Filtering very short posts
- Stratified splits (70/15/15)
- 6 random seeds for robustness

### Classical Pipeline
- TF-IDF (10,000 max features)
- Logistic Regression with grid search (C)
- Naive Bayes with grid search (alpha)

### Transformer Pipeline
- DistilBERT fine-tuning
- Batch size: 16
- Max epochs: 5
- Learning rate sweep: {2e-5, 3e-5, 5e-5}
- Early stopping on validation F1

---

## 📈 Results

| Model | Macro F1 | ROC-AUC |
|------|----------|--------|
| DistilBERT | **0.793 ± 0.015** | **0.880** |
| Logistic Regression | 0.756 ± 0.019 | 0.841 |
| Naive Bayes | 0.748 ± 0.017 | 0.832 |

### Key Takeaways
- DistilBERT consistently outperformed classical models across all seeds
- +0.038 average F1 improvement over Logistic Regression
- Statistically significant improvements (p < 0.05)
- Transformer gains are consistent but not free (compute cost ↑)

---

## 🔍 Error Analysis

- DistilBERT errors: 46% high-confidence misclassifications
- Logistic Regression errors: mostly low-confidence boundary cases
- Main error source across all models: ambiguous context (~84%)

### Common failure cases:
- indirect emotional expressions
- sarcasm/humor
- implicit stress signals

---

## 🧠 Key Insights

- Transformers improve performance but change error behavior
- Classical models remain strong baselines for small datasets
- Confidence calibration is critical for mental health applications
- Statistical testing is essential for small-sample ML comparisons

---

## 📁 Repository Structure
├── notebooks/

│ └── stress_detection_benchmark.ipynb

├── results/

│ ├── figures/

└── README.md


---

## 🚀 How to Run

### 1. Clone repository
```bash
git clone https://github.com/sossyh/classical-vs-transformer-stress-nlp.git
cd classical-vs-transformer-stress-nlp
```

### 2. Run notebook

Open:

notebooks/stress_detection_benchmark.ipynb

Or run in Google Colab.