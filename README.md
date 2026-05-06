# Meridian: ESL-Aware AI Text Detection

**Meridian** is a dual-stream perplexity architecture for equitable AI text detection that reduces false positive rates on ESL student writing by 15.7× compared to standard detectors, while maintaining 98.3% detection of AI-mimicked ESL text.

> Associated paper: *"Meridian: A Bivariate Perplexity Architecture for Trustworthy and Equitable AI Text Detection"* — submitted to the AI4GOOD Workshop @ ICML 2026.

---

## The Problem

Standard AI text detectors flag up to 72% of authentic ESL student essays as AI-generated. This is a population-scale fairness failure: the same perplexity axis that identifies AI text also penalizes learner writing patterns.

## How Meridian Works

Each essay is mapped to a point in a 2D space:
- **x-axis (P_esl):** How natural does this look to a character-level LSTM trained on ESL learner writing? (lower = more ESL-like)
- **y-axis (P_native):** How surprising does this look to DistilGPT-2? (higher = more idiosyncratic)

Human ESL writers occupy the upper-left; AI mimics cluster lower-left. A logistic classifier separates them.

## Results

| Method | ESL FPR | AI-ESL TPR | AUC |
|---|---|---|---|
| Fast-DetectGPT | 85.0% | 94.0% | 0.313 |
| Binoculars (ICML 2024) | 1.7% | 89.3% | — |
| **Meridian (ours)** | **8.3%** | **>99%** | **0.962** |
| **Meridian zero-shot TOEFL11** | **3.9%** | **98.3%** | — |

Low-proficiency writers: **0.0% FPR** (95% CI: [0.0%, 3.3%], n=110).

---

## Quickstart

```bash
pip install torch transformers numpy

# Score a single essay
python inference.py --text "I think that students would benefit from learn at home because..."

# Score from a file
python inference.py --file my_essay.txt

# Use a conservative threshold (lower FPR)
python inference.py --file essay.txt --threshold 0.80
```

**Output:**
```
Meridian Result
=============================================
  P_esl (ESL-Expert LSTM):     2.341
  P_native (DistilGPT-2):      87.4
  P(AI-generated):             12.3%
  Prediction:                  Human (ESL)
=============================================

Note: Meridian is a screening tool, not a verdict system.
All flagged essays should receive human review.
```

---

## Repository Contents

| File | Description |
|---|---|
| `LSTM_ELS.ipynb` | Full training and evaluation notebook |
| `inference.py` | Standalone inference script |
| `SUPER_esl_lstm_weights.pth` | Trained ESL-Expert LSTM weights |

---

## Responsible Use

Meridian is a **screening tool, not a verdict system**. No automated detection output should initiate disciplinary proceedings without review by a qualified educator familiar with the student's language background.

See the paper's Responsible Deployment Framework for full guidelines.

---

## Citation

```bibtex
@article{meridian2026,
  title   = {Meridian: A Bivariate Perplexity Architecture for
             Trustworthy and Equitable AI Text Detection},
  author  = {Anonymous},
  year    = {2026},
  note    = {Submitted to AI4GOOD Workshop @ ICML 2026}
}
```
