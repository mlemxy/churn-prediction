# churn-prediction

A machine learning pipeline built to predict customer churn in subscription-based
industries. Using XGBoost with SMOTE-ENN resampling and SHAP explainability, it identifies
at-risk customers and surfaces actionable, customer-level reasons for churn risk rather
than just a probability score.

Submitted for AAI2002. This project is being extended as part of an ongoing industry
application project toward real-time edge deployment on point-of-sale terminals.

**Best result:** 81.28% recall, 62.94% F1, 0.8263 AUC (SMOTE-ENN + recall-tuned XGBoost,
threshold 0.50)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlemxy/churn-prediction/blob/main/main.ipynb)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## Results

| Model | Recall | F1 | AUC |
|---|---|---|---|
| LR baseline (tau=0.41) | 64.35% | 61.50% | 0.8411 |
| XGBoost, no resampling | 52.05% | 55.67% | 0.8142 |
| SMOTE-ENN + XGBoost default | 68.98% | 60.99% | 0.8183 |
| **SMOTE-ENN + XGBoost, recall-tuned** | **81.28%** | **62.94%** | **0.8263** |
| FE pipeline, recall-tuned (rejected) | 72.91% | 61.92% | 0.8256 |

---

## Problem Framing

Churn prediction is a class imbalance problem. Only 26.58% of customers in the IBM Telco
dataset churned (1:2.76 class ratio), so a naive classifier can hit decent accuracy by
predicting "retained" for everyone. Recall is the operationally correct primary metric:
missing a churner forfeits that customer's remaining lifetime value and triggers
reacquisition costs reported as five to twenty-five times the cost of retention.

An untuned XGBoost baseline only hits 52.05% recall, leaving nearly half of all churners
undetected. Eight resampling techniques were compared under identical model and threshold
conditions. SMOTE-ENN produced the highest recall and also cleaned noisy boundary samples,
reducing training set size compared to pure oversampling methods. Dual GridSearchCV runs
(scoring='f1' and scoring='recall') were conducted; the recall-tuned model (lr=0.01,
depth=3, n=100) was selected as the final model.

SHAP was selected over XGBoost built-in importance and LIME for its directional,
per-customer explanations. A waterfall plot can tell a retention team exactly which features
pushed a specific customer's churn probability up or down, which is more actionable than a
global feature ranking.

> Note: An exploratory XGBoost+SMOTE result was produced with SMOTE applied before the
> train/test split, introducing data leakage. All technical claims in this README use
> main.ipynb figures only.

---

## Key Findings

**Contract type** is the strongest churn driver (mean |SHAP| 0.7143): month-to-month
customers are significantly more likely to churn than those on one or two year contracts.

**Tenure** ranks second: customers who have recently joined are at substantially higher
risk than long-term subscribers.

**Absence of online security and tech support** consistently push churn probability up,
ranking third and fourth in global SHAP importance respectively.

Optimal classification threshold is 0.50. Feature engineering (charge_per_tenure,
service_count, has_any_addon) produced an 8.37pp recall regression versus the base pipeline
and was rejected.

---

## Limitations

Precision sits at 51.35%, so roughly every other customer flagged as at-risk did not
actually churn. Without a CLV estimate and campaign cost to build a profit curve against,
it is hard to define an acceptable precision floor.

---

## Repo Structure

```
churn-prediction/
├── main.ipynb
├── requirements.txt
└── experiment/
    ├── imbalance_technique_comparison.ipynb
    ├── logistic_regression.ipynb
    └── xgboost_smote.ipynb
```

---

## How to Run

```bash
git clone https://github.com/mlemxy/churn-prediction
pip install -r requirements.txt
```

Open `main.ipynb`. Developed on Google Colab free tier but runs locally as long as
dependencies are installed. Requires a Kaggle API token to download the dataset.

---

## Acknowledgements

Claude was used in a supporting capacity: research, code debugging, error diagnosis, and
formatting only. All analytical reasoning, including problem framing, pipeline architecture,
technique selection, hyperparameter tuning, and interpretation of results, was conducted
independently with reference to cited literature and Kaggle notebooks.
