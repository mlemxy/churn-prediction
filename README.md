# churn-prediction

Two trimesters of work on customer churn prediction: a foundational classification model
(Y2T2, AAI2002), extended into a real-time, on-device deployment at the point-of-sale
(Y2T3, AAI2114). Each trimester's work is self-contained in its own folder below,
read the relevant section for the one you're looking at.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/174MK_kL2FJ2Lij4funMmOMgxIfi9EAk2)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## Repo Structure

```
churn-prediction/
├── Y2T2/                          # AAI2002 foundation (previous trimester)
│   ├── main.ipynb
│   ├── requirements.txt
│   └── experiment/
└── Y2T3/                          # AAI2114 edge deployment (current)
    ├── main.ipynb
    ├── requirements.txt
    ├── sidecar.py
    ├── docker-compose.yml
    ├── score.py
    ├── churn_model_edge.onnx
    └── ChurnBenchmark/
```

---

## Y2T2, AAI2002: Churn Prediction Foundation (previous trimester)

A machine learning pipeline built to predict customer churn in subscription-based
industries. Using XGBoost with SMOTE-ENN resampling and SHAP explainability, it identifies
at-risk customers and surfaces actionable, customer-level reasons for churn risk rather
than just a probability score. This is the foundation the Y2T3 project below extends
toward real-time edge deployment.

**Best result:** 81.28% recall, 62.94% F1, 0.8263 AUC (SMOTE-ENN + recall-tuned XGBoost,
threshold 0.50)

### Results

| Model | Recall | F1 | AUC |
|---|---|---|---|
| LR baseline (tau=0.41) | 64.35% | 61.50% | 0.8411 |
| XGBoost, no resampling | 52.05% | 55.67% | 0.8142 |
| SMOTE-ENN + XGBoost default | 68.98% | 60.99% | 0.8183 |
| SMOTE-ENN + XGBoost, recall-tuned | 81.28% | 62.94% | 0.8263 |

### Problem framing

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
per-customer explanations. A waterfall plot can tell a retention team exactly which
features pushed a specific customer's churn probability up or down, which is more
actionable than a global feature ranking.

### Key findings

Contract type is the strongest churn driver (mean absolute SHAP value 0.7143):
month-to-month customers are significantly more likely to churn than those on one or two
year contracts.

Tenure ranks second: customers who have recently joined are at substantially higher risk
than long-term subscribers.

Absence of online security and tech support consistently push churn probability up,
ranking third and fourth in global SHAP importance respectively.

Optimal classification threshold is 0.50. Feature engineering (charge_per_tenure,
service_count, has_any_addon) was evaluated and dropped after ablation confirmed a recall
regression versus the base pipeline.

### Run it

```bash
git clone https://github.com/mlemxy/churn-prediction
cd churn-prediction/Y2T2
pip install -r requirements.txt
```

Open `main.ipynb`. Developed on Google Colab free tier but runs locally as long as
dependencies are installed. Requires a Kaggle API token to download the dataset.

---

## Y2T3, AAI2114: Edge Deployment at the Point-of-Sale (current)

Edge-native customer churn prediction and real-time retention at the point-of-sale. An
XGBoost model trained on retail RFM features, exported to ONNX, and deployed to run
entirely on-device, no cloud connection needed at inference time.

**Problem:** a cloud-hosted churn model needs a round trip to a server before returning a
score, seconds to minutes. A checkout interaction is over in seconds. This project deploys
the model directly on point-of-sale hardware instead: fast and small enough to influence
the transaction in progress, with zero cloud dependency at inference time.

**Pipeline:** loyalty card scan, SQLite profile lookup, RFM feature computation (recency,
frequency, log-monetary), ONNX inference, rule engine, retention action (win-back coupon,
cross-sell prompt, or loyalty points), all on-device. An optional background step syncs
results to a dashboard when connectivity is available.

**Result:** 67.91% recall, 72.02% AUC, 60.08% F1 (test set, XGBoost + SMOTE-ENN,
small_20_depth2). Exported to ONNX at 4.9 KB, 1.0000 prediction match rate against the
original model. End-to-end on-device latency on a real ARM Android device: p50 = 0.188ms,
p99 = 1.843ms, roughly 271x inside a 500ms checkout budget even at the 99th percentile.

### Model comparison

| Model | Recall | AUC | F1 | Size |
|---|---|---|---|---|
| Logistic Regression baseline | 62.96% | 73.85% | 63.55% | not exported |
| XGBoost, no resampling | 56.48% | 70.92% | 57.96% | not exported |
| SMOTE-ENN + XGBoost, GridSearchCV tuned | 71.76% | 71.54% | 63.66% | 170.4 KB |
| SMOTE-ENN + XGBoost, small_20_depth2 (deployed) | 76.39% | 72.94% | 66.40% | 28.4 KB |

Test set (evaluated once, on the selected model): 67.91% recall, 72.02% AUC, 60.08% F1,
53.87% precision.

### On-device latency (Poco X3 Pro, Snapdragon 855, 200 runs, 100 real customer profiles)

| Stage | p50 | p99 |
|---|---|---|
| SQLite lookup | 0.0823ms | 1.5632ms |
| RFM computation | 0.0044ms | 0.0106ms |
| ONNX inference | 0.1008ms | 1.0180ms |
| End-to-end | 0.1880ms | 1.8431ms |

### Run it

```bash
git clone https://github.com/mlemxy/churn-prediction
cd churn-prediction/Y2T3
pip install -r requirements.txt
```

Open `main.ipynb` (developed on Google Colab, x86_64, Python 3.12.13, runs locally with
dependencies installed). For the full pipeline demo, see `docker-compose.yml` to bring up
OSPOS, then run `sidecar.py`. For ARM latency benchmarking, build and install
`ChurnBenchmark/` on a physical Android device.
