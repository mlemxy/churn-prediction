# churn-prediction
Developed as part of a project under Team ITP_14, churn-prediction is a machine learning pipeline built to predict customer churn in subscription-based industries. Using XGBoost with SMOTE-ENN resampling and SHAP explainability, it identifies at-risk customers and surfaces actionable, customer-level reasons for churn risk rather than just a score.

**Best result:** 81.28% recall, 62.94% F1, 0.8263 AUC (SMOTE-ENN + Recall-tuned XGBoost, threshold 0.50)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlemxy/churn-prediction/blob/main/main.ipynb)

## Technologies
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## Results
| Model | Recall | F1 | AUC |
|-------|--------|----|-----|
| LR baseline (τ=0.41) | 67.56% | 63.27% | 0.8377 |
| XGBoost, no resampling | 52.05% | 55.30% | 0.8120 |
| XGBoost + SMOTE (exploratory) | 61.14% | 58.73% | 0.8137 |
| SMOTE-ENN + XGBoost default | 69.70% | 60.67% | 0.8200 |
| **SMOTE-ENN + XGBoost, recall-tuned** | **81.28%** | **62.94%** | **0.8263** |

## Process
Churn prediction is a class imbalance problem. Only 26.58% of customers in the IBM Telco dataset churned, so a model can hit decent accuracy just by predicting "retained" for everyone. Recall matters more here since missing a churner costs far more in lost revenue than accidentally contacting someone who was going to stay anyway.

An untuned XGBoost baseline only hits 52.05% recall, so nearly half of all churners go undetected. Eight resampling techniques were compared under identical model settings. SMOTE-ENN produced the highest recall at 69.70%, 16.58pp above ADASYN, and also helped by cleaning noisy boundary samples which reduced the training set compared to pure oversampling methods. From there, dual GridSearchCV runs (`scoring='f1'` and `scoring='recall'`) were done, with the recall-tuned model (lr=0.01, depth=3, n=100) selected as the final model.

SHAP was picked over XGBoost built-in importance and LIME since it gives directional, per-customer explanations. A waterfall plot can tell a retention team exactly which features pushed a specific customer's churn probability up or down, which is a lot more useful than a global feature ranking.

> ⚠️ **a/n:** Week 6 XGBoost+SMOTE standalone hit a higher recall figure but SMOTE was applied before the train/test split, so there was data leakage. All technical claims use `main.ipynb` figures only. The XGBoost+SMOTE row in the results table above is an exploratory result (three variables changed simultaneously vs the baseline) and is not part of the controlled pipeline progression.

## Key Findings
- **Contract type** is far and away the strongest churn driver (mean |SHAP| 0.7143): month-to-month customers are much more likely to churn than those on one or two year contracts
- **Tenure** comes in second: customers who have barely joined are significantly more at risk than long-term ones
- **No online security and no tech support** consistently push churn probability up, both globally and in individual customer explanations (ranks 3 and 4 in global SHAP importance)
- Optimal classification threshold is **0.50**: no adjustment from default was needed for this pipeline
- Feature engineering (charge_per_tenure, service_count, has_any_addon) actually hurt recall by 12.12pp and was dropped

## Limitations
Precision sits at 51.35%, so roughly every other customer flagged as at-risk did not actually churn. Without a real CLV estimate and campaign cost to build a profit curve against, it is hard to say exactly where an acceptable precision floor would be. Also, the model CV'd at 96.27% recall on resampled training folds but only hit 81.28% on the natural test set. That gap is expected since the resampled folds are near-balanced while the real test set sits at 1:2.76, but it is easy to be misled by the CV number alone.

## Challenges
- **KMeans-SMOTE** failed outright with "No clusters found". The minority class in this dataset is too diffuse for centroid initialisation to work, which was a good reminder that an algorithm working in theory does not guarantee it will run on a given dataset
- **ADASYN reversal**: ADASYN was selected at Week 8 based on a standalone notebook, then overturned in Week 9 when the same comparison ran under a standardised pipeline and SMOTE-ENN came out 16.58pp ahead on recall. Lesson learned on controlling for pipeline configuration before drawing conclusions

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

## How to Run
```bash
git clone https://github.com/mlemxy/churn-prediction
pip install -r requirements.txt
```
Then open `main.ipynb`. Developed on Google Colab free tier, but runs locally as long as the dependencies are installed.

## Acknowledgements
Claude was used in a supporting capacity: research, code debugging, error diagnosis, and formatting only. All analytical reasoning, including problem framing, pipeline architecture, technique selection, hyperparameter tuning, and interpretation of results, was conducted independently by me with reference to the cited literature and Kaggle notebooks.
