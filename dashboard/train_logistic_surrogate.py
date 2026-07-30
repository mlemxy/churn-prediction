"""
Train a 6-feature logistic-regression surrogate of the full XGBoost churn model.

The 6 features are the SHAP top-6 from main.ipynb:
    Contract_Month-to-month, tenure, OnlineSecurity_No,
    TechSupport_No, InternetService_Fiber optic, MonthlyCharges

Features are trained on RAW (unscaled) values so the resulting coefficients
plug directly into a Power BI DAX measure in customer units (months, dollars, 0/1).

Outputs:
    - prints AUC, accuracy, intercept and coefficients
    - model_params.json  (intercept + coefficients, for the DAX measure)
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# 1. Load raw data and apply the same TotalCharges fix used elsewhere
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(), errors="coerce"
).fillna(0)

# 2. Build the 6 SHAP features (raw values, binary flags as 0/1)
X = pd.DataFrame({
    "tenure":                df["tenure"].astype(float),
    "MonthlyCharges":        df["MonthlyCharges"].astype(float),
    "Contract_MtM":          (df["Contract"] == "Month-to-month").astype(int),
    "OnlineSecurity_No":     (df["OnlineSecurity"] == "No").astype(int),
    "TechSupport_No":        (df["TechSupport"] == "No").astype(int),
    "InternetService_Fiber": (df["InternetService"] == "Fiber optic").astype(int),
})
y = (df["Churn"] == "Yes").astype(int)

# 3. Train/test split (same seed as the notebook for comparability)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Fit logistic regression
#    class_weight="balanced" mirrors the recall-oriented full pipeline
clf = LogisticRegression(max_iter=5000, class_weight="balanced")
clf.fit(X_train, y_train)

# 5. Evaluate
proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
acc = accuracy_score(y_test, (proba >= 0.5).astype(int))

print(f"AUC (6-feature LR surrogate): {auc:.4f}")
print(f"Accuracy @0.5 threshold     : {acc:.4f}")
print(f"Intercept (b0)              : {clf.intercept_[0]:+.6f}")
print("Coefficients:")
for name, c in zip(X.columns, clf.coef_[0]):
    print(f"  {name:<24s} {c:+.6f}")

# 6. Save params for the DAX measure
params = {
    "intercept": float(clf.intercept_[0]),
    "coef": {name: float(c) for name, c in zip(X.columns, clf.coef_[0])},
    "test_auc": float(auc),
}
with open("model_params.json", "w") as f:
    json.dump(params, f, indent=2)
print("\nSaved coefficients to model_params.json")

# 7. Sample predictions (used to sanity-check the DAX formula)
samples = X_test.head(3).copy()
samples["predicted_churn_prob"] = proba[:3]
print("\nSample predictions:")
print(samples.to_string())
