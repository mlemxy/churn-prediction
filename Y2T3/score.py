"""
score.py — offline batch churn scoring
Loads churn_model_edge.onnx and scores a CSV of customer RFM values.
No internet, no database, no Docker required.

Usage:
    python score.py customers.csv
    python score.py customers.csv --output results.csv

Input CSV must have columns: customer_id, recency, frequency, monetary
Output adds: log_monetary, churn_prob, risk_tier, action
"""

import sys
import argparse
import csv
import math
import os
import time

import numpy as np
import onnxruntime as ort

ONNX_PATH = "churn_model_edge.onnx"
HIGH_RISK  = 0.65
MED_RISK   = 0.40

def risk_tier(prob):
    if prob >= HIGH_RISK:
        return "HIGH", "offer win-back coupon at POS"
    if prob >= MED_RISK:
        return "MEDIUM", "show cross-sell prompt"
    return "LOW", "award loyalty points"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument("--output", default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    assert os.path.exists(ONNX_PATH), f"Model not found: {ONNX_PATH}"
    assert os.path.exists(args.input), f"Input not found: {args.input}"

    session    = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    latencies = []

    for row in rows:
        recency    = float(row["recency"])
        frequency  = float(row["frequency"])
        monetary   = float(row["monetary"])
        log_m      = math.log1p(monetary)

        x = np.array([[recency, frequency, log_m]], dtype=np.float32)

        t0   = time.perf_counter()
        prob = float(session.run(None, {input_name: x})[1][0][1])
        latencies.append((time.perf_counter() - t0) * 1000)

        tier, action = risk_tier(prob)
        results.append({
            "customer_id":   row["customer_id"],
            "recency":       recency,
            "frequency":     frequency,
            "monetary":      monetary,
            "log_monetary":  round(log_m, 4),
            "churn_prob":    round(prob, 4),
            "risk_tier":     tier,
            "action":        action,
        })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    lats = np.array(latencies)
    print(f"Scored {len(results)} customers -> {args.output}")
    print(f"Latency (n={len(lats)})  p50={np.percentile(lats,50):.3f}ms  p95={np.percentile(lats,95):.3f}ms  p99={np.percentile(lats,99):.3f}ms")
    print(f"Risk distribution: HIGH={sum(1 for r in results if r['risk_tier']=='HIGH')}  MEDIUM={sum(1 for r in results if r['risk_tier']=='MEDIUM')}  LOW={sum(1 for r in results if r['risk_tier']=='LOW')}")

if __name__ == "__main__":
    main()
