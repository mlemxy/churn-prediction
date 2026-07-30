"""
sidecar.py — OSPOS to ONNX churn-scoring bridge.

Polls the OSPOS MariaDB database every 5 seconds for newly completed sales.
On each new sale, computes RFM features for that customer from their full
completed-sale history, runs ONNX inference, maps the churn probability to
a risk tier via the 3-tier rule engine, and prints the retention action.

Reconstructed Week 11 from the documented behavior in wiki/Sidecar-Setup.md
after the original file was lost in the Windows-to-macOS migration.
"""

import time
import math
import pymysql
import numpy as np
import onnxruntime as ort

# --- Configuration (matches docker-compose.yml) ---
DB_CONFIG = {
    "host": "mysql",
    "user": "admin",
    "password": "pointofsale",
    "database": "ospos",
    "port": 3306,
    # Fix for MariaDB default REPEATABLE READ isolation level, which caused
    # new connections to read a frozen snapshot and miss newly committed
    # sales. See Decisions.md / Sidecar-Setup.md for the full explanation.
    "init_command": "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED",
}

ONNX_MODEL_PATH = "churn_model_edge.onnx"
POLL_INTERVAL_SECONDS = 5

# Rule engine thresholds (confirmed in wiki/Sidecar-Setup.md)
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.40


def get_connection():
    """Open a fresh connection per call. No persistent connection is kept,
    per the documented design decision (avoids stale-connection drop after
    a backfill burst on startup)."""
    return pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)


def get_new_sales(conn, last_seen_sale_id):
    """Return completed sales (sale_status = 0) with sale_id greater than
    last_seen, for customers with customer_id > 0 (customer_id = 0 means
    no customer was attached to the sale, and is filtered out)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sale_id, customer_id, sale_time
            FROM ospos_sales
            WHERE sale_id > %s
              AND customer_id > 0
              AND sale_status = 0
            ORDER BY sale_id ASC
            """,
            (last_seen_sale_id,),
        )
        return cur.fetchall()


def compute_rfm(conn, customer_id, as_of):
    """Compute Recency, Frequency, log(Monetary) for a customer from their
    full completed-sale history, as of a given reference time."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.sale_id, s.sale_time,
                   SUM(si.item_unit_price * si.quantity_purchased) AS sale_total
            FROM ospos_sales s
            JOIN ospos_sales_items si ON si.sale_id = s.sale_id
            WHERE s.customer_id = %s
              AND s.sale_status = 0
            GROUP BY s.sale_id, s.sale_time
            """,
            (customer_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return None

    frequency = len(rows)
    monetary = sum(float(r["sale_total"]) for r in rows)
    last_purchase = max(r["sale_time"] for r in rows)
    recency_days = (as_of - last_purchase).days
    if recency_days < 0:
        recency_days = 0

    log_monetary = math.log1p(monetary)
    return recency_days, frequency, log_monetary


def score(session, recency, frequency, log_monetary):
    """Run ONNX inference. Input tensor: float32[1, 3] = [recency, frequency, log_monetary]."""
    input_name = session.get_inputs()[0].name
    x = np.array([[recency, frequency, log_monetary]], dtype=np.float32)
    outputs = session.run(None, {input_name: x})
    # onnxmltools-exported sklearn classifiers typically return
    # [label_array, probability_array_of_dicts_or_array]; take P(churn=1).
    probs = outputs[1]
    if isinstance(probs, list):
        churn_prob = float(probs[0][1])
    else:
        churn_prob = float(probs[0][1])
    return churn_prob


def risk_tier_and_action(churn_prob):
    if churn_prob >= HIGH_RISK_THRESHOLD:
        return "HIGH RISK", "offer win-back coupon at POS"
    elif churn_prob >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM RISK", "show cross-sell prompt"
    else:
        return "LOW RISK", "award loyalty points"


def main():
    print("Sidecar running - polling every 5s. Ctrl+C to stop.\n")

    session = ort.InferenceSession(ONNX_MODEL_PATH)
    last_seen_sale_id = 0  # Start at 0 to backfill all existing sales on startup

    while True:
        try:
            conn = get_connection()
            new_sales = get_new_sales(conn, last_seen_sale_id)

            for sale in new_sales:
                sale_id = sale["sale_id"]
                customer_id = sale["customer_id"]
                sale_time = sale["sale_time"]

                rfm = compute_rfm(conn, customer_id, as_of=sale_time)
                if rfm is None:
                    last_seen_sale_id = max(last_seen_sale_id, sale_id)
                    continue

                recency, frequency, log_monetary = rfm
                churn_prob = score(session, recency, frequency, log_monetary)
                tier, action = risk_tier_and_action(churn_prob)

                monetary_display = math.expm1(log_monetary)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{timestamp}] sale_id={sale_id}  customer={customer_id}  "
                    f"R={recency}d  F={frequency}  M=${monetary_display:.2f}"
                )
                print(f"  {tier} ({churn_prob:.2f})   -> {action}\n")

                last_seen_sale_id = max(last_seen_sale_id, sale_id)

            conn.close()

        except Exception as e:
            print(f"[error] {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
