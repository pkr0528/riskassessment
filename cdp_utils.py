# cdp_utils.py
"""
CDP Utilities:
- aggregate_transactional_features: turn raw transaction rows into partner-level KPIs
- compute_transaction_index: normalize those KPIs into a single Transaction Index (TI)
- apply_final_scoring: blend model outputs + TI into Final Score & Final Signal

Usage in streamlit.py:
  txn_df = pd.read_csv("partner_transactional_dataset.csv")  # hardcoded
  agg_df = aggregate_transactional_features(txn_df, partner_col="Partner_Code")
  results = DataFrame with Partner_Code, Predicted Score, Predicted Signal
  final = apply_final_scoring(results, agg_df, partner_key="Partner_Code", alpha=0.7)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# ---------------------------
# 1. Aggregate raw → partner KPIs
# ---------------------------
def aggregate_transactional_features(txn_df: pd.DataFrame, partner_col: str = "Partner_Code") -> pd.DataFrame:
    df = txn_df.copy()

    # ensure datetime parsing
    for col in ["invoice_date", "installment_due_date", "payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # days past due
    if "payment_date" in df.columns and "installment_due_date" in df.columns:
        df["days_past_due"] = (df["payment_date"] - df["installment_due_date"]).dt.days
    else:
        df["days_past_due"] = np.nan

    # numeric safety casts
    for col in ["invoice_amount", "installment_amount", "payment_amount", "allocated_amount", "adjustment_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # dispute flag
    if "dispute_status" in df.columns:
        df["_is_dispute"] = df["dispute_status"].astype(str).str.upper().isin({"Y","YES","TRUE","1"})
    else:
        df["_is_dispute"] = False

    g = df.groupby(partner_col, dropna=False)

    agg = g.agg(
        total_invoices=("invoice_id","nunique"),
        total_installments=("installment_number","count"),
        total_invoice_amount=("invoice_amount","sum"),
        total_payment_amount=("payment_amount","sum"),
        total_allocated=("allocated_amount","sum"),
        total_adjustments=("adjustment_amount","sum"),
        disputes=("_is_dispute","sum"),
        avg_days_past_due=("days_past_due","mean"),
        max_days_past_due=("days_past_due","max")
    ).reset_index()

    # ratios
    def safe_div(a, b):
        return np.where((b==0)|(pd.isna(b)), 0.0, a/b)

    agg["dispute_ratio"] = safe_div(agg["disputes"], agg["total_invoices"])
    agg["adjustment_ratio"] = safe_div(agg["total_adjustments"], agg["total_invoice_amount"])
    agg["allocation_ratio"] = safe_div(agg["total_allocated"], agg["total_payment_amount"])
    agg["collection_rate"] = safe_div(agg["total_payment_amount"], agg["total_invoice_amount"])

    # late ratio from raw days_past_due
    late_counts = df.groupby(partner_col)["days_past_due"].apply(lambda s: (s > 0).sum()).reset_index(name="late_count")
    agg = agg.merge(late_counts, on=partner_col, how="left")
    agg["late_ratio"] = safe_div(agg["late_count"], agg["total_installments"])

    # fill NaNs
    numeric_cols = agg.select_dtypes(include=[np.number]).columns
    agg[numeric_cols] = agg[numeric_cols].fillna(0.0)

    return agg

# ---------------------------
# 2. Transaction Index (TI)
# ---------------------------
def _minmax_scale(s: pd.Series, invert: bool=False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(50.0, index=s.index)
    lo, hi = s.min(), s.max()
    if np.isclose(lo, hi):
        return pd.Series(50.0, index=s.index)
    scaled = (s - lo) / (hi - lo) * 100
    return 100 - scaled if invert else scaled

def compute_transaction_index(agg_df: pd.DataFrame,
                              params: Optional[List[str]]=None,
                              invert_map: Optional[Dict[str,bool]]=None) -> pd.Series:
    if params is None:
        params = ["avg_days_past_due","max_days_past_due","late_ratio",
                  "dispute_ratio","adjustment_ratio","allocation_ratio","collection_rate"]
    if invert_map is None:
        invert_map = {"collection_rate": True, "allocation_ratio": True}

    scores = []
    for col in params:
        if col not in agg_df.columns:
            scores.append(pd.Series(50.0, index=agg_df.index))
            continue
        invert = invert_map.get(col, False)
        scores.append(_minmax_scale(agg_df[col], invert=invert))
    ti = pd.concat(scores, axis=1).mean(axis=1).clip(0,100)
    return ti

# ---------------------------
# 3. Final Score + Signal
# ---------------------------
def apply_final_scoring(results_df: pd.DataFrame,
                        agg_df: pd.DataFrame,
                        partner_key: str="Partner_Code",
                        alpha: float=0.7,
                        hard_thresholds: Optional[Dict[str,Dict[str,object]]]=None,
                        cutoffs: Optional[Dict[str,float]]=None) -> pd.DataFrame:
    if hard_thresholds is None:
        hard_thresholds = {"max_days_past_due":{"gt":90,"force":"Red"}}
    if cutoffs is None:
        cutoffs = {"green":70.0, "yellow":50.0}

    merged = results_df.merge(agg_df, on=partner_key, how="left")
    merged["TransactionIndex"] = compute_transaction_index(merged[agg_df.columns])

    # ensure Predicted Score
    if "Predicted Score" in merged.columns:
        ms = pd.to_numeric(merged["Predicted Score"], errors="coerce").fillna(50.0)
    else:
        ms = pd.Series(50.0, index=merged.index)

    # transactional safety
    merged["TransactionalSafety"] = 100 - merged["TransactionIndex"]

    # blend
    a = max(0,min(1,float(alpha)))
    merged["Final Score"] = a*ms + (1-a)*merged["TransactionalSafety"]

    # signals
    final_signals, reasons = [], []
    for i,row in merged.iterrows():
        sig, reason = None, ""
        for col, rule in hard_thresholds.items():
            if col in merged.columns:
                try: val = float(row[col])
                except: continue
                if "gt" in rule and val > rule["gt"]:
                    sig, reason = rule["force"], f"{col}>{rule['gt']}"
                    break
                if "lt" in rule and val < rule["lt"]:
                    sig, reason = rule["force"], f"{col}<{rule['lt']}"
                    break
        if sig:
            final_signals.append(sig); reasons.append(reason); continue

        fs = row["Final Score"]
        if fs >= cutoffs["green"]:
            final_signals.append("Green"); reasons.append("")
        elif fs >= cutoffs["yellow"]:
            final_signals.append("Yellow"); reasons.append("")
        else:
            final_signals.append("Red"); reasons.append("")
    merged["Final Signal"] = final_signals
    merged["OverrideReason"] = reasons
    merged["AdjustmentDelta"] = merged["Final Score"] - ms

    return merged
