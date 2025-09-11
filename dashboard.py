import os
from typing import Optional, Callable, Tuple, Any
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from cdp_utils import aggregate_transactional_features, apply_final_scoring


# ---------- Utility / model loading ----------
def try_load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def load_disk_models(reg_path="regression_model.pkl",
                     clf_path="signal_classification_model.pkl",
                     scl_path="standard_scaler_model.pkl"):
    reg = clf = scl = None
    try:
        reg = try_load_model(reg_path)
    except Exception:
        reg = None
    try:
        clf = try_load_model(clf_path)
    except Exception:
        clf = None
    if os.path.exists(scl_path):
        try:
            scl = try_load_model(scl_path)
        except Exception:
            scl = None
    return reg, clf, scl

def preprocess_features(features_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    med = df.median(numeric_only=True)
    df = df.fillna(med).fillna(0)
    return df

def predict_with_models(reg, clf, scl, features_df: pd.DataFrame):
    X = features_df.values if scl is None else scl.transform(features_df)
    pred_score = reg.predict(X) if reg is not None else np.full((len(X),), np.nan)
    pred_signal_raw = clf.predict(X) if clf is not None else np.array([None]*len(X))
    signal_map = {0: "Green", 1: "Red", 2: "Yellow"}
    labels = [signal_map.get(s, s) for s in pred_signal_raw]
    return pred_score, pred_signal_raw, labels

# ---------- Dashboard renderer ----------
def render_dashboard(
    *,
    prediction_results: Optional[pd.DataFrame] = None,
    input_features: Optional[pd.DataFrame] = None,
    analyze_fn: Optional[Callable[..., str]] = None,
    gemini_model_name: str = "gemini-2.5-flash",
    use_upload: bool = False,      # <-- default: False (no uploader shown)
):
    """
    Dynamic dashboard:
    - Uses `prediction_results` & `input_features` if provided (preferred).
    - If `use_upload=True` dashboard will also show an upload widget (optional).
    - If no predictions provided and use_upload=False, dashboard simply explains how to get results.
    """

    st.header("Dynamic Dashboard")

    results = None
    features = None

    # Optionally allow uploader (disabled by default)
    uploaded = None
    if use_upload:
        uploaded = st.file_uploader(
            "Upload cleaned CSV (Partner_Code + features) — dashboard will auto-predict if local models exist.",
            type=["csv"]
        )

    # 1) Prefer results passed from main app
    if prediction_results is not None and not prediction_results.empty:
        results = prediction_results.copy()
        features = input_features.copy() if input_features is not None else None

    # If nothing passed and uploader is enabled
    if (results is None or results.empty) and use_upload and uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        results = df_raw.copy()

    if results is None or results.empty:
        st.info("No prediction results available. Run Tab 1 prediction or pass prediction_results.")
        return

    results = results.reset_index(drop=True)

    # ✅ Always prefer Final columns if available
    score_col = "Final Score"
    signal_col = "Final Signal"

    # ----------- SECTION SELECTOR -----------
    section_options = [
        "All",
        "Key Insights",
        "Visualizations",
        "Top Partners",
        "Inspect & AI Insight",
        "Download Results"
    ]
    selected_section = st.selectbox("Choose dashboard section to view", options=section_options, index=0)

    def show(section_name: str) -> bool:
        return selected_section == "All" or selected_section == section_name

    # ---------- KPIs ----------
    if show("Key Insights"):
        st.subheader("Key Insights")
        c1, c2, c3, c4 = st.columns(4)
        counts = results[signal_col].value_counts().to_dict() if signal_col in results.columns else {}
        green = counts.get('Green', 0)
        yellow = counts.get('Yellow', 0)
        red = counts.get('Red', 0)
        mean_score = results[score_col].mean() if score_col in results.columns else np.nan

        c1.metric(" Green (Safe)", f"{int(green)}")
        c2.metric(" Yellow (Watchlist)", f"{int(yellow)}")
        c3.metric(" Red (High Risk)", f"{int(red)}")
        c4.metric("Mean Score", f"{mean_score:.2f}" if not np.isnan(mean_score) else "N/A")
        st.markdown("---")

    # ---------- Visualizations ----------
    if show("Visualizations"):
        st.subheader("Visualizations")
        viz_col, control_col = st.columns([3,1])

        with control_col:
            st.markdown("**Filters**")
            if score_col in results.columns:
                min_val = float(results[score_col].min())
                max_val = float(results[score_col].max())
                score_range = st.slider("Score range", min_value=0.0, max_value=100.0, value=(min_val, max_val))
            else:
                score_range = (0.0, 100.0)

            avail_signals = sorted(results[signal_col].unique()) if signal_col in results.columns else []
            selected_signals = st.multiselect("Signals", options=avail_signals, default=avail_signals)
            show_labels = st.checkbox("Show partner codes on points", value=False)

        viz_df = results.copy()
        if score_col in viz_df.columns:
            viz_df = viz_df[(viz_df[score_col] >= score_range[0]) & (viz_df[score_col] <= score_range[1])]
        if signal_col in viz_df.columns and selected_signals:
            viz_df = viz_df[viz_df[signal_col].isin(selected_signals)]

        with viz_col:
            if score_col in viz_df.columns and signal_col in viz_df.columns:
                fig_box = go.Figure()
                for sig, color in [("Green","#22C55E"), ("Yellow","#F59E0B"), ("Red","#EF4444")]:
                    subset = viz_df[viz_df[signal_col]==sig]
                    if subset.empty: continue
                    fig_box.add_trace(go.Box(
                        y=subset[score_col],
                        name=sig,
                        marker_color=color,
                        boxmean="sd",
                        boxpoints="all",
                        hovertemplate="Signal: %{name}<br>Score: %{y:.2f}<extra></extra>"
                    ))
                if show_labels and 'Partner_Code' in viz_df.columns:
                    fig_box.add_trace(go.Scatter(
                        x=viz_df[signal_col],
                        y=viz_df[score_col],
                        mode='text',
                        text=viz_df['Partner_Code'].astype(str),
                        textposition='top center',
                        hoverinfo='skip',
                        showlegend=False
                    ))
                fig_box.update_layout(title="Score distribution by Signal", yaxis_title=score_col)
                st.plotly_chart(fig_box, use_container_width=True)

            if signal_col in viz_df.columns:
                pie_df = viz_df[signal_col].value_counts().reset_index()
                pie_df.columns = ['Signal','Count']
                import plotly.express as px
                fig_pie = px.pie(pie_df, names='Signal', values='Count', title="Signal Share",
                                 hole=0.4,
                                 color='Signal',
                                 color_discrete_map={'Green':'#22C55E','Yellow':'#F59E0B','Red':'#EF4444'})
                st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("---")

       # ---------- Top risky / safe ----------
    if show("Top Partners"):
        st.subheader("Top risky / safe partners")
        top_n = st.number_input("Top N entries", min_value=1, max_value=100, value=10)

        # prefer Final Score/Final Signal; if Final columns not present, warn and fallback to Predicted
        if "Final Score" in results.columns and "Final Signal" in results.columns:
            display_cols = ["Partner_Code", "Final Score", "Final Signal"]
            score_for_sort = "Final Score"
        elif "Predicted Score" in results.columns and "Predicted Signal" in results.columns:
            display_cols = ["Partner_Code", "Predicted Score", "Predicted Signal"]
            score_for_sort = "Predicted Score"
            st.warning("Final Score/Signal not found — showing Predicted Score/Signal instead.")
        else:
            # best-effort fallback
            display_cols = [c for c in ["Partner_Code","Final Score","Predicted Score","Final Signal","Predicted Signal"] if c in results.columns]
            score_for_sort = next((c for c in ["Final Score","Predicted Score"] if c in results.columns), None)
            st.warning("Showing best-available columns.")

        if score_for_sort:
            top_risk = results[results[display_cols[2] if len(display_cols)>2 else display_cols[1]] == "Red"].sort_values(score_for_sort).head(top_n) if ("Final Signal" in results.columns or "Predicted Signal" in results.columns) else results.sort_values(score_for_sort).head(top_n)
            top_safe = results[results[display_cols[2] if len(display_cols)>2 else display_cols[1]] == "Green"].sort_values(score_for_sort, ascending=False).head(top_n) if ("Final Signal" in results.columns or "Predicted Signal" in results.columns) else results.sort_values(score_for_sort, ascending=False).head(top_n)
        else:
            top_risk = results.head(top_n)
            top_safe = results.head(top_n)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Risk (Red)**")
            st.dataframe(top_risk.reset_index(drop=True)[display_cols], use_container_width=True)
        with c2:
            st.markdown("**Top Safe (Green)**")
            st.dataframe(top_safe.reset_index(drop=True)[display_cols], use_container_width=True)
        st.markdown("---")



    # ---------- Inspect partner & AI insight  ----------
    if show("Inspect & AI Insight"):
        st.subheader("Inspect partner & get AI insight")

        # partner selector (give a unique key to avoid collisions)
        partner_list = results['Partner_Code'].tolist() if 'Partner_Code' in results.columns else list(results.index.astype(str))
        chosen = st.selectbox("Choose partner", partner_list, key="dashboard_inspect_partner")

        # get partner row from results (robust)
        try:
            prow = results[results['Partner_Code'] == chosen].iloc[0]
        except Exception:
            # fallback to index-based selection
            try:
                idx = results.index[results.index.astype(str) == chosen][0]
                prow = results.loc[idx]
            except Exception:
                prow = results.iloc[0]

        # show minimal partner summary (no large feature lists)
        try:
            score_display = prow.get("Final Score", prow.get("Predicted Score", np.nan))
            score_text = f"{score_display:.2f}" if pd.notna(score_display) else "N/A"
        except Exception:
            score_text = "N/A"
        sig_text = prow.get("Final Signal", prow.get("Predicted Signal", "N/A"))
        st.markdown(f"**{chosen}** — Score: **{score_text}** — Signal: **{sig_text}**")


        # Gather transactional KPIs

        kpis = pd.Series(dtype="float64")
        if input_features is not None:
            try:
                if 'Partner_Code' in input_features.columns:
                    tmp = input_features[input_features['Partner_Code'] == chosen]
                    if not tmp.empty:
                        kpis = tmp.iloc[0]
                else:
                    # ordinal fallback: align by results index
                    idxs = results[results['Partner_Code'] == chosen].index
                    if len(idxs) > 0:
                        ridx = idxs[0]
                        if ridx < len(input_features):
                            kpis = input_features.reset_index(drop=True).iloc[ridx]
            except Exception:
                kpis = pd.Series(dtype="float64")

        # -------------------------
        # Gather original CDP input row (internal only)
        # try few common session_state keys where you may have stored the uploaded CDP input
        # -------------------------
        partner_input_row = pd.Series(dtype="float64")
        for key in ("cdp_input", "cdp_uploaded", "raw_input", "input_raw", "cdp_input_cleaned"):
            if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
                df_input = st.session_state[key]
                if "Partner_Code" in df_input.columns:
                    tmp = df_input[df_input["Partner_Code"] == chosen]
                    if not tmp.empty:
                        partner_input_row = tmp.drop(columns=["Partner_Code"], errors="ignore").iloc[0]
                        break

        # -------------------------
        # Helper: convert series -> readable lines (internal)
        # -------------------------
        def series_to_lines(s: pd.Series, keys: list, fmt="{:.2f}"):
            lines = []
            if s is None or s.empty:
                return lines
            for k in keys:
                if k in s.index:
                    try:
                        lines.append(f"- {k}: {fmt.format(float(s.get(k)))}")
                    except Exception:
                        lines.append(f"- {k}: {s.get(k)}")
            return lines

        # KPI keys we care about (only those present will be used)
        kpi_keys = ["avg_days_past_due", "max_days_past_due", "late_ratio",
                    "dispute_ratio", "adjustment_ratio", "allocation_ratio",
                    "collection_rate", "total_invoices", "total_installments",
                    "total_invoice_amount"]
        kpi_lines = series_to_lines(kpis, kpi_keys)

        # pick top numeric CDP input features (up to 6) for the prompt only
        input_lines = []
        if not partner_input_row.empty:
            numeric_input = pd.to_numeric(partner_input_row, errors="coerce").dropna()
            if not numeric_input.empty:
                top_input = numeric_input.abs().nlargest(6)
                for nm, v in top_input.items():
                    input_lines.append(f"- {nm}: {v:.2f}")

        # fallback text if missing (these are *only* used inside the prompt)
        if not kpi_lines:
            kpi_lines = ["- (no transactional KPI snapshot available)"]
        if not input_lines:
            input_lines = ["- (no CDP input snapshot available)"]

        # -------------------------
        # Build AI prompt (forces LLM to reference BOTH sources)
        # -------------------------
        final_score_str = f"{prow.get('Final Score', prow.get('Predicted Score', 'N/A')):.2f}" \
            if pd.notna(prow.get('Final Score', prow.get('Predicted Score', np.nan))) else "N/A"
        final_signal_str = prow.get('Final Signal', prow.get('Predicted Signal', 'N/A'))
        predicted_score = prow.get('Predicted Score', 'N/A')
        predicted_signal = prow.get('Predicted Signal', 'N/A')
        override_reason = prow.get('OverrideReason', 'None')
        adjustment_delta = prow.get('AdjustmentDelta', 'N/A')

        # tone control + trigger
        tone = st.selectbox("Tone of AI response", ["Concise", "Analytical", "Actionable"], index=1, key="dashboard_ai_tone")
        if st.button("Get AI insight for selected partner", key="dashboard_ai_button"):
            # compact prompt that includes both KPI and CDP input snippets (but those snippets are NOT shown in UI)
            prompt = f"""
You are a senior credit risk analyst. Provide a short structured analysis for Partner {chosen}.
Tone: {tone}

Context:
- Final Score: {final_score_str}
- Final Signal: {final_signal_str}
- Model Predicted Score: {predicted_score}
- Model Predicted Signal: {predicted_signal}
- Override reason: {override_reason}
- Adjustment delta: {adjustment_delta}

Top Transactional KPIs:
{chr(10).join(kpi_lines)}

Top CDP input features:
{chr(10).join(input_lines)}

Requirements (keep concise, max ~180 words):
1) Key Factors — 3 bullets referencing BOTH KPIs and CDP inputs (call out contradictions).
2) Risk Assessment — 1–2 sentences summarizing urgency/likelihood.
3) Recommendation — Approve / Review / Decline with one-line justification.
Make sure the confidentiality maintain use the data whatever the data is needed for analysis but try not to reveal the data in the response.
"""
            # call AI (analyze_fn is passed into render_dashboard)
            if analyze_fn is None or not callable(analyze_fn):
                ai_text = "AI analysis not configured in this environment."
            else:
                try:
                    # analyze_fn(prompt, model_name)
                    ai_text = analyze_fn(prompt, gemini_model_name)
                except Exception as e:
                    ai_text = f"Error calling AI: {e}"

            st.subheader("AI Analysis")
            st.markdown(ai_text)

        st.markdown("---")


