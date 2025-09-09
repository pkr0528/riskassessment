# dashboard.py
import os
from typing import Optional, Callable, Tuple, Any
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go

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
    use_upload: bool = False,           # <-- default: False (no uploader shown)
):
    """
    Dynamic dashboard:
    - Uses `prediction_results` & `input_features` if provided (preferred).
    - If `use_upload=True` dashboard will also show an upload widget (optional).
    - If no predictions provided and use_upload=False, dashboard simply explains how to get results.
    """

    st.header("📊 Dynamic Dashboard")

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

    # 2) If not passed and uploader is enabled & user uploaded file: compute or use present prediction cols
    if (results is None or results.empty) and use_upload and uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        # if predictions already present
        if 'Predicted Score' in df_raw.columns and 'Predicted Signal' in df_raw.columns:
            results = df_raw.copy()
            features = df_raw.drop(columns=['Partner_Code','Predicted Score','Predicted Signal'], errors='ignore')
        else:
            # try local model prediction
            reg, clf, scl = load_disk_models()
            if reg is None or clf is None:
                st.warning("Local models not found. Showing uploaded raw data.")
                results = df_raw.copy()
            else:
                feats = df_raw.drop(columns=['Partner_Code'], errors='ignore')
                feats_p = preprocess_features(feats)
                try:
                    pred_score, pred_signal_raw, labels = predict_with_models(reg, clf, scl, feats_p)
                    id_cols = df_raw[['Partner_Code']] if 'Partner_Code' in df_raw.columns else pd.DataFrame(index=feats_p.index)
                    results = id_cols.copy()
                    results['Predicted Score'] = pred_score
                    results['Predicted Signal'] = labels
                    features = feats_p
                    st.success("Predictions computed using local models.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    results = df_raw.copy()

    # If still nothing, guide the user (but do not show uploader if use_upload=False)
    if results is None or results.empty:
        if use_upload:
            st.info("Upload a cleaned CSV (Partner_Code + features) or pass prediction_results from Tab 1.")
        else:
            st.info("No prediction results available. Run Tab 1 prediction or call render_dashboard with prediction_results and input_features.")
        return

    # normalize
    results = results.reset_index(drop=True)

    # ----------- SECTION SELECTOR (single dropdown) -----------
    section_options = [
        "All",
        "Key Insights",
        "Visualizations",
        "Top Partners",
        "Inspect & AI Insight",
        "Download Results"
    ]
    selected_section = st.selectbox("Choose dashboard section to view", options=section_options, index=0)

    # helper to check if we should render a given section
    def show(section_name: str) -> bool:
        return selected_section == "All" or selected_section == section_name

    # ---------- KPIs ----------
    if show("Key Insights"):
        st.subheader("Key Insights")
        c1, c2, c3, c4 = st.columns(4)
        counts = results['Predicted Signal'].value_counts().to_dict() if 'Predicted Signal' in results.columns else {}
        green = counts.get('Green', 0)
        yellow = counts.get('Yellow', 0)
        red = counts.get('Red', 0)
        mean_score = results['Predicted Score'].mean() if 'Predicted Score' in results.columns else np.nan

        c1.metric("🟢 Green (Safe)", f"{int(green)}")
        c2.metric("🟡 Yellow (Watchlist)", f"{int(yellow)}")
        c3.metric("🔴 Red (High Risk)", f"{int(red)}")
        c4.metric("Mean Score", f"{mean_score:.2f}" if not np.isnan(mean_score) else "N/A")
        st.markdown("---")

    # ---------- Visualizations ----------
    if show("Visualizations"):
        st.subheader("Visualizations")
        viz_col, control_col = st.columns([3,1])

        with control_col:
            st.markdown("**Filters**")
            if 'Predicted Score' in results.columns:
                min_val = float(results['Predicted Score'].min())
                max_val = float(results['Predicted Score'].max())
                score_range = st.slider("Score range", min_value=0.0, max_value=100.0, value=(min_val, max_val))
            else:
                score_range = (0.0, 100.0)

            avail_signals = sorted(results['Predicted Signal'].unique()) if 'Predicted Signal' in results.columns else []
            selected_signals = st.multiselect("Signals", options=avail_signals, default=avail_signals)
            show_labels = st.checkbox("Show partner codes on points", value=False)

        viz_df = results.copy()
        if 'Predicted Score' in viz_df.columns:
            viz_df = viz_df[(viz_df['Predicted Score'] >= score_range[0]) & (viz_df['Predicted Score'] <= score_range[1])]
        if 'Predicted Signal' in viz_df.columns and selected_signals:
            viz_df = viz_df[viz_df['Predicted Signal'].isin(selected_signals)]

        with viz_col:
            # Box plot by signal (no histogram)
            if 'Predicted Score' in viz_df.columns and 'Predicted Signal' in viz_df.columns:
                fig_box = go.Figure()
                # add boxes for signals present (order stable)
                for sig, color in [("Green","#22C55E"), ("Yellow","#F59E0B"), ("Red","#EF4444")]:
                    subset = viz_df[viz_df['Predicted Signal']==sig]
                    if subset.empty:
                        continue
                    fig_box.add_trace(go.Box(
                        y=subset['Predicted Score'],
                        name=sig,
                        marker_color=color,
                        boxmean="sd",
                        boxpoints="all",   # show points
                        hovertemplate="Signal: %{name}<br>Score: %{y:.2f}<extra></extra>"
                    ))

                # optional text labels overlay
                if show_labels and 'Partner_Code' in viz_df.columns:
                    fig_box.add_trace(go.Scatter(
                        x=viz_df['Predicted Signal'],
                        y=viz_df['Predicted Score'],
                        mode='text',
                        text=viz_df['Partner_Code'].astype(str),
                        textposition='top center',
                        hoverinfo='skip',
                        showlegend=False
                    ))

                fig_box.update_layout(title="Score distribution by Signal (box + points)", yaxis_title="Predicted Score")
                st.plotly_chart(fig_box, use_container_width=True)

            # Pie chart for signal share
            if 'Predicted Signal' in viz_df.columns:
                pie_df = viz_df['Predicted Signal'].value_counts().reset_index()
                pie_df.columns = ['Signal','Count']
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

        if 'Predicted Signal' in results.columns and 'Predicted Score' in results.columns:
            top_risk = results[results['Predicted Signal']=='Red'].sort_values('Predicted Score').head(top_n)
            top_safe = results[results['Predicted Signal']=='Green'].sort_values('Predicted Score', ascending=False).head(top_n)
        elif 'Predicted Score' in results.columns:
            top_risk = results.sort_values('Predicted Score').head(top_n)
            top_safe = results.sort_values('Predicted Score', ascending=False).head(top_n)
        else:
            top_risk = results.head(top_n)
            top_safe = results.head(top_n)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Risk (Red)**")
            st.dataframe(top_risk.reset_index(drop=True), use_container_width=True)
        with c2:
            st.markdown("**Top Safe (Green)**")
            st.dataframe(top_safe.reset_index(drop=True), use_container_width=True)
        st.markdown("---")

    # ---------- Inspect partner & AI insight ----------
    if show("Inspect & AI Insight"):
        st.subheader("Inspect partner & get AI insight")
        partner_list = results['Partner_Code'].tolist() if 'Partner_Code' in results.columns else list(results.index.astype(str))
        chosen = st.selectbox("Choose partner", partner_list)

        prow = None
        try:
            prow = results[results['Partner_Code']==chosen].iloc[0]
            score_display = f"{prow.get('Predicted Score','N/A'):.2f}" if 'Predicted Score' in results.columns else "N/A"
            st.markdown(f"**{chosen}** — Score: **{score_display}** — Signal: **{prow.get('Predicted Signal','N/A')}**")
        except Exception:
            st.write("Partner row not found or missing columns.")

        # show feature snippet if available
        feats_src = input_features if input_features is not None else features
        if feats_src is not None:
            try:
                idx = results[results['Partner_Code']==chosen].index[0]
                frecord = feats_src.reset_index(drop=True).iloc[idx]
                st.markdown("**Top features for this partner**")
                for f,v in frecord.abs().nlargest(6).items():
                    st.write(f"- **{f}**: {v:.3f}")
            except Exception:
                pass

        # AI insight enriched prompt
        if analyze_fn is not None:
            tone = st.selectbox("Tone of AI response", ["Concise","Analytical","Actionable"], index=2)
            if st.button("🧠 Get AI insight for selected partner"):
                with st.spinner("Calling AI..."):
                    feat_text = ""
                    try:
                        idx = results[results['Partner_Code']==chosen].index[0]
                        if feats_src is not None:
                            feat_text = feats_src.reset_index(drop=True).iloc[idx].to_string()
                    except Exception:
                        feat_text = ""

                    prompt = f"""You are a senior credit risk analyst. Provide a structured, actionable analysis for Partner {chosen}.
- Predicted Score: {prow.get('Predicted Score','N/A')}
- Predicted Signal: {prow.get('Predicted Signal','N/A')}
Partner features snapshot:
{feat_text if feat_text else 'None'}

Return:
1) Key Factors (top 3)
2) Risk Assessment (1-2 sentences)
3) Clear Recommendation with one-line justification
4) 3-step Quick Action Plan (prioritized)
Format with headings and bullets. Tone: {tone}
"""

                    try:
                        ai_text = analyze_fn(prompt, gemini_model_name)
                    except Exception as e:
                        ai_text = f"Error calling AI: {e}"
                    st.subheader("AI Analysis")
                    st.markdown(ai_text)
        st.markdown("---")

    # ---------- Download results ----------
    if show("Download Results"):
        try:
            download_df = viz_df if 'viz_df' in locals() else results
        except Exception:
            download_df = results
        st.download_button("Download filtered results (CSV)", data=download_df.to_csv(index=False), file_name="dashboard_results.csv")

    