import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📡", layout="wide")

@st.cache_resource
def load_models():
    rf  = joblib.load("rf_smote.pkl")
    xgb = joblib.load("xgb_smote.pkl")
    cols = json.load(open("feature_columns.json"))
    return rf, xgb, cols

rf_model, xgb_model, FEATURE_COLS = load_models()

@st.cache_resource
def load_explainers():
    return shap.TreeExplainer(rf_model), shap.TreeExplainer(xgb_model)

explainer_rf, explainer_xgb = load_explainers()

def empty_input():
    return pd.DataFrame([[0] * len(FEATURE_COLS)], columns=FEATURE_COLS)

def shap_plot(explainer, row_df, model_name, is_rf=True):
    sv = explainer.shap_values(row_df)
    arr = np.array(sv)
    if is_rf:
        if arr.ndim == 3 and arr.shape[-1] == 2:
            vals = arr[0, :, 1]
        elif arr.ndim == 3:
            vals = arr[1, 0, :]
        else:
            vals = arr[0]
    else:
        vals = arr[0]

    importance = pd.Series(np.abs(vals), index=FEATURE_COLS).sort_values(ascending=False).head(10)
    top_features = importance.index.tolist()
    top_vals = pd.Series(vals, index=FEATURE_COLS)[top_features]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['tomato' if v > 0 else 'steelblue' for v in top_vals]
    ax.barh(top_features[::-1], top_vals[::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on churn prediction)")
    ax.set_title(f"SHAP Explanation — {model_name}")
    plt.tight_layout()
    return fig

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📡 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer will churn, with SHAP-based explanations.")

model_choice = st.sidebar.radio("Model", ["Random Forest (SMOTE)", "XGBoost (SMOTE)"])
use_rf = model_choice.startswith("Random")
model = rf_model if use_rf else xgb_model
explainer = explainer_rf if use_rf else explainer_xgb

tab1, tab2 = st.tabs(["🔍 Single Customer", "📂 Batch Prediction (CSV)"])

# ── TAB 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter customer information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account info**")
        tenure      = st.slider("Tenure (months)", 0, 72, 12)
        monthly     = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0)
        total       = st.slider("Total Charges ($)", 0.0, 8684.0, 600.0)
        senior      = st.checkbox("Senior Citizen")
        partner     = st.checkbox("Has Partner")
        dependents  = st.checkbox("Has Dependents")

    with col2:
        st.markdown("**Services**")
        phone       = st.checkbox("Phone Service")
        multi_lines = st.checkbox("Multiple Lines")
        internet    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec  = st.checkbox("Online Security")
        online_back = st.checkbox("Online Backup")
        dev_prot    = st.checkbox("Device Protection")
        tech_sup    = st.checkbox("Tech Support")
        stream_tv   = st.checkbox("Streaming TV")
        stream_mov  = st.checkbox("Streaming Movies")

    with col3:
        st.markdown("**Contract & Billing**")
        contract    = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless   = st.checkbox("Paperless Billing")
        payment     = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    if st.button("🔮 Predict", use_container_width=True):
        inp = empty_input()

        # Continuous (normalized)
        inp["tenure"]         = tenure / 72.0
        inp["MonthlyCharges"] = monthly / 120.0
        inp["TotalCharges"]   = total / 8684.0

        # Binary direct
        inp["SeniorCitizen"]          = int(senior)
        inp["Partner_Yes"]            = int(partner)
        inp["Dependents_Yes"]         = int(dependents)
        inp["PhoneService_Yes"]       = int(phone)
        inp["MultipleLines_Yes"]      = int(multi_lines)
        inp["OnlineSecurity_Yes"]     = int(online_sec)
        inp["OnlineBackup_Yes"]       = int(online_back)
        inp["DeviceProtection_Yes"]   = int(dev_prot)
        inp["TechSupport_Yes"]        = int(tech_sup)
        inp["StreamingTV_Yes"]        = int(stream_tv)
        inp["StreamingMovies_Yes"]    = int(stream_mov)
        inp["PaperlessBilling_Yes"]   = int(paperless)

        # gender → drop-first = Male kept
        inp["gender_Male"] = int(True)  # default, no gender field needed

        # Multiple lines — no phone service
        inp["MultipleLines_No phone service"] = int(not phone)

        # Internet service — DSL is reference (drop-first)
        inp["InternetService_Fiber optic"] = int(internet == "Fiber optic")
        inp["InternetService_No"]          = int(internet == "No")

        # No internet service flags
        no_inet = internet == "No"
        inp["OnlineSecurity_No internet service"]   = int(no_inet)
        inp["OnlineBackup_No internet service"]     = int(no_inet)
        inp["DeviceProtection_No internet service"] = int(no_inet)
        inp["TechSupport_No internet service"]      = int(no_inet)
        inp["StreamingTV_No internet service"]      = int(no_inet)
        inp["StreamingMovies_No internet service"]  = int(no_inet)

        # Contract — Month-to-month is reference
        inp["Contract_One year"]  = int(contract == "One year")
        inp["Contract_Two year"]  = int(contract == "Two year")

        # Payment — Bank transfer is reference
        inp["PaymentMethod_Electronic check"]       = int(payment == "Electronic check")
        inp["PaymentMethod_Mailed check"]           = int(payment == "Mailed check")
        inp["PaymentMethod_Credit card (automatic)"] = int(payment == "Credit card (automatic)")

        proba = model.predict_proba(inp)[0][1]
        pred  = int(proba >= 0.5)

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            if pred == 1:
                st.error(f"⚠️ **Churn predicted** — Probability: {proba:.1%}")
            else:
                st.success(f"✅ **No churn predicted** — Probability: {proba:.1%}")
            st.progress(float(proba))
        with r2:
            fig = shap_plot(explainer, inp, model_choice, is_rf=use_rf)
            st.pyplot(fig)
            plt.close()

# ── TAB 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload a CSV file")
    st.markdown("The CSV must have the same columns as the training data (after preprocessing).")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.write(f"**{len(df_batch)} customers loaded**")
        st.dataframe(df_batch.head())

        missing = [c for c in FEATURE_COLS if c not in df_batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X_batch = df_batch[FEATURE_COLS]

            if st.button("🔮 Predict All", use_container_width=True):
                probas = model.predict_proba(X_batch)[:, 1]
                preds  = (probas >= 0.5).astype(int)

                df_batch["Churn_Predicted"]    = preds
                df_batch["Churn_Probability"]  = probas.round(3)
                df_batch["Risk"] = pd.cut(
                    probas,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["🟢 Low", "🟡 Medium", "🔴 High"]
                )

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total customers", len(df_batch))
                c2.metric("Predicted churners", int(preds.sum()))
                c3.metric("Churn rate", f"{preds.mean():.1%}")

                st.dataframe(
                    df_batch[["Churn_Predicted", "Churn_Probability", "Risk"]].head(50),
                    use_container_width=True
                )

                csv_out = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download results", csv_out, "churn_predictions.csv", "text/csv")

                churners = X_batch[preds == 1]
                if len(churners) > 0:
                    st.subheader("SHAP explanation — first predicted churner")
                    fig2 = shap_plot(explainer, churners.iloc[[0]], model_choice, is_rf=use_rf)
                    st.pyplot(fig2)
                    plt.close()