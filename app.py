"""
Diamond Price Predictor — Streamlit App
Loads model.pkl (a scikit-learn Pipeline) and predicts the price of a
diamond in USD from its physical and quality specifications.
"""

import os
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.pkl")

# Valid ordinal options (same order used during training)
CUT_OPTIONS     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_OPTIONS   = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_OPTIONS = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# Feature columns — must match training order exactly
FEATURE_COLS = ["carat", "depth", "table", "x", "y", "z", "cut", "color", "clarity"]


import joblib

# ---------------------------------------------------------------------------
# Load model (cached so it is read from disk only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="centered",
)

st.title("💎 Diamond Price Predictor")
st.markdown(
    "Enter the specifications of a diamond and get an instant price estimate "
    "in **US dollars**, powered by a machine-learning model trained on "
    "53,940 real diamonds."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Diamond Specifications")

    col1, col2 = st.columns(2)

    with col1:
        carat = st.number_input(
            "Carat (weight)",
            min_value=0.20, max_value=5.01, value=0.90, step=0.01,
            help="Weight of the diamond in carats (0.20 – 5.01)",
        )
        cut = st.selectbox(
            "Cut quality",
            options=CUT_OPTIONS,
            index=CUT_OPTIONS.index("Ideal"),
            help="Fair < Good < Very Good < Premium < Ideal",
        )
        color = st.selectbox(
            "Colour grade",
            options=COLOR_OPTIONS,
            index=COLOR_OPTIONS.index("G"),
            help="J (most yellow) → D (most colourless)",
        )
        clarity = st.selectbox(
            "Clarity grade",
            options=CLARITY_OPTIONS,
            index=CLARITY_OPTIONS.index("VS2"),
            help="I1 (most inclusions) → IF (internally flawless)",
        )

    with col2:
        depth = st.number_input(
            "Total depth % (depth)",
            min_value=43.0, max_value=79.0, value=61.5, step=0.1,
            help="z / mean(x, y) × 100 — typical range 56 – 65",
        )
        table = st.number_input(
            "Table width % (table)",
            min_value=43.0, max_value=95.0, value=57.0, step=0.1,
            help="Width of the diamond's top facet — typical range 53 – 62",
        )
        x = st.number_input(
            "Length (x mm)",
            min_value=0.10, max_value=10.74, value=4.50, step=0.01,
            help="Length of the diamond in mm",
        )
        y = st.number_input(
            "Width (y mm)",
            min_value=0.10, max_value=58.90, value=4.50, step=0.01,
            help="Width of the diamond in mm",
        )
        z = st.number_input(
            "Depth (z mm)",
            min_value=0.10, max_value=31.80, value=2.78, step=0.01,
            help="Depth of the diamond in mm",
        )

    submitted = st.form_submit_button("💰 Predict Price", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    try:
        model = load_model()

        input_df = pd.DataFrame([{
            "carat":   carat,
            "depth":   depth,
            "table":   table,
            "x":       x,
            "y":       y,
            "z":       z,
            "cut":     cut,
            "color":   color,
            "clarity": clarity,
        }])[FEATURE_COLS]

        predicted_price = float(model.predict(input_df)[0])

        st.divider()
        st.success(f"### 💎 Estimated Price: **${predicted_price:,.2f} USD**")
        st.caption(
            "This estimate comes from a machine-learning model trained on "
            "53,940 diamonds. Actual market price may differ based on "
            "certificate, fluorescence, and retailer margin."
        )

    except FileNotFoundError:
        st.error(
            f"Model file not found at `{MODEL_FILE}`. "
            "Please run the notebook to generate `model.pkl` first."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Dataset: *Diamonds Prices 2022* — Kaggle (CC BY 4.0). "
    "Model trained using scikit-learn Pipelines with 10-Fold Cross Validation."
)
