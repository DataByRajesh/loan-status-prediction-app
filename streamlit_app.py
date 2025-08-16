
import os
import io
import pandas as pd
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

st.set_page_config(page_title="Spark MLlib Model - Web App", layout="wide")

st.title("🔮 Spark MLlib Model — Batch & Single Prediction")
st.caption("Upload data with the same column names your training pipeline expects. The app loads a saved Spark ML PipelineModel and runs .transform().")

# ---- Spark session ----
MODEL_DIR = os.getenv("MODEL_DIR", "model")
APP_NAME = os.getenv("APP_NAME", "mllib-webapp")
MASTER = os.getenv("SPARK_MASTER", "local[*]")

@st.cache_resource(show_spinner=False)
def get_spark():
    return (
        SparkSession.builder
        .appName(APP_NAME)
        .master(MASTER)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return PipelineModel.load(model_path)

spark = get_spark()
st.success(f"SparkSession started (master={MASTER}).")

# ---- Model loading ----
try:
    model = load_model(MODEL_DIR)
    st.success(f"Loaded PipelineModel from: {MODEL_DIR}")
except Exception as e:
    st.error(f"Failed to load model from '{MODEL_DIR}'. Make sure the path exists and is a Spark PipelineModel. Error: {e}")
    st.stop()

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Settings")
    MODEL_DIR = st.text_input("Model directory", MODEL_DIR)
    show_prob = st.checkbox("Show probabilities (if available)", value=True)
    show_raw = st.checkbox("Show raw predictions", value=False)
    st.caption("Change path then click 'Reload model'.")
    if st.button("Reload model"):
        try:
            model = load_model(MODEL_DIR)
            st.success(f"Reloaded model from: {MODEL_DIR}")
        except Exception as e:
            st.error(f"Reload failed: {e}")

tab1, tab2 = st.tabs(["📄 Single row input", "📦 Batch CSV upload"])

# ---- Single row prediction ----
with tab1:
    st.subheader("Single Row Prediction")
    st.caption("Paste a JSON object with keys matching your raw feature columns. Example: {"age": 42, "income": 50000, "gender": "M"}")
    json_text = st.text_area("JSON row", value="", height=140, placeholder='{"feature1": 1.0, "feature2": "A"}')
    if st.button("Predict (single)"):
        if not json_text.strip():
            st.warning("Please paste a JSON object first.")
        else:
            try:
                import json as _json
                obj = _json.loads(json_text)
                pdf = pd.DataFrame([obj])
                sdf = spark.createDataFrame(pdf)
                preds = model.transform(sdf)
                cols = [c for c in ["prediction", "probability", "rawPrediction"] if c in preds.columns]
                # Keep original columns too
                select_cols = list(sdf.columns) + cols
                out_pdf = preds.select(*select_cols).toPandas()
                st.dataframe(out_pdf)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---- Batch prediction ----
with tab2:
    st.subheader("Batch CSV Upload")
    up = st.file_uploader("Upload a CSV file", type=["csv"])
    if up is not None:
        try:
            # Read into Pandas, then to Spark to ensure clean schema handling
            pdf = pd.read_csv(up)
            st.write(f"Detected {pdf.shape[0]} rows, {pdf.shape[1]} columns.")
            sdf = spark.createDataFrame(pdf)
            preds = model.transform(sdf)
            cols = [c for c in ["prediction", "probability", "rawPrediction"] if c in preds.columns]
            select_cols = list(sdf.columns) + cols
            out_pdf = preds.select(*select_cols).toPandas()
            st.success("Predictions generated.")
            st.dataframe(out_pdf.head(50))

            # Download
            buf = io.BytesIO()
            out_pdf.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download predictions CSV", data=buf, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.divider()
st.caption("Tips: ensure your input column names match the raw feature columns your pipeline expects. If you used StringIndexer/OneHotEncoder/VectorAssembler, pass the original raw columns — the pipeline will build features internally.")
