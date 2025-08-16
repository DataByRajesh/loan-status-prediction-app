
import os
from typing import List, Union, Dict, Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

APP_NAME = os.getenv("APP_NAME", "spark-ml-api")
MASTER = os.getenv("SPARK_MASTER", "local[*]")
MODEL_DIR = os.getenv("MODEL_DIR", "model")

app = FastAPI(title="Spark MLlib API", version="1.0.0")

def get_spark() -> SparkSession:
    return (SparkSession.builder
            .appName(APP_NAME)
            .master(MASTER)
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())

spark = get_spark()
model = PipelineModel.load(MODEL_DIR)

class Item(BaseModel):
    # Accept arbitrary payload — we validate dynamically
    __root__: Dict[str, Any]

@app.get("/healthz")
def healthz():
    return {"status": "ok", "master": MASTER, "model_dir": MODEL_DIR}

@app.post("/predict")
def predict(payload: Union[Item, List[Item]]):
    # Normalize to list of dicts
    if isinstance(payload, list):
        rows = [p.__root__ for p in payload]
    else:
        rows = [payload.__root__]
    pdf = pd.DataFrame(rows)
    sdf = spark.createDataFrame(pdf)
    out = model.transform(sdf)

    cols = [c for c in ["prediction", "probability", "rawPrediction"] if c in out.columns]
    select_cols = list(sdf.columns) + cols
    res = out.select(*select_cols).toPandas().to_dict(orient="records")
    # Cast numpy types to Python builtins for JSON
    from numpy import floating, integer
    def py(v):
        if isinstance(v, (floating,)):
            return float(v)
        if isinstance(v, (integer,)):
            return int(v)
        if hasattr(v, "values") and hasattr(v, "tolist"):
            return v.tolist()
        return v
    return [{k: py(v) for k, v in row.items()} for row in res]
