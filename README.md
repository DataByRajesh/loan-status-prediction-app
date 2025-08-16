
# Spark MLlib Web App (Streamlit)

This is a lightweight web UI to run predictions using a saved **Spark ML PipelineModel**.

## 1) Save your model from the notebook

In your training notebook (Spark MLlib), ensure you save the **trained PipelineModel**:

```python
from pyspark.ml import PipelineModel

# Suppose your trained pipeline model is named `pipelineModel` or `bestModel`
# Replace `pipelineModel` with the variable that holds your trained PipelineModel
pipelineModel.write().overwrite().save("model")
```

This creates a directory called `model/` with the Spark model metadata and stage params.

> If you only have a fitted Estimator (e.g., RandomForestClassificationModel without the preprocessing stages), consider wrapping everything into a single `Pipeline` and fitting it, so the web app can call `model.transform()` directly on raw inputs.

## 2) Run locally

```bash
# Option A: Python
pip install -r requirements.txt
export MODEL_DIR=./model  # or the path to your saved PipelineModel
streamlit run streamlit_app.py

# Option B: Docker
docker build -t mllib-webapp .
docker run -p 8501:8501 -e MODEL_DIR=/app/model -v $(pwd)/model:/app/model mllib-webapp
```

Open http://localhost:8501 in your browser.

## 3) Deploy

- **Streamlit Community Cloud**: Push this folder to GitHub, set `MODEL_DIR` to the path where your model lives in the repo (or upload separately).
- **Render / Railway / Azure WebApp for Containers**: Use the Dockerfile. Provide the model directory as a volume or bake into image.
- **Kubernetes**: Build the image and deploy; mount the model via a volume.

## 4) Using the app

- **Single row**: paste a JSON object with keys matching your raw feature columns (before feature engineering).
- **Batch CSV**: upload a CSV with the same column names as training data.
- The pipeline handles preprocessing (StringIndexer/OneHotEncoder/VectorAssembler/Scaler) and outputs `prediction`, optionally `probability` and `rawPrediction`.

## 5) Common gotchas

- **Java/Spark versions**: This project uses `pyspark==3.3.2` and installs Java 8 (OpenJDK). Align with your notebook versions if different.
- **Mismatched columns**: Ensure the uploaded CSV has the raw input columns the pipeline expects (not the assembled/encoded columns).
- **Label column**: If your CSV contains the label column (e.g., `loan_status`), it will be carried through but not required for inference.
- **Big files**: For large CSVs, adjust Streamlit's upload limit via `server.maxUploadSize` config or switch to a proper API service (FastAPI) with chunked uploads.

## 6) Optional: FastAPI skeleton (if you want an API later)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd

app = FastAPI(title="Spark ML API")
spark = SparkSession.builder.master("local[*]").appName("spark-api").getOrCreate()
model = PipelineModel.load("model")

class Item(BaseModel):
    # Define your raw features here, e.g. age: int; income: float; gender: str
    # Use Optional[...] with defaults if needed
    pass

@app.post("/predict")
def predict(item: dict):
    pdf = pd.DataFrame([item])
    sdf = spark.createDataFrame(pdf)
    pred = model.transform(sdf).select("prediction").toPandas().iloc[0,0]
    return {"prediction": float(pred)}
```

Happy deploying!


---

## 7) FastAPI (optional API deployment)

Build and run:

```bash
docker build -f Dockerfile.api -t spark-ml-api .
docker run -p 8000:8000 -e MODEL_DIR=/app/model -v $(pwd)/model:/app/model spark-ml-api
# Test
curl -X GET http://localhost:8000/healthz
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature1": 1, "feature2": "A"}'
```

Or run both UI and API via docker-compose:

```bash
docker compose up --build
```

Ensure your trained **PipelineModel** is saved to `./model` (or set `MODEL_DIR`).
