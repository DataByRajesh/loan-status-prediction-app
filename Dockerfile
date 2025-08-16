
FROM python:3.10-slim

# Install Java (needed by Spark)
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-8-jre-headless && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:${PATH}"

# Workdir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY streamlit_app.py ./

# Default model directory (mount your model there or bake it into the image)
ENV MODEL_DIR=/app/model

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
