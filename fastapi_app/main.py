import os
import json
import uvicorn
import pickle
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from datetime import datetime

# Defines the log file path.
LOG_FILE_PATH = os.path.join("logs", "prediction_logs.json")

# Creates the FastAPI app.
app = FastAPI(title="FastAPI Prediction Service")

# Loads the model.
with open("sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

# Inputs as strings.
class PredictionRequest(BaseModel):
    text: str
    true_sentiment: Optional[str] = None

# Define predictions, pos or neg.
@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = sentiment_model.predict([request.text])[0]
    predicted_sentiment = "positive" if prediction == 1 else "negative"

    # Ensure logs dir exists (sanity check due to previous log issues).
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_text": request.text,
        "predicted_sentiment": predicted_sentiment,
        "true_sentiment": request.true_sentiment,
    }

    try:
        with open(LOG_FILE_PATH, "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
        print(f"📁 Logged to {LOG_FILE_PATH}")
    except Exception as e:
        print(f"❌ ERROR writing log: {e}")

    return {"predicted_sentiment": predicted_sentiment}

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Prediction Service!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
