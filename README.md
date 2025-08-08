# Assignment 5: Multi-Container Sentiment API + Monitoring Dashboard

This repo contains two Dockerized services:

1. **FastAPI Prediction Service** in `fastapi_app/`  
   - Exposes `/predict`  
   - Logs every request/response to `logs/prediction_logs.json`

2. **Streamlit Monitoring Dashboard** in `monitoring/`  
   - Reads the original `IMDB_Dataset.csv` and your live JSON log  
   - Shows:
     1. **Data Drift** histogram of sentence lengths  
     2. **Target Drift** bar chart of positive vs. negative predictions  
     3. **Accuracy & Precision** (once you supply `true_sentiment`)  
     4. **Alert** if accuracy < 80%

---

## Quickstart

```bash
git clone https://github.com/wisemichael/assignment5-sentiment-monitoring.git
cd assignment5-sentiment-monitoring

# Build both containers
make build

# Start FastAPI + Streamlit (in one terminal)
make run

# In another terminal, run evaluation:
python evaluate.py --url http://localhost:8000/predict --file test.json

# View logs (optional)
make logs

### Evaluate your API

# In one line, loop through test.json and print accuracy:
python evaluate.py --url http://localhost:8000/predict --file test.json

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this movie!","true_sentiment":"positive"}' \
  http://localhost:8000/predict
