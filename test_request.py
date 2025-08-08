import requests

data = {
    "text": "This movie was excellent!",
    "true_sentiment": "positive"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
