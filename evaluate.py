import requests
import pandas as pd
import time
import sys

# System configuration and paths.
DATA_PATH = "IMDB_Dataset.csv"
API_URL = "http://localhost:8000/predict"
CHUNK_SIZE = 100

def send_prediction_request(text, sentiment):
    """Sends a single POST request to the prediction API."""
    try:
        payload = {
            "text": text,
            "true_sentiment": sentiment
        }
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (trouble shooting).
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred with the request: {e}")
        return None

def main():
    """Reads test data, sends it to the API, and calculates final accuracy."""
    try:
        # Loads the test data.
        df = pd.read_csv(DATA_PATH)
        print(f"INFO: Loaded {len(df)} samples from {DATA_PATH}")

        correct_predictions = 0
        total_predictions = 0
        
        # Iterates over the DataFrame in chunks.
        for i in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[i:i + CHUNK_SIZE]
            print(f"INFO: Sending chunk {i // CHUNK_SIZE + 1}...")

            for _, row in chunk.iterrows():
                review_text = row['review']
                true_sentiment = row['sentiment']
                
                # Sends the request.
                response_data = send_prediction_request(review_text, true_sentiment)
                
                if response_data:
                    predicted_sentiment = response_data["predicted_sentiment"]

                    
                    if predicted_sentiment == true_sentiment:
                        correct_predictions += 1
                    total_predictions += 1

            # Pause to prevent overwhelming the FastAPI service. Chunked for run time.
            time.sleep(1)

        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            print("-" * 50)
            print(f"EVALUATION COMPLETE")
            print(f"Total requests sent: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Final Accuracy: {accuracy:.2f}")
            print("-" * 50)
        else:
            print("WARNING: No predictions were successfully logged.")

    except FileNotFoundError:
        print(f"ERROR: The file {DATA_PATH} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()