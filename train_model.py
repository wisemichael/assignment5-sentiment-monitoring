import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Loads and sample IMDB dataset
df = pd.read_csv(r"C:\Users\wisem\Desktop\ML_Ops\Assignment 5\IMDB_Dataset.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(10000)

# Prepares features and labels.
X = df["review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})

# Trains the model.
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)

# Saves the model with sanity check and ensures consistent numpy and scikit-learn versions.
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved with numpy 2.1.1 and scikit-learn 1.5.2")