import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load EMNIST letters
X, y = fetch_openml('EMNIST_Letters', version=1, as_frame=False)

# normalize pixels
X = X / 255.0

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# save model
joblib.dump(model, "handwriting_model.pkl")