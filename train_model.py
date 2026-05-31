import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/loan_data..csv")

# Features and Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Encode categorical columns
encoders = {}

for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/loan_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")