import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("patient_data.csv")

# Split features and target
X = df.drop(columns=["Disease"])  # Features now include Body Temperature
y = df["Disease"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as 'disease_model.pkl' successfully!")