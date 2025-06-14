import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Sample pulse data
data = {
    'pulse_rate': [60, 65, 70, 75, 80, 85, 90, 95, 100, 55, 50, 110, 120, 130, 45, 40],
    'label':      [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1]  # 0=Normal, 1=Emergency
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split into features and target
X = df[['pulse_rate']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
#joblib.dump(model, 'pulse_emergency_model.pkl')
# Load model
#model = joblib.load('pulse_emergency_model.pkl')

# Predict new pulse input
def check_pulse(pulse_value):
    prediction = model.predict([[pulse_value]])
    return "Unusual (Emergency)" if prediction[0] == 1 else "Normal"
