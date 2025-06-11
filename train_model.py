import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
X = pd.read_csv('X_scaled.csv')
y = pd.read_csv('y.csv')

# Flatten y to 1D array
y = y.values.ravel()

# Split into train and test sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42,class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Save the trained model
joblib.dump(model, 'traffic_model.pkl')
print("✅ Model saved as 'traffic_model.pkl'")
