import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("real_time_traffic_dataset.csv")
print("✅ Dataset loaded.")

# Clean column names
df.columns = df.columns.str.strip()

# Show column names and types
print("\n🧾 Columns and Data Types BEFORE encoding:")
print(df.dtypes)

# Encode categorical columns
label_cols = ['Sentiment', 'Time_of_Day', 'Traffic_State', 'Server_Load']
for col in label_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Show column names and types after encoding
print("\n🧾 Columns and Data Types AFTER encoding:")
print(df.dtypes)

# Define features and target
if "Traffic_State" in df.columns:
    X = df.drop("Traffic_State", axis=1)
    y = df["Traffic_State"]
else:
    raise KeyError("❌ 'Traffic_State' column missing in dataframe!")

# Final check for any object columns
print("\n🧪 Any non-numeric columns in X:")
print(X.select_dtypes(include='object').columns.tolist())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaled data
pd.DataFrame(X_scaled, columns=X.columns).to_csv("X_scaled.csv", index=False)
pd.DataFrame(y, columns=["Traffic_State"]).to_csv("y.csv", index=False)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Preprocessing complete. Files saved: X_scaled.csv, y.csv, scaler.pkl")
