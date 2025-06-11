import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("messaging_traffic_data.csv")
print("✅ Dataset loaded.")

# Clean column names
df.columns = df.columns.str.strip()

# Show column names and types
print("\n🧾 Columns and Data Types BEFORE encoding:")
print(df.dtypes)

# Show original Traffic_State counts BEFORE encoding
if 'Traffic state' in df.columns:
    print("\n📊 Original 'Traffic state' value counts before encoding:")
    print(df['Traffic state'].value_counts())
else:
    print("⚠️ 'Traffic state' column not found before encoding!")

# Fix column name for consistency
df.rename(columns={'Server load': 'Server load'}, inplace=True)

# Encode categorical columns
label_cols = ['User sentiment', 'Time of day', 'Traffic state', 'Server load']
for col in label_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# Show encoded label values
if 'Traffic state' in df.columns:
    print("\n🗂️ Encoded 'Traffic state' sample mapping:")
    print(df[['Traffic state']].drop_duplicates().sort_values('Traffic state'))
else:
    raise KeyError("❌ 'Traffic state' column missing in dataframe!")

# Show column names and types after encoding
print("\n🧾 Columns and Data Types AFTER encoding:")
print(df.dtypes)

# Define features and target
X = df.drop("Traffic state", axis=1)
y = df["Traffic state"]

# Final check for any object columns
print("\n🧪 Any non-numeric columns in X:")
print(X.select_dtypes(include='object').columns.tolist())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaled data
pd.DataFrame(X_scaled, columns=X.columns).to_csv("X_scaled.csv", index=False)
pd.DataFrame(y, columns=["Traffic state"]).to_csv("y.csv", index=False)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Preprocessing complete. Files saved: X_scaled.csv, y.csv, scaler.pkl")
