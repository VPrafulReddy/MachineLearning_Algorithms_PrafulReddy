# Random Forest Classification on Wine Dataset
# ---------------------------------------------

# Step 1: Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Wine dataset
# The dataset contains chemical analysis results of wines grown in the same region in Italy
# and classifies them into 3 categories based on their properties.
wine = datasets.load_wine()

# Convert to pandas DataFrame for easier handling
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print("First 5 rows of dataset:")
print(df.head())

# Step 3: Prepare features (X) and target (y)
X = df.drop(columns=['target'])  # All columns except target
y = df['target']                 # Wine category labels (0, 1, 2)

# Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create Random Forest model
# n_estimators=100 → number of decision trees in the forest
# random_state=42 → ensures reproducibility
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 6: Train the model
rf_model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = rf_model.predict(X_test)

# Step 8: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Step 9: Predict for a new wine sample
# Example: use the first test sample
sample_data = X_test.iloc[0].values.reshape(1, -1)
sample_prediction = rf_model.predict(sample_data)

print("\nSample Input (first test sample):")
print(X_test.iloc[0])
print("Predicted Class:", wine.target_names[sample_prediction[0]])
print("Actual Class:", wine.target_names[y_test.iloc[0]])


