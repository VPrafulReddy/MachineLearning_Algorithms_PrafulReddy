# K-Nearest Neighbors (KNN) Classification on Digits Dataset


# Step 1: Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Digits dataset
# Digits dataset contains 8x8 images of handwritten digits (0-9)
digits = datasets.load_digits()

# Features (X) → flattened pixel values
X = digits.data  # Shape: (1797, 64) → 1797 samples, 64 features
# Target (y) → actual digit labels
y = digits.target

print("Dataset Shape:", X.shape)
print("Target Classes:", set(y))
print("First Image Label:", y[0])
print("First Image Data (flattened):\n", X[0])

# Step 3: Split dataset into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create KNN model
# n_neighbors = 5 → Look at 5 nearest neighbors for classification
knn_model = KNeighborsClassifier(n_neighbors=5)

# Step 5: Train the model
knn_model.fit(X_train, y_train)

# Step 6: Make predictions on test data
y_pred = knn_model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Predict for a new handwritten digit
# Example: take one test sample and predict
sample_index = 0
sample_data = X_test[sample_index].reshape(1, -1)  # Reshape to (1, 64)
sample_prediction = knn_model.predict(sample_data)

print("\nSample Input Index:", sample_index)
print("Predicted Digit:", sample_prediction[0])
print("Actual Digit:", y_test[sample_index])




