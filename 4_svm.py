# Support Vector Machine (SVM) Classification on Breast Cancer Dataset
# ---------------------------------------------------------------------

# Step 1: Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Breast Cancer dataset
# The dataset contains features about tumors (size, texture, etc.)
# and labels: malignant (0) or benign (1)
cancer = datasets.load_breast_cancer()

# Create a DataFrame for easier handling
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

print("First 5 rows of dataset:")
print(df.head())

# Step 3: Prepare features (X) and target (y)
X = df.drop(columns=['target'])  # All features except target
y = df['target']                 # Target: 0 = malignant, 1 = benign

# Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create the SVM model
# kernel='linear' → linear decision boundary
# C=1.0 → regularization parameter
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Step 6: Train the model
svm_model.fit(X_train, y_train)

# Step 7: Make predictions on test set
y_pred = svm_model.predict(X_test)

# Step 8: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Step 9: Predict for a new tumor sample
# Example: take the first test sample
sample_data = X_test.iloc[0].values.reshape(1, -1)
sample_prediction = svm_model.predict(sample_data)

print("\nSample Input (first test sample):")
print(X_test.iloc[0])
print("Predicted Class:", cancer.target_names[sample_prediction[0]])
print("Actual Class:", cancer.target_names[y_test.iloc[0]])



   malignant       0.96      0.96      0.96        42
      benign       0.98      0.98      0.98        72

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

Sample Input (first test sample):
mean radius                  13.54
mean texture                 14.36
mean perimeter               87.46
mean area                   566.30
... (more features)
Predicted Class: benign
Actual Class: benign

