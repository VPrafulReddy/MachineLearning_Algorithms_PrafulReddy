# Logistic Regression with Iris Dataset
# --------------------------------------

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Step 2: Prepare features (X) and target (y)
X = df.iloc[:, :-1]
y = df['target']

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Make predictions on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Predict for new data
sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Example measurements
sample_prediction = model.predict(sample_data)
print("Sample Input:", sample_data)
print("Predicted Class:", iris.target_names[sample_prediction[0]])





#Output:

Accuracy: 1.00

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00         9
  versicolor       1.00      1.00      1.00        13
   virginica       1.00      1.00      1.00         8

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

Sample Input: [[5.1, 3.5, 1.4, 0.2]]
Predicted Class: setosa


#The accuracy might not always be 1.00 — it depends on the train-test split.

#The prediction for the sample input [5.1, 3.5, 1.4, 0.2] will always be "setosa" because those measurements match that species closely.