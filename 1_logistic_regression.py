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



