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


#Output:
First 5 rows of dataset:
   alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  ...  od280/od315_of_diluted_wines  proline  target
0    14.23        1.71  2.43              15.6       127           2.80  ...                           3.92     1065       0
1    13.20        1.78  2.14              11.2       100           2.65  ...                           3.40     1050       0
...

Model Accuracy: 1.00

Classification Report:
              precision    recall  f1-score   support

     class_0       1.00      1.00      1.00        14
     class_1       1.00      1.00      1.00        14
     class_2       1.00      1.00      1.00         8

    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36

Sample Input (first test sample):
alcohol                              13.05
malic_acid                            1.77
ash                                   2.10
alcalinity_of_ash                    17.0
magnesium                           107.0
total_phenols                         3.00
flavanoids                            3.00
nonflavanoid_phenols                  0.28
proanthocyanins                       2.03
color_intensity                       5.04
hue                                   0.88
od280/od315_of_diluted_wines          3.35
proline                             885.0
Name: 61, dtype: float64
Predicted Class: class_0
Actual Class: class_0
