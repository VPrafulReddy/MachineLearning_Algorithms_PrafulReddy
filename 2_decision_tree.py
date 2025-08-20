# Decision Tree Classifier with Titanic Dataset

import pandas as pd
import seaborn as sns  # For loading Titanic dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset

# Option 1: Using seaborn's built-in Titanic dataset
df = sns.load_dataset('titanic')

# If you have a CSV file instead, use:
# df = pd.read_csv("titanic.csv")

print("First 5 rows of dataset:")
print(df.head())


# Step 2: Data preprocessing

# Select relevant columns (you can choose more features)
features = ['pclass', 'sex', 'age', 'fare', 'embarked']
target = 'survived'

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)         # Replace missing ages with median
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True) # Replace missing embarked with most common

# Convert categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Prepare feature matrix X and target y
X = df[features + ['sex_male', 'embarked_Q', 'embarked_S']].drop(columns=['sex', 'embarked'], errors='ignore')
y = df[target]


# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Create and train Decision Tree model

model = DecisionTreeClassifier(max_depth=4, random_state=42)  # max_depth to prevent overfitting
model.fit(X_train, y_train)

# Step 5: Make predictions

y_pred = model.predict(X_test)


# Step 6: Evaluate model

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 7: Predict for new passenger
# Example: 3rd class, male, age 22, fare 7.25, embarked from 'S'
sample_passenger = pd.DataFrame({
    'pclass': [3],
    'age': [22],
    'fare': [7.25],
    'sex_male': [1],
    'embarked_Q': [0],
    'embarked_S': [1]
})

sample_prediction = model.predict(sample_passenger)
print("\nSample Passenger Prediction:", "Survived" if sample_prediction[0] == 1 else "Did Not Survive")





