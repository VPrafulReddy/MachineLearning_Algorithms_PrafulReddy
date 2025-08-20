{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Decision Tree Classification with PCA\n",
        "\n",
        "**Assignment:** End-Term Practical Question 11\n",
        "\n",
        "**Dataset:** Lung_Cancer_dataset.csv\n",
        "\n",
        "**Tasks:**\n",
        "1. Preprocess the dataset (handle missing values, encoding, scaling, train-test split)\n",
        "2. Build a Decision Tree classifier and evaluate using accuracy, precision, recall, F1-score, and confusion matrix\n",
        "3. Apply PCA (retain components explaining \u226595% variance), retrain the Decision Tree, and compare results\n",
        "4. Discuss feature importance (baseline DT) and the effect of PCA on performance"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Import Libraries"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Data Preprocessing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Load the dataset\n",
        "data = pd.read_csv('Lung_Cancer_dataset.csv')\n",
        "print(\"Dataset shape:\", data.shape)\n",
        "print(\"\\nDataset info:\")\n",
        "print(data.info())\n",
        "print(\"\\nMissing values:\")\n",
        "print(data.isnull().sum())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Remove non-predictive columns and prepare features\n",
        "# Drop Name and Surname as they are not predictive features\n",
        "X = data.drop(['Name', 'Surname', 'Result'], axis=1)\n",
        "y = data['Result']\n",
        "\n",
        "print(\"Features:\", X.columns.tolist())\n",
        "print(\"Feature matrix shape:\", X.shape)\n",
        "print(\"Target distribution:\")\n",
        "print(y.value_counts())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Train-test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
        "\n",
        "print(\"Training set size:\", X_train.shape[0])\n",
        "print(\"Test set size:\", X_test.shape[0])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Feature scaling\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "print(\"Feature scaling completed\")\n",
        "print(\"Scaled training data shape:\", X_train_scaled.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Baseline Decision Tree Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Build Decision Tree classifier\n",
        "dt_baseline = DecisionTreeClassifier(random_state=42)\n",
        "dt_baseline.fit(X_train_scaled, y_train)\n",
        "\n",
        "# Make predictions\n",
        "y_pred_baseline = dt_baseline.predict(X_test_scaled)\n",
        "\n",
        "print(\"Baseline Decision Tree model trained\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Evaluate baseline model\n",
        "baseline_accuracy = accuracy_score(y_test, y_pred_baseline)\n",
        "baseline_precision = precision_score(y_test, y_pred_baseline)\n",
        "baseline_recall = recall_score(y_test, y_pred_baseline)\n",
        "baseline_f1 = f1_score(y_test, y_pred_baseline)\n",
        "baseline_cm = confusion_matrix(y_test, y_pred_baseline)\n",
        "\n",
        "print(\"Baseline Model Performance:\")\n",
        "print(f\"Accuracy: {baseline_accuracy:.4f}\")\n",
        "print(f\"Precision: {baseline_precision:.4f}\")\n",
        "print(f\"Recall: {baseline_recall:.4f}\")\n",
        "print(f\"F1-Score: {baseline_f1:.4f}\")\n",
        "print(\"\\nConfusion Matrix:\")\n",
        "print(baseline_cm)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Feature importance analysis\n",
        "feature_importance = pd.DataFrame({\n",
        "    'Feature': X.columns,\n",
        "    'Importance': dt_baseline.feature_importances_\n",
        "}).sort_values('Importance', ascending=False)\n",
        "\n",
        "print(\"Feature Importance (Baseline Model):\")\n",
        "print(feature_importance)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. PCA Implementation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Apply PCA to retain components explaining \u226595% variance\n",
        "pca_full = PCA()\n",
        "pca_full.fit(X_train_scaled)\n",
        "\n",
        "# Calculate cumulative explained variance\n",
        "cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)\n",
        "\n",
        "# Find number of components for \u226595% variance\n",
        "n_components = np.argmax(cumulative_variance >= 0.95) + 1\n",
        "\n",
        "print(\"PCA Analysis:\")\n",
        "print(f\"Explained variance ratios: {pca_full.explained_variance_ratio_}\")\n",
        "print(f\"Cumulative variance: {cumulative_variance}\")\n",
        "print(f\"Components needed for \u226595% variance: {n_components}\")\n",
        "print(f\"Actual variance retained: {cumulative_variance[n_components-1]:.4f}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Apply PCA transformation\n",
        "pca = PCA(n_components=n_components, random_state=42)\n",
        "X_train_pca = pca.fit_transform(X_train_scaled)\n",
        "X_test_pca = pca.transform(X_test_scaled)\n",
        "\n",
        "print(f\"Original dimensions: {X_train_scaled.shape[1]}\")\n",
        "print(f\"PCA dimensions: {X_train_pca.shape[1]}\")\n",
        "print(f\"Variance retained: {pca.explained_variance_ratio_.sum():.4f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Decision Tree with PCA"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Build Decision Tree with PCA features\n",
        "dt_pca = DecisionTreeClassifier(random_state=42)\n",
        "dt_pca.fit(X_train_pca, y_train)\n",
        "\n",
        "# Make predictions\n",
        "y_pred_pca = dt_pca.predict(X_test_pca)\n",
        "\n",
        "print(\"PCA Decision Tree model trained\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Evaluate PCA model\n",
        "pca_accuracy = accuracy_score(y_test, y_pred_pca)\n",
        "pca_precision = precision_score(y_test, y_pred_pca)\n",
        "pca_recall = recall_score(y_test, y_pred_pca)\n",
        "pca_f1 = f1_score(y_test, y_pred_pca)\n",
        "pca_cm = confusion_matrix(y_test, y_pred_pca)\n",
        "\n",
        "print(\"PCA Model Performance:\")\n",
        "print(f\"Accuracy: {pca_accuracy:.4f}\")\n",
        "print(f\"Precision: {pca_precision:.4f}\")\n",
        "print(f\"Recall: {pca_recall:.4f}\")\n",
        "print(f\"F1-Score: {pca_f1:.4f}\")\n",
        "print(\"\\nConfusion Matrix:\")\n",
        "print(pca_cm)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Model Comparison"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Compare baseline and PCA models\n",
        "comparison = pd.DataFrame({\n",
        "    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],\n",
        "    'Baseline': [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1],\n",
        "    'PCA': [pca_accuracy, pca_precision, pca_recall, pca_f1]\n",
        "})\n",
        "\n",
        "comparison['Difference'] = comparison['PCA'] - comparison['Baseline']\n",
        "\n",
        "print(\"Model Comparison:\")\n",
        "print(comparison)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 6. Discussion"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Discussion of results\n",
        "print(\"DISCUSSION:\")\n",
        "print(\"\\n1. Feature Importance (Baseline DT):\")\n",
        "for idx, row in feature_importance.iterrows():\n",
        "    if row['Importance'] > 0:\n",
        "        print(f\"   {row['Feature']}: {row['Importance']:.4f}\")\n",
        "\n",
        "print(f\"\\n2. PCA Effect on Performance:\")\n",
        "print(f\"   - Dimensions reduced from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}\")\n",
        "print(f\"   - Variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%\")\n",
        "\n",
        "accuracy_change = pca_accuracy - baseline_accuracy\n",
        "if accuracy_change > 0:\n",
        "    print(f\"   - Accuracy improved by {accuracy_change:.4f}\")\n",
        "elif accuracy_change < 0:\n",
        "    print(f\"   - Accuracy decreased by {abs(accuracy_change):.4f}\")\n",
        "else:\n",
        "    print(f\"   - Accuracy remained the same\")\n",
        "\n",
        "print(f\"\\n3. Summary:\")\n",
        "print(f\"   - Baseline model accuracy: {baseline_accuracy:.4f}\")\n",
        "print(f\"   - PCA model accuracy: {pca_accuracy:.4f}\")\n",
        "if pca_accuracy > baseline_accuracy:\n",
        "    print(f\"   - PCA improved performance while reducing dimensionality\")\n",
        "elif pca_accuracy < baseline_accuracy:\n",
        "    print(f\"   - PCA reduced performance but simplified the model\")\n",
        "else:\n",
        "    print(f\"   - PCA maintained performance with reduced complexity\")"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
