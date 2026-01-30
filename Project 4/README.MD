# Random Forest Classifier for Breast Cancer Detection

A machine learning project that uses Random Forest algorithm to classify breast cancer tumors as malignant or benign using the Wisconsin Breast Cancer dataset.

## Overview

This project demonstrates the implementation of a Random Forest Classifier for binary classification of breast cancer diagnosis. The model achieves approximately 95.6% accuracy on the test set.

## Dataset

- **Source**: scikit-learn's `load_breast_cancer()` dataset
- **Features**: 30 numerical features computed from digitized images of breast mass
- **Samples**: 569 instances
- **Classes**: 
  - Malignant (0)
  - Benign (1)

## Requirements

```
numpy
pandas
scikit-learn
```

## Installation

```bash
pip install numpy pandas scikit-learn
```

## Project Structure

The notebook contains the following steps:

1. **Import Libraries**: Load necessary packages for data processing and modeling
2. **Load Dataset**: Import the breast cancer dataset from scikit-learn
3. **Train-Test Split**: Split data into 80% training and 20% testing sets with stratification
4. **Train Model**: Build a Random Forest with 100 trees
5. **Make Predictions**: Generate predictions on the test set
6. **Evaluate Performance**: Calculate accuracy, precision, recall, and F1-score
7. **Hyperparameter Tuning**: Use GridSearchCV to find optimal parameters
8. **Best Model Results**: Display the best performing model configuration

## Model Performance

### Initial Model (100 estimators, max_depth=None)
- **Accuracy**: 95.61%

### Confusion Matrix
```
[[39  3]
 [ 2 70]]
```

### Classification Report
```
              precision    recall  f1-score   support
   malignant       0.95      0.93      0.94        42
      benign       0.96      0.97      0.97        72
```

### Best Model Parameters (After GridSearchCV)
- **criterion**: entropy
- **max_depth**: None
- **n_estimators**: 50
- **Accuracy**: 95.61%

## Hyperparameter Tuning

GridSearchCV was used to test the following parameters:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [None, 5, 10]
- `criterion`: ['gini', 'entropy']

The search used 5-fold cross-validation to find the optimal combination.

## Key Concepts

### Random Forest
- Ensemble learning method using multiple decision trees
- Reduces overfitting through bootstrap aggregating (bagging)
- Makes predictions by majority voting from individual trees

### Model Parameters
- **n_estimators**: Number of trees in the forest
- **max_depth**: Maximum depth of each tree (None = fully expanded)
- **random_state**: Ensures reproducibility

## Usage

Run the notebook cells sequentially in Google Colab or Jupyter Notebook:

```python
# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
```

## Results Interpretation

- **True Positives (Benign)**: 70 cases correctly identified as benign
- **True Negatives (Malignant)**: 39 cases correctly identified as malignant
- **False Positives**: 3 malignant cases misclassified as benign
- **False Negatives**: 2 benign cases misclassified as malignant

The model shows strong performance with high precision and recall for both classes, making it suitable for medical screening applications.

## Future Improvements

- Feature importance analysis
- Cross-validation with different seeds
- Comparison with other algorithms (SVM, XGBoost, Neural Networks)
- Feature engineering and selection
- Ensemble methods combination

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Dataset provided by scikit-learn
- Wisconsin Breast Cancer Dataset (Original)
