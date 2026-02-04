# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Random Forest and XGBoost classifiers.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

## 🔍 Overview

This project implements a fraud detection system for credit card transactions using machine learning algorithms. Given the highly imbalanced nature of fraud detection (fraud cases are rare compared to normal transactions), the project employs advanced ensemble methods and evaluates performance using multiple metrics beyond simple accuracy.

## 📊 Dataset

- **Source**: Synthetic credit card transaction dataset
- **Size**: 15,000 transactions
- **Features**: 30 features (V1-V28, Amount, Class)
  - V1-V28: Transaction features (anonymized for privacy)
  - Amount: Transaction amount
  - Class: 0 = Normal, 1 = Fraud
- **Distribution**: 
  - Normal transactions: 14,895 (99.3%)
  - Fraudulent transactions: 105 (0.7%)

**Important Note**: This dataset is highly imbalanced, which is realistic for fraud detection scenarios.

## ✨ Features

- **Data Loading**: Direct loading from GitHub repository
- **Exploratory Data Analysis**: Class distribution visualization
- **Data Preprocessing**: Label encoding and train-test split with stratification
- **Multiple Models**: Implementation of both Random Forest and XGBoost
- **Comprehensive Evaluation**: Multiple metrics including:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- **Confusion Matrix**: Visual representation of model performance
- **Feature Importance**: Analysis of which features contribute most to fraud detection

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
xgboost>=1.5.0
```

## 💻 Usage

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

## 🤖 Models

### 1. Random Forest Classifier
- **Algorithm**: Ensemble of decision trees
- **Parameters**:
  - n_estimators: 200
  - random_state: 42
  - n_jobs: -1 (use all CPU cores)
- **Advantages**:
  - Fast training
  - Good for text classification
  - Low memory usage

### 2. XGBoost Classifier
- **Algorithm**: Gradient boosting method
- **Parameters**:
  - n_estimators: 300
  - max_depth: 6
  - learning_rate: 0.1
  - subsample: 0.8
  - eval_metric: 'logloss'
  - n_jobs: -1
- **Advantages**:
  - Builds stronger trees iteratively
  - Often performs better on imbalanced data
  - Handles complex patterns well

## 📈 Results

### Random Forest Performance
```
Accuracy: ~99.3%
Precision: 0.0
Recall: 0.0
F1-Score: 0.0
ROC-AUC: 0.50
```

### XGBoost Performance
```
Accuracy: ~99.3%
Precision: 0.0
Recall: 0.0
F1-Score: 0.0
ROC-AUC: 0.39
```

**Note on Results**: The current results indicate that both models are predicting all transactions as normal (Class 0). This is a common issue with highly imbalanced datasets and suggests the need for:
- Class balancing techniques (SMOTE, undersampling, oversampling)
- Adjusting class weights
- Threshold tuning
- More sophisticated ensemble methods

## 📊 Visualizations

The project includes several visualizations:

1. **Class Distribution Bar Chart**: Shows the imbalance between normal and fraudulent transactions
2. **Confusion Matrix Heatmap**: Visualizes true positives, true negatives, false positives, and false negatives
3. **Feature Importance Plot**: Displays the top 10 most important features for fraud detection (XGBoost)

## 🛠 Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and metrics
- **XGBoost**: Gradient boosting framework

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb   # Main Jupyter notebook
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
└── data/
    └── creditcard_synthetic.csv        # Dataset (loaded from GitHub)
```

## 🎓 Key Learnings

### Why Accuracy Alone is Misleading
With 99.3% normal transactions, a model that predicts everything as "normal" achieves 99.3% accuracy but catches 0% of fraud! This is why we need:
- **Precision**: Quality of fraud predictions
- **Recall**: How many frauds we actually catch (most critical for fraud detection)
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall discriminative ability

### Handling Imbalanced Data
This project demonstrates the challenges of imbalanced datasets:
- Missing fraud (False Negative) = HIGH financial damage
- False alarms (False Positive) = Customer inconvenience
- Need for specialized techniques beyond standard algorithms

### Understanding the Confusion Matrix
- **TP (True Positive)**: Correctly detected fraud
- **FN (False Negative)**: Missed fraud (DANGEROUS - high financial loss)
- **FP (False Positive)**: Flagged normal as fraud (customer inconvenience)
- **TN (True Negative)**: Normal detected correctly

## 🔮 Future Improvements

1. **Class Balancing**:
   - Implement SMOTE (Synthetic Minority Over-sampling Technique)
   - Try undersampling majority class
   - Use class weights in models

2. **Advanced Techniques**:
   - Anomaly detection algorithms (Isolation Forest, One-Class SVM)
   - Deep learning approaches (Autoencoders, LSTM)
   - Ensemble of multiple models

3. **Feature Engineering**:
   - Time-based features
   - Transaction frequency patterns
   - Customer behavior profiling

4. **Model Optimization**:
   - Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
   - Cross-validation with stratified K-Fold
   - Threshold optimization for better recall

5. **Deployment**:
   - Create REST API for real-time predictions
   - Build web interface for fraud monitoring
   - Implement model monitoring and retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This project is for educational purposes only. The synthetic dataset does not represent real credit card transactions, and the models should not be used in production without proper validation, testing, and compliance with relevant regulations.

---

**Note**: This is an educational project demonstrating machine learning techniques for fraud detection. For production use, implement additional security measures, model monitoring, and compliance with financial regulations.

