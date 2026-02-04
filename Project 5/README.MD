# Spam Detection Using Naive Bayes + TF-IDF

A machine learning project that classifies SMS messages as spam or ham (legitimate) using Naive Bayes classification and TF-IDF vectorization.

## Overview

This project implements a spam detection system that can accurately identify spam messages from legitimate ones. It uses the Multinomial Naive Bayes algorithm combined with TF-IDF (Term Frequency-Inverse Document Frequency) text vectorization to achieve high accuracy in spam classification.

## Features

- **Text Classification**: Classifies messages as spam or ham
- **TF-IDF Vectorization**: Converts text into meaningful numerical features
- **Naive Bayes Algorithm**: Fast and efficient classification
- **Model Evaluation**: Includes accuracy, precision, recall, and confusion matrix
- **Custom Input Testing**: Test the model with your own messages
- **Data Visualization**: Bar chart showing spam vs ham distribution

## Requirements

```
pandas
scikit-learn
matplotlib
```

## Installation

1. Clone this repository or download the notebook
2. Install required packages:
```bash
pip install pandas scikit-learn matplotlib
```

## Dataset

The project uses a spam dataset containing SMS messages labeled as either "ham" (legitimate) or "spam". The dataset is loaded from GitHub:
- URL: `https://raw.githubusercontent.com/programmer-sahil/Ardent_ML_Training/main/Project%205/spam_dataset.csv`
- Features: `label` (ham/spam) and `message` (text content)

## How It Works

### 1. Data Loading
The dataset is loaded using pandas from a GitHub repository.

### 2. Label Encoding
Text labels ('ham', 'spam') are converted to numerical values (0, 1) for machine learning processing.

### 3. Train/Test Split
- 80% of data used for training
- 20% of data used for testing
- Ensures model generalization

### 4. TF-IDF Vectorization
Converts text messages into numerical features by:
- Calculating term frequency (TF)
- Calculating inverse document frequency (IDF)
- Removing English stop words
- Weighting important words like "FREE", "OFFER", "WIN", "PRIZE"

### 5. Model Training
Uses Multinomial Naive Bayes classifier which:
- Works great for text classification
- Has fast training and prediction times
- Requires low memory usage

### 6. Evaluation
The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Quality of spam predictions
- **Recall**: How many spam messages were caught
- **Confusion Matrix**: Breakdown of True Positives, True Negatives, False Positives, False Negatives

## Usage

### Running the Notebook

1. Open the notebook in Google Colab or Jupyter
2. Run all cells sequentially
3. The model will train and display performance metrics

### Testing with Custom Messages

The notebook includes a section to test custom messages:

```python
sample_msgs = [
    "Congratulations! You have won a FREE prize. Call now!",
    "Hey bro, are we still meeting tomorrow?"
]

sample_tfidf = tfidf.transform(sample_msgs)
print(model.predict(sample_tfidf))
# Output: [1 0] (spam, ham)
```

## Results

The model achieves strong performance metrics:
- **Accuracy**: ~87.5%
- Effectively identifies spam messages while minimizing false positives

Sample output:
```
Classification Report:
              precision    recall  f1-score   support

         ham       1.00      0.83      0.91         6
        spam       0.67      1.00      0.80         2
```

## Project Structure

```
spam-detection/
│
├── spam_detection.ipynb    # Main notebook with all code
├── README.md               # This file
└── spam_dataset.csv        # Dataset (loaded from GitHub)
```

## Why Naive Bayes?

Naive Bayes is chosen for this project because:
1. **Speed**: Very fast training and prediction
2. **Text Classification**: Particularly effective for text data
3. **Memory Efficient**: Low memory requirements
4. **Simplicity**: Easy to implement and understand
5. **Good Performance**: Despite its simplicity, achieves high accuracy

## Why TF-IDF?

TF-IDF is used because it:
1. **Weighs Important Words**: Gives higher weight to distinctive words
2. **Reduces Noise**: Filters out common, less meaningful words
3. **Numerical Representation**: Converts text to numbers ML algorithms can process
4. **Spam Indicators**: Highlights spam-indicating words like "FREE", "WIN", "URGENT"

## Visualization

The project includes a bar chart visualization showing the distribution of ham vs spam messages in the dataset, helping to understand data balance.

## Future Improvements

- Add more sophisticated text preprocessing (lemmatization, stemming)
- Experiment with other algorithms (SVM, Random Forest, Neural Networks)
- Implement cross-validation for more robust evaluation
- Add feature importance analysis
- Create a web interface for real-time spam detection
- Handle imbalanced datasets with techniques like SMOTE

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Dataset source: Ardent ML Training GitHub repository
- Built using scikit-learn machine learning library
- Inspired by common spam detection systems

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This is an educational project demonstrating basic spam detection techniques. For production use, consider more sophisticated approaches and additional security measures.
