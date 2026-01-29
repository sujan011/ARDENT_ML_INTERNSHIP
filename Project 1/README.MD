# Python Machine Learning Project

A collection of practical machine learning examples demonstrating data visualization, regression, and classification using popular Python libraries.

## 📋 Overview

This project contains hands-on examples of fundamental machine learning concepts implemented in Python. It covers data visualization with Matplotlib and machine learning models using scikit-learn, including linear regression and classification.

## 🚀 Features

- **Data Visualization**: Bar charts for academic performance analysis
- **Linear Regression**: Predictive model for student marks based on study hours
- **Classification**: Iris flower species classification using K-Nearest Neighbors
- **Interactive Predictions**: Real-time prediction capabilities

## 📦 Dependencies

```python
numpy
pandas
matplotlib
tensorflow
scikit-learn
```

## 🛠️ Installation

1. Install required packages:
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

2. For Jupyter Notebook environment:
```bash
pip install jupyter
```

## 📊 Project Components

### 1. Basic Data Visualization

**Subject Performance Analysis**
- Creates a bar chart showing average marks across different subjects
- Subjects: Maths, DBMS, OS, ML, Python
- Visual representation with grid lines and proper labeling

### 2. Linear Regression Model

**Hours Studied vs Marks Predictor**

**Dataset:**
- Features: Hours studied (1-8 hours)
- Target: Marks obtained (35-85)

**Model Details:**
- Algorithm: Linear Regression
- Slope (m): ~7.08
- Intercept (b): ~27.5

**Usage Example:**
```python
# Predict marks for 5 hours of study
my_hours = np.array([[5]])
predicted_marks = model.predict(my_hours)
print(f"Predicted marks: {predicted_marks[0]:.2f}")
```

### 3. Classification (Iris Dataset)

**Iris Flower Species Classification**
- Dataset: Famous Iris flower dataset
- Features: Sepal length, Sepal width (first 2 features used for visualization)
- Classes: 3 species of Iris flowers
- Algorithm: K-Nearest Neighbors (KNN)

## 📈 Visualizations

The project includes several visualizations:

1. **Bar Chart**: Average marks per subject with gridlines
2. **Scatter Plot**: Raw data visualization for study hours vs marks
3. **Regression Line**: Fitted line showing the linear relationship
4. **Classification Plot**: (Iris dataset visualization - in progress)

## 🎯 Key Learning Objectives

- Understanding basic Python libraries for data science
- Data visualization techniques using Matplotlib
- Implementing supervised learning algorithms
- Model training and prediction
- Feature engineering and data preprocessing

## 💡 How to Use

### Running the Regression Example

1. Load the data:
```python
hours = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
marks = np.array([35,40,50,55,65,70,75,85])
```

2. Train the model:
```python
model = LinearRegression()
model.fit(hours, marks)
```

3. Make predictions:
```python
prediction = model.predict([[5]])  # Predict for 5 hours
```

### Visualizing Results

```python
plt.figure(figsize=(6,4))
plt.scatter(hours, marks, label="Actual Data")
plt.plot(hours_new, pred_marks, label="Fitted Line", linewidth=2)
plt.title("Linear Regression: Hours vs Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
```

## 📝 Code Structure

```
project/
│
├── Data Import and Setup
│   ├── Import libraries (datetime, math, pandas, matplotlib, tensorflow, numpy)
│   └── Basic Python operations
│
├── Visualization Module
│   └── Bar chart creation for subject performance
│
├── Regression Module
│   ├── Data preparation
│   ├── Model training
│   ├── Visualization
│   └── Prediction interface
│
└── Classification Module
    ├── Iris dataset loading
    ├── Feature selection
    └── KNN classifier (in progress)
```

## 🔍 Model Performance

### Linear Regression Results

- **Slope**: 7.08 (marks increase per hour of study)
- **Intercept**: 27.5 (baseline marks)
- **Interpretation**: For every additional hour of study, marks increase by approximately 7 points

### Example Predictions

| Hours Studied | Predicted Marks |
|--------------|----------------|
| 1            | ~34.58         |
| 3            | ~48.75         |
| 5            | ~62.92         |
| 7            | ~77.08         |

## 🎓 Educational Value

This project is ideal for:
- **Beginners** learning Python and machine learning
- **Students** studying data science fundamentals
- **Educators** teaching ML concepts with practical examples
- **Practitioners** refreshing basic ML implementations

## 🤝 Contributing

Feel free to fork this project and add more examples or improve existing ones. Potential enhancements:
- Add more regression examples
- Complete the Iris classification with cross-validation
- Add model evaluation metrics (R², accuracy, confusion matrix)
- Include more visualization types
- Add polynomial regression examples

## 📄 License

This project is open-source and available for educational purposes.

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project is designed for educational purposes and demonstrates fundamental machine learning concepts. For production use, consider adding error handling, validation, and more robust data preprocessing.

## 🔗 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [NumPy Documentation](https://numpy.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

## 🏆 Acknowledgments

- Dataset sources: Scikit-learn built-in datasets
- Visualization techniques inspired by Matplotlib gallery examples
