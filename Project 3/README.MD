# Customer Segmentation using K-Means Clustering

A machine learning project that implements K-Means clustering to segment customers based on their annual income and spending patterns. This unsupervised learning approach helps identify distinct customer groups for targeted marketing strategies.

## 📋 Project Overview

This project demonstrates the application of K-Means clustering for customer segmentation analysis. By analyzing customer behavior patterns (income vs. spending score), the model identifies three distinct customer segments, enabling businesses to tailor their marketing strategies to different customer groups.

## 🎯 Key Features

- **Unsupervised Learning**: K-Means clustering algorithm
- **Customer Segmentation**: Identifies 3 distinct customer groups
- **Data Visualization**: Before and after clustering comparisons
- **Centroid Analysis**: Visual representation of cluster centers
- **Synthetic Dataset**: Simulated customer data for demonstration

## 📊 Dataset Information

**Source**: Synthetic data generated using NumPy's random normal distribution

**Statistics**:
- Total Samples: 150 customers
- Features: 2 (Annual Income, Spending Score)
- Clusters: 3
- Data Distribution:
  - Cluster 1: Low income, low spending (50 samples)
  - Cluster 2: Mid income, mid spending (50 samples)
  - Cluster 3: High income, medium spending (50 samples)

**Feature Descriptions**:
- **Annual Income**: Scaled representation of customer's yearly income
- **Spending Score**: Customer's propensity to spend (0-100 scale)

## 🛠️ Technologies Used

```python
- Python 3.x
- NumPy 1.x
- Matplotlib 3.x
- scikit-learn (sklearn)
```

## 📦 Installation

1. Install required packages:
```bash
pip install numpy matplotlib scikit-learn
```

2. For Jupyter Notebook:
```bash
pip install jupyter
```

## 🚀 Usage

### 1. Generate Synthetic Customer Data

```python
import numpy as np

np.random.seed(42)

# Create customer dataset: [Annual_Income, Spending_Score]
customer_data = np.vstack([
    np.random.normal(loc=[30,30], scale=[5,5], size=(50,2)),  # Low income, low spend
    np.random.normal(loc=[60,60], scale=[5,5], size=(50,2)),  # Mid income, mid spend
    np.random.normal(loc=[90,40], scale=[5,5], size=(50,2)),  # High income, medium spend
])
```

### 2. Apply K-Means Clustering

```python
from sklearn.cluster import KMeans

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(customer_data)

labels = kmeans.labels_           # Cluster assignments
centers = kmeans.cluster_centers_  # Cluster centroids
```

### 3. Visualize Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

# Plot each cluster
for cluster_id in range(k):
    cluster_data = customer_data[labels == cluster_id]
    plt.scatter(
        cluster_data[:,0], 
        cluster_data[:,1],
        label=f"Cluster {cluster_id}"
    )

# Plot centroids
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    edgecolor="black",
    marker="X",
    s=200,
    linewidth=2,
    label="Centroids"
)

plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score")
plt.title("Customer Segments using K-Means")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
```

## 📈 Model Results

### Cluster Characteristics

| Cluster | Description | Size | Income Range | Spending Pattern |
|---------|-------------|------|--------------|------------------|
| 0 | Low Income, Low Spenders | ~50 | 20-40 | 20-40 |
| 1 | Mid Income, Mid Spenders | ~50 | 50-70 | 50-70 |
| 2 | High Income, Medium Spenders | ~50 | 80-100 | 30-50 |

### Cluster Centers (Centroids)

The algorithm identified the following centroids:
- **Cluster 0**: (~30, ~30) - Budget-conscious segment
- **Cluster 1**: (~60, ~60) - Balanced segment
- **Cluster 2**: (~90, ~40) - High-earner conservative spenders

### Visual Analysis

**Before Clustering**:
- Data points appear as a mixed distribution
- No clear pattern visible
- Difficult to identify customer groups

**After Clustering**:
- Three distinct groups clearly visible
- Cluster centers marked with large X markers
- Color-coded for easy identification
- Clear separation between segments

## 🔍 Key Insights

### Customer Segments

1. **Budget-Conscious Customers** (Cluster 0)
   - Lower income bracket
   - Conservative spending habits
   - Marketing Strategy: Value-based promotions, discounts

2. **Balanced Customers** (Cluster 1)
   - Middle income bracket
   - Moderate spending aligned with income
   - Marketing Strategy: Quality-value balance, loyalty programs

3. **High-Income Conservative Spenders** (Cluster 2)
   - High income bracket
   - Moderate spending despite higher earnings
   - Marketing Strategy: Premium products, exclusive offers

## 🧮 Algorithm Details

### K-Means Clustering

**How it Works**:
1. Initialize k cluster centroids randomly
2. Assign each data point to nearest centroid
3. Recalculate centroids based on assigned points
4. Repeat steps 2-3 until convergence

**Key Parameters**:
- `n_clusters=3`: Number of customer segments to identify
- `random_state=42`: Ensures reproducible results
- `n_init=10`: Number of times algorithm runs with different centroid seeds

**Distance Metric**: Euclidean distance

### Why K-Means?

- **Simplicity**: Easy to understand and implement
- **Efficiency**: Fast computation, suitable for large datasets
- **Scalability**: Works well with various dataset sizes
- **Interpretability**: Clear cluster assignments and centroids

## 📊 Visualizations

### 1. Raw Customer Distribution
- Scatter plot showing all customers before clustering
- No color coding or segmentation
- Grid overlay for easier reading

### 2. Clustered Customer Segments
- Color-coded clusters (Cluster 0, 1, 2)
- Large X markers indicating centroids
- Legend for cluster identification
- Grid and axis labels for context

## 💡 Business Applications

### Marketing Strategies

1. **Targeted Campaigns**
   - Customize marketing messages for each segment
   - Optimize advertising spend
   - Improve conversion rates

2. **Product Development**
   - Design products for specific segments
   - Adjust pricing strategies
   - Create segment-specific features

3. **Customer Retention**
   - Identify high-value segments
   - Develop segment-specific loyalty programs
   - Predict customer churn by segment

### Use Cases

- **Retail**: Store layout optimization, inventory management
- **E-commerce**: Personalized recommendations, dynamic pricing
- **Banking**: Custom financial products, credit limit optimization
- **Telecommunications**: Plan recommendations, upselling strategies

## 🔧 Code Structure

```
Customer Segmentation/
│
├── Data Generation
│   ├── Generate synthetic customer data
│   └── Create three distinct customer groups
│
├── Visualization - Before Clustering
│   └── Raw scatter plot of all customers
│
├── K-Means Clustering
│   ├── Initialize KMeans (k=3)
│   ├── Fit model to customer data
│   └── Extract labels and centroids
│
└── Visualization - After Clustering
    ├── Color-coded cluster plots
    ├── Centroid markers
    └── Legend and labels
```

## 🚀 Future Improvements

1. **Optimal k Selection**
   - Elbow method implementation
   - Silhouette score analysis
   - Gap statistic evaluation

2. **Feature Engineering**
   - Add more customer features (age, gender, purchase frequency)
   - Feature scaling and normalization
   - Dimensionality reduction (PCA)

3. **Advanced Clustering**
   - Hierarchical clustering
   - DBSCAN for density-based clustering
   - Gaussian Mixture Models

4. **Real Data Integration**
   - Connect to actual customer databases
   - Handle missing values and outliers
   - Time-series analysis for behavior changes

5. **Interactive Visualization**
   - Plotly for interactive plots
   - Dashboard creation (Streamlit/Dash)
   - 3D visualization for additional features

6. **Model Evaluation**
   - Silhouette coefficient
   - Davies-Bouldin index
   - Calinski-Harabasz index

## 📝 Results Summary

- ✅ Successfully segments customers into 3 distinct groups
- ✅ Clear visual separation between clusters
- ✅ Identifiable centroids for each segment
- ✅ Actionable insights for marketing strategies
- ✅ Scalable approach for larger datasets

## ⚠️ Limitations

1. **Predefined k**: Requires manual selection of cluster count
2. **Sensitive to Outliers**: Extreme values can skew results
3. **Assumes Spherical Clusters**: May not capture complex shapes
4. **Initial Centroid Dependency**: Different starting points may yield different results
5. **Feature Scaling**: Assumes features are on similar scales

## 📚 Mathematical Foundation

### Euclidean Distance

Distance between points (x₁, y₁) and (x₂, y₂):

```
d = √[(x₂-x₁)² + (y₂-y₁)²]
```

### Objective Function

K-Means minimizes within-cluster sum of squares:

```
J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

Where:
- k = number of clusters
- Cᵢ = cluster i
- μᵢ = centroid of cluster i
- x = data point

## 🤝 Contributing

Potential improvements:
- Implement elbow method for optimal k
- Add more customer features
- Create interactive dashboard
- Integrate with real customer data
- Add cluster profiling statistics

## 📄 License

This project is available for educational and commercial purposes.

## 👏 Acknowledgments

- Dataset: Synthetic data generated for demonstration
- Libraries: NumPy, Matplotlib, scikit-learn
- Algorithm: K-Means clustering (MacQueen, 1967)

---

**Note**: This is a demonstration project using synthetic data. For production use with real customer data, ensure compliance with data privacy regulations (GDPR, CCPA, etc.) and implement proper data anonymization techniques.

## 🔗 Additional Resources

- [K-Means Clustering - scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Customer Segmentation Guide](https://en.wikipedia.org/wiki/Market_segmentation)
- [Clustering Validation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

## 📊 Sample Output

The clustering algorithm produces:
- **Visual Segmentation**: Clear grouping of customers
- **Centroid Locations**: Center points for each segment
- **Cluster Labels**: Assignment of each customer to a cluster
- **Actionable Insights**: Marketing strategies per segment

## 🎓 Educational Value

This project demonstrates:
- **Unsupervised Learning**: Learning patterns without labeled data
- **Clustering Algorithms**: Group similar data points
- **Data Visualization**: Effective communication of results
- **Business Analytics**: Practical application of ML
- **Scientific Computing**: NumPy for efficient data operations
