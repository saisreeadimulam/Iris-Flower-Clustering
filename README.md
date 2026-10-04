# Iris Flower Clustering using K-Means and PCA

A professional end-to-end Unsupervised Machine Learning project created for Machine Learning & AI Internship (Week 3 Assignment).

---

## Project Overview

This project focuses on unsupervised pattern discovery in tabular data using **K-Means Clustering** and **Principal Component Analysis (PCA)**. Using the classic Iris dataset, we build a complete data pipeline that scales numerical features, determines the optimal number of clusters via the Elbow Method, performs cluster prediction for $K=3$, projects four-dimensional feature vectors into two principal components, evaluates cluster quality against ground truth labels, and serializes the trained model and scaler for deployment.

---

## Objective

The main objective is to discover natural clusters in flower measurements without using true species labels during training, and subsequently evaluate how well the unsupervised clusters align with true botanical species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`).

---

## Problem Statement

Botanical feature measurements (sepal length, sepal width, petal length, petal width) vary across flower species. In real-world data science applications, target class labels are rarely available. The task is to partition unlabeled Iris samples into $K=3$ distinct clusters using geometric distance metrics and evaluate the effectiveness of unsupervised feature scaling, clustering, and PCA reduction.

---

## Dataset

- **Dataset Name**: Iris Dataset (`Iris.csv` / `iris.data.csv`)
- **Total Samples**: 150
- **Total Columns**: 5 (4 numerical features + 1 target species string)
- **Missing Values**: 0
- **Duplicate Rows**: 3 (retained to match original dataset structure)

---

## Features

The model utilizes the four numerical measurements:

1. **SepalLengthCm**: Sepal length in centimeters (range: 4.3 cm to 7.9 cm, mean: 5.84 cm)
2. **SepalWidthCm**: Sepal width in centimeters (range: 2.0 cm to 4.4 cm, mean: 3.05 cm)
3. **PetalLengthCm**: Petal length in centimeters (range: 1.0 cm to 6.9 cm, mean: 3.76 cm)
4. **PetalWidthCm**: Petal width in centimeters (range: 0.1 cm to 2.5 cm, mean: 1.20 cm)

**True Target Label** (excluded during K-Means training):
- **Species**: `Iris-setosa` (50 samples), `Iris-versicolor` (50 samples), `Iris-virginica` (50 samples)

---

## Machine Learning Approach

1. **Exploratory Data Analysis (EDA)**: Inspecting shapes, data types, statistics, missing values, duplicates, and label distributions.
2. **Feature Scaling**: Standardizing 4 numerical features to zero mean and unit variance ($\mu=0, \sigma=1$).
3. **Hyperparameter Selection**: Testing $K=1$ through $K=10$ using the Elbow Method and inertia curves.
4. **Unsupervised Clustering**: Fitting `KMeans(n_clusters=3, random_state=42, n_init=10)` on scaled feature matrix.
5. **Dimensionality Reduction**: Applying `PCA(n_components=2)` to project 4D features to 2D components (`PC1`, `PC2`).
6. **Cluster Evaluation**: Evaluating cluster separation using Inertia, Silhouette Score, and Adjusted Rand Index (ARI).
7. **Model Serialization**: Saving `KMeans` model and `StandardScaler` using `joblib`.

---

## Data Preprocessing

Feature selection strictly isolates the four numerical columns (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`). ID/index columns and target class strings are excluded prior to model fitting to prevent data leakage.

---

## Feature Scaling

Feature scaling is critical for distance-based algorithms like K-Means. Because K-Means computes Euclidean distances between sample points and cluster centroids:
$$d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

If features have different scales (e.g., Sepal Length values around 7.0 vs. Petal Width values around 0.2), features with larger magnitudes dominate the distance calculations. Using `StandardScaler` transforms every feature $X_j$ to:
$$Z_j = \frac{X_j - \mu_j}{\sigma_j}$$

This ensures every feature contributes equally to cluster formation.

---

## Elbow Method

The Elbow Method evaluates total within-cluster sum of squares (**Inertia**) across varying cluster numbers $K \in [1, 10]$.

```
============================================================
ELBOW METHOD RESULTS
============================================================
K (Clusters)    | Inertia             
----------------------------------------
1               | 600.0000            
2               | 223.7320            
3               | 140.9658            
4               | 114.6155            
5               | 91.2954             
6               | 80.2306             
7               | 70.8813             
8               | 63.1536             
9               | 55.3994             
10              | 47.3839             
============================================================
```

The sharpest inflection point ("elbow") occurs between $K=2$ and $K=3$, confirming $K=3$ as the optimal cluster choice. The generated plot is saved to `elbow_method.png`.

---

## K-Means Clustering

Using $K=3$, the final K-Means model produces the following cluster distribution across 150 Iris samples:

- **Cluster 0**: 53 samples
- **Cluster 1**: 50 samples
- **Cluster 2**: 47 samples

---

## PCA Dimensionality Reduction

To visualize 4-dimensional feature space in 2D, Principal Component Analysis (`PCA(n_components=2)`) was fitted on scaled features.

```
============================================================
PCA RESULTS
============================================================
PC1 Explained Variance : 0.7277 (72.77%)
PC2 Explained Variance : 0.2303 (23.03%)
Total Explained Variance : 0.9580 (95.80%)
============================================================
```

Together, `PC1` and `PC2` preserve **95.80%** of the original 4-dimensional variance.

---

## Cluster Visualization

The 2D projection scatter plot displays the three discovered K-Means clusters along `PC1` and `PC2`. 

- `Cluster 1` (Setosa) forms a distinct, linearly separable cluster on the far right of `PC1`.
- `Cluster 0` and `Cluster 2` (Versicolor and Virginica) occupy adjacent regions along `PC1` with slight overlap around boundary points.

The generated scatter plot is saved to `iris_clusters_pca.png`.

---

## Cluster vs True Label Comparison

A contingency cross-tabulation table compares ground truth species with predicted K-Means clusters:

```
============================================================
TRUE SPECIES VS PREDICTED CLUSTER COMPARISON
============================================================
Cluster           0   1   2
Actual Species             
Iris-setosa       0  50   0
Iris-versicolor  39   0  11
Iris-virginica   14   0  36
============================================================
```

> **Important Concept**: K-Means is strictly unsupervised; cluster identifiers (0, 1, 2) are arbitrary cluster numbers assigned during random initialization. Cluster 1 corresponds to `Iris-setosa` (100% purity), Cluster 0 contains 39 `Iris-versicolor` and 14 `Iris-virginica`, while Cluster 2 contains 36 `Iris-virginica` and 11 `Iris-versicolor`.

Saved output: `cluster_comparison.csv`.

---

## Model Evaluation

Unsupervised model performance was evaluated using intrinsic geometric metric (Inertia, Silhouette Score) and external ground-truth comparison metric (Adjusted Rand Index).

### Inertia
- **Value**: **140.9658**
- **Meaning**: Total squared distance of samples to their assigned cluster center.

### Silhouette Score
- **Value**: **0.4590**
- **Meaning**: Quantifies how similar an object is to its own cluster compared to other clusters (range: -1 to +1). A score of 0.4590 indicates strong cluster cohesion and clear separation for Setosa, with moderate density overlap between Versicolor and Virginica.

### Adjusted Rand Index
- **Value**: **0.6201**
- **Meaning**: Measures similarity between true species labels and predicted cluster assignments, corrected for chance (range: -1 to +1). An ARI of **0.6201** indicates substantial agreement between unsupervised K-Means clusters and botanical taxonomies.

---

## Model Saving and Loading

The trained K-Means model and fitted `StandardScaler` are serialized for production reuse:

- **Model File**: `kmeans_iris_model.joblib`
- **Scaler File**: `iris_scaler.joblib`

Upon reloading both binaries from disk, prediction consistency was verified:
`np.array_equal(original_predictions, loaded_predictions)` returned `True`.

---

## Technologies Used

- **Python 3.11**
- **Pandas**: Data manipulation and cross-tabulation
- **NumPy**: Matrix computation and vector operations
- **Matplotlib**: Graph rendering and image export
- **Seaborn**: High-level statistical visualization
- **Scikit-Learn / SciPy**: Scaling, K-Means, PCA, metrics calculation
- **Joblib**: Model serialization and persistence

---

## Project Structure

```
Iris-Flower-Clustering/
│
├── Iris.csv                   # Clean dataset with column headers
├── iris.data.csv              # Original Iris dataset file
├── iris_clustering.py         # Main modular Python execution script
├── requirements.txt           # Python dependency requirements
├── README.md                  # Detailed project documentation & real results
├── .gitignore                 # Git ignore file for environment and caches
│
├── elbow_method.png           # Generated Elbow Method inertia graph
├── iris_clusters_pca.png      # Generated 2D PCA cluster scatter plot
├── cluster_comparison.csv     # Contingency table (Species vs Clusters)
│
├── kmeans_iris_model.joblib   # Saved K-Means clustering model
└── iris_scaler.joblib         # Saved StandardScaler object
```

---

## Installation

1. **Clone or Navigate to the Workspace**:
   ```bash
   cd c:\Users\DELL\Downloads\Iris-Flower-Clustering
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Required Packages**:
   ```bash
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

---

## How to Run

To run the full end-to-end clustering pipeline, execute:

```bash
.venv\Scripts\python.exe iris_clustering.py
```

---

## Results

```
============================================================
IRIS FLOWER CLUSTERING
K-MEANS + PCA
============================================================
Dataset Status       : SUCCESS (iris.data.csv)
Dataset Samples      : 150
Features Used        : 4
Number of Clusters   : 3
------------------------------------------------------------
PCA RESULTS
------------------------------------------------------------
PC1 Explained Variance : 0.7277
PC2 Explained Variance : 0.2303
Total Explained        : 0.9580
------------------------------------------------------------
CLUSTER EVALUATION
------------------------------------------------------------
Inertia               : 140.9658
Silhouette Score      : 0.4590
Adjusted Rand Index   : 0.6201
------------------------------------------------------------
Elbow Plot            : elbow_method.png
Cluster Plot          : iris_clusters_pca.png
Comparison File       : cluster_comparison.csv
K-Means Model         : kmeans_iris_model.joblib
Scaler                : iris_scaler.joblib
------------------------------------------------------------
MODEL SAVE/LOAD STATUS : SUCCESS
PROJECT STATUS         : COMPLETED SUCCESSFULLY
============================================================
```

---

## Key Learnings

1. **Feature Scaling is Essential**: Unscaled features bias distance metrics toward features with larger absolute numerical ranges.
2. **Unsupervised Identification**: K-Means successfully separated `Iris-setosa` completely (50/50 samples in Cluster 1) without seeing species labels.
3. **PCA Preserves Variance**: Reducing 4 dimensions to 2 retaining 95.80% variance allows 2D visual diagnosis of cluster boundaries.
4. **Arbitrary Cluster Indexing**: Cluster labels (0, 1, 2) assigned by K-Means are arbitrary numerical IDs, requiring cross-tabulation and ARI for proper evaluation against true classes.

---

## Future Improvements

- Explore non-linear dimensionality reduction algorithms like t-SNE or UMAP.
- Test hierarchical agglomerative clustering and DBSCAN for non-spherical cluster boundaries.
- Build an interactive Streamlit or Web dashboard for real-time sample cluster prediction.

---

## Conclusion

The Iris Flower Clustering project successfully satisfies all Week 3 Machine Learning internship objectives. It demonstrates high-quality modular code, rigorous feature preprocessing, Elbow Method determination, PCA visualization, metrics evaluation, and complete model serialization.
