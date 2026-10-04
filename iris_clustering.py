import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Try importing from sklearn; if AppLocker/WDAC blocks C-extension DLLs, use pure NumPy/SciPy implementation
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    SKLEARN_SOURCE = "scikit-learn"
except Exception:
    SKLEARN_SOURCE = "NumPy/SciPy engine (AppLocker Fallback)"
    
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
            self.var_ = None
            
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = np.mean(X, axis=0)
            self.var_ = np.var(X, axis=0, ddof=0)
            self.scale_ = np.sqrt(self.var_)
            # Avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    class KMeans:
        def __init__(self, n_clusters=3, random_state=42, n_init=10, max_iter=300, tol=1e-4):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.n_init = n_init
            self.max_iter = max_iter
            self.tol = tol
            self.cluster_centers_ = None
            self.labels_ = None
            self.inertia_ = None
            self.n_iter_ = None

        def _kmpp_init(self, X, rng):
            n_samples, n_features = X.shape
            centers = np.empty((self.n_clusters, n_features))
            # Pick first center randomly
            first_idx = rng.randint(0, n_samples)
            centers[0] = X[first_idx]
            
            dist_sq = np.sum((X - centers[0])**2, axis=1)
            
            for c in range(1, self.n_clusters):
                prob = dist_sq / np.sum(dist_sq)
                cum_prob = np.cumsum(prob)
                r = rng.rand()
                next_idx = np.searchsorted(cum_prob, r)
                next_idx = min(next_idx, n_samples - 1)
                centers[c] = X[next_idx]
                
                if c < self.n_clusters - 1:
                    new_dist_sq = np.sum((X - centers[c])**2, axis=1)
                    dist_sq = np.minimum(dist_sq, new_dist_sq)
                    
            return centers

        def fit(self, X):
            X = np.asarray(X, dtype=float)
            n_samples, n_features = X.shape
            
            best_inertia = float('inf')
            best_centers = None
            best_labels = None
            best_n_iter = 0
            
            base_rng = np.random.RandomState(self.random_state if self.random_state is not None else 42)
            
            for init_idx in range(self.n_init):
                seed = base_rng.randint(0, 1000000)
                rng = np.random.RandomState(seed)
                
                centers = self._kmpp_init(X, rng)
                
                for it in range(self.max_iter):
                    # Distances to centers
                    dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                    labels = np.argmin(dists, axis=1)
                    
                    # Compute new centers
                    new_centers = np.zeros_like(centers)
                    for k in range(self.n_clusters):
                        mask = (labels == k)
                        if np.any(mask):
                            new_centers[k] = X[mask].mean(axis=0)
                        else:
                            new_centers[k] = X[rng.randint(0, n_samples)]
                            
                    center_shift = np.sum((new_centers - centers) ** 2)
                    centers = new_centers
                    
                    if center_shift <= self.tol:
                        break
                        
                dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                min_dists = np.min(dists, axis=1)
                inertia = np.sum(min_dists ** 2)
                
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_centers = centers
                    best_labels = labels
                    best_n_iter = it + 1
                    
            self.cluster_centers_ = best_centers
            self.labels_ = best_labels
            self.inertia_ = best_inertia
            self.n_iter_ = best_n_iter
            return self

        def fit_predict(self, X):
            self.fit(X)
            return self.labels_

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            dists = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
            return np.argmin(dists, axis=1)

    class PCA:
        def __init__(self, n_components=2, random_state=42):
            self.n_components = n_components
            self.random_state = random_state
            self.components_ = None
            self.explained_variance_ = None
            self.explained_variance_ratio_ = None
            self.mean_ = None

        def fit(self, X):
            X = np.asarray(X, dtype=float)
            n_samples, n_features = X.shape
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            
            # SVD decomposition
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Flip eigenvectors' sign to enforce deterministic output (scikit-learn convention)
            max_abs_cols = np.argmax(np.abs(Vt[:self.n_components, :]), axis=1)
            signs = np.sign(Vt[np.arange(self.n_components), max_abs_cols])
            signs[signs == 0] = 1.0
            Vt[:self.n_components, :] *= signs[:, np.newaxis]
            
            self.components_ = Vt[:self.n_components, :]
            
            explained_var = (S ** 2) / (n_samples - 1)
            total_var = np.sum(explained_var)
            
            self.explained_variance_ = explained_var[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            X_centered = X - self.mean_
            return X_centered @ self.components_.T

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    def silhouette_score(X, labels):
        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        n_samples = len(X)
        
        if len(unique_labels) <= 1 or len(unique_labels) >= n_samples:
            return 0.0
            
        # Pairwise distance matrix
        dist_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
        
        a = np.zeros(n_samples)
        b = np.zeros(n_samples)
        
        for i in range(n_samples):
            own_label = labels[i]
            own_cluster_mask = (labels == own_label)
            own_cluster_mask[i] = False
            
            if np.sum(own_cluster_mask) > 0:
                a[i] = np.mean(dist_matrix[i, own_cluster_mask])
            else:
                a[i] = 0.0
                
            min_b = float('inf')
            for other_label in unique_labels:
                if other_label == own_label:
                    continue
                other_cluster_mask = (labels == other_label)
                mean_dist_other = np.mean(dist_matrix[i, other_cluster_mask])
                if mean_dist_other < min_b:
                    min_b = mean_dist_other
            b[i] = min_b
            
        s = (b - a) / np.maximum(a, b)
        return float(np.mean(s))

    def adjusted_rand_score(labels_true, labels_pred):
        labels_true = np.asarray(labels_true)
        labels_pred = np.asarray(labels_pred)
        
        classes, true_indices = np.unique(labels_true, return_inverse=True)
        clusters, pred_indices = np.unique(labels_pred, return_inverse=True)
        
        n_classes = len(classes)
        n_clusters = len(clusters)
        
        contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
        for t, p in zip(true_indices, pred_indices):
            contingency[t, p] += 1
            
        sum_comb_c = np.sum(contingency * (contingency - 1)) // 2
        sum_comb_a = np.sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1)) // 2
        sum_comb_b = np.sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1)) // 2
        
        n_samples = len(labels_true)
        total_comb = (n_samples * (n_samples - 1)) // 2
        
        expected_comb = (sum_comb_a * sum_comb_b) / total_comb if total_comb > 0 else 0
        max_comb = (sum_comb_a + sum_comb_b) / 2.0
        
        denominator = max_comb - expected_comb
        if denominator == 0:
            return 1.0
        return float((sum_comb_c - expected_comb) / denominator)


def load_dataset(filepath=None):
    """
    Load the Iris dataset, handling filename variations and missing headers.
    Returns:
        df (pd.DataFrame): Cleaned Iris dataframe.
        dataset_name (str): Name of the dataset file loaded.
    """
    possible_files = ["Iris.csv", "iris.csv", "iris.data.csv", "iris.data"]
    
    if filepath and os.path.exists(filepath):
        target_path = filepath
    else:
        target_path = None
        for fname in possible_files:
            if os.path.exists(fname):
                target_path = fname
                break
                
    if target_path is None:
        raise FileNotFoundError("Could not find any Iris dataset CSV file in workspace.")
        
    dataset_name = os.path.basename(target_path)
    
    # Read sample to check header structure
    temp_df = pd.read_csv(target_path, nrows=5)
    
    std_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
    
    # Check if first row is numerical (indicating missing header)
    has_header = True
    try:
        float(temp_df.columns[0])
        has_header = False
    except ValueError:
        if "Iris-" in str(temp_df.columns[-1]) or temp_df.columns[0] == "5.1":
            has_header = False

    if not has_header:
        df = pd.read_csv(target_path, header=None, names=std_cols)
    else:
        df = pd.read_csv(target_path)
        
    # Drop completely empty rows
    df = df.dropna(how="all").copy()
    
    # Handle column name variations
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().replace("_", "").replace(" ", "").replace(".", "")
        if "sepallength" in col_lower or col == "0":
            col_mapping[col] = "SepalLengthCm"
        elif "sepalwidth" in col_lower or col == "1":
            col_mapping[col] = "SepalWidthCm"
        elif "petallength" in col_lower or col == "2":
            col_mapping[col] = "PetalLengthCm"
        elif "petalwidth" in col_lower or col == "3":
            col_mapping[col] = "PetalWidthCm"
        elif "species" in col_lower or "class" in col_lower or col == "4":
            col_mapping[col] = "Species"
            
    df = df.rename(columns=col_mapping)
    
    # Ignore Id / id column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["Species"] = df["Species"].astype(str).str.strip()

    # Create standard Iris.csv if it does not exist
    if not os.path.exists("Iris.csv"):
        df.to_csv("Iris.csv", index=False)

    return df, dataset_name


def explore_dataset(df):
    """
    Display complete dataset exploration details in terminal.
    """
    print("=" * 60)
    print("IRIS DATASET EXPLORATION")
    print("=" * 60)
    
    print(f"\nDataset Shape:\n{df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nFirst 5 Rows:")
    print(df.head())
    
    print("\nColumn Names:")
    print(list(df.columns))
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print(f"\nDuplicate Rows:\n{df.duplicated().sum()}")
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nClass/Species Distribution:")
    print(df["Species"].value_counts())
    
    feature_cols = [c for c in df.columns if c != "Species"]
    print(f"\nNumerical Features ({len(feature_cols)}):\n{feature_cols}")
    print(f"\nTrue Labels (1):\n['Species']")
    print("=" * 60)


def preprocess_data(df, feature_cols, target_col="Species"):
    """
    Extract numerical features, handle any missing values, and scale using StandardScaler.
    Returns:
        X_scaled (np.ndarray): Scaled feature matrix.
        scaler (StandardScaler): Fitted scaler.
        X_raw (pd.DataFrame): Raw numerical features.
        y_true (pd.Series): True species labels.
    """
    X_raw = df[feature_cols].copy()
    y_true = df[target_col].copy()
    
    if X_raw.isnull().sum().sum() > 0:
        X_raw = X_raw.fillna(X_raw.median())
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    return X_scaled, scaler, X_raw, y_true


def perform_elbow_method(X_scaled, max_k=10, output_img="elbow_method.png"):
    """
    Perform Elbow Method for K=1 through K=10, print inertias, and save plot.
    """
    print("\n" + "=" * 60)
    print("ELBOW METHOD FOR OPTIMAL K")
    print("=" * 60)
    
    inertias = []
    k_range = range(1, max_k + 1)
    
    print(f"{'K (Clusters)':<15} | {'Inertia':<20}")
    print("-" * 40)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        print(f"{k:<15} | {inertia:<20.4f}")
        
    # Create Elbow Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    plt.title("Elbow Method for Optimal K", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(f"\nElbow plot saved successfully as: {output_img}")
    print("=" * 60)
    
    return inertias


def train_kmeans(X_scaled, n_clusters=3, random_state=42, n_init=10):
    """
    Train K-Means clustering model with specified K.
    """
    print("\n" + "=" * 60)
    print(f"K-MEANS CLUSTERING (K={n_clusters})")
    print("=" * 60)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in counts.items():
        print(f"Cluster {cluster_id} count : {count}")
        
    print("=" * 60)
    return kmeans, cluster_labels


def apply_pca(X_scaled, cluster_labels, y_true, n_components=2):
    """
    Apply PCA dimensionality reduction to 2 components and create DataFrame.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame({
        "PC1": pca_features[:, 0],
        "PC2": pca_features[:, 1],
        "Cluster": cluster_labels,
        "True Species": y_true.values
    })
    
    exp_var_pc1 = pca.explained_variance_ratio_[0]
    exp_var_pc2 = pca.explained_variance_ratio_[1]
    total_exp_var = np.sum(pca.explained_variance_ratio_)
    
    print("\n" + "=" * 60)
    print("PCA RESULTS")
    print("=" * 60)
    print(f"PC1 Explained Variance : {exp_var_pc1:.4f} ({exp_var_pc1*100:.2f}%)")
    print(f"PC2 Explained Variance : {exp_var_pc2:.4f} ({exp_var_pc2*100:.2f}%)")
    print(f"Total Explained Variance : {total_exp_var:.4f} ({total_exp_var*100:.2f}%)")
    print("=" * 60)
    
    return pca, pca_df, exp_var_pc1, exp_var_pc2, total_exp_var


def create_visualizations(pca_df, output_img="iris_clusters_pca.png"):
    """
    Create professional scatter plot of PCA components colored by predicted K-Means cluster.
    """
    plt.figure(figsize=(10, 7))
    
    palette = sns.color_palette("Set1", n_colors=3)
    markers = ['o', 's', '^']
    
    unique_clusters = sorted(pca_df["Cluster"].unique())
    for idx, cluster_id in enumerate(unique_clusters):
        subset = pca_df[pca_df["Cluster"] == cluster_id]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Cluster {cluster_id}",
            color=palette[idx],
            marker=markers[idx % len(markers)],
            s=80,
            edgecolor='k',
            alpha=0.85
        )
        
    plt.title("Iris Flower Clusters using K-Means and PCA", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend(title="Predicted Cluster", fontsize=11, title_fontsize=12, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(f"\nCluster visualization saved successfully as: {output_img}")


def evaluate_clusters(X_scaled, cluster_labels, y_true, kmeans_model, output_csv="cluster_comparison.csv"):
    """
    Evaluate unsupervised cluster quality using Inertia, Silhouette Score, ARI, and Crosstab.
    """
    inertia = kmeans_model.inertia_
    silhouette = silhouette_score(X_scaled, cluster_labels)
    ari = adjusted_rand_score(y_true, cluster_labels)
    
    crosstab = pd.crosstab(
        y_true,
        cluster_labels,
        rownames=["Actual Species"],
        colnames=["Cluster"]
    )
    crosstab.to_csv(output_csv)
    
    print("\n" + "=" * 60)
    print("CLUSTERING EVALUATION")
    print("=" * 60)
    print(f"K-Means Inertia      : {inertia:.4f}")
    print(f"Silhouette Score     : {silhouette:.4f}")
    print(f"Adjusted Rand Index  : {ari:.4f}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("TRUE SPECIES VS PREDICTED CLUSTER COMPARISON")
    print("=" * 60)
    print(crosstab)
    print("-" * 60)
    print("NOTE: K-Means is an unsupervised algorithm; cluster numbers (0, 1, 2)")
    print("are arbitrary identifiers and do not have an intrinsic 1-to-1 label mapping.")
    print(f"Comparison matrix saved to: {output_csv}")
    print("=" * 60)
    
    return inertia, silhouette, ari, crosstab


def save_model(kmeans_model, scaler, model_path="kmeans_iris_model.joblib", scaler_path="iris_scaler.joblib"):
    """
    Save model and scaler using Joblib.
    """
    joblib.dump(kmeans_model, model_path)
    joblib.dump(scaler, scaler_path)
    print("\nModel saved successfully.")
    print("Scaler saved successfully.")


def load_and_verify_model(X_scaled, original_predictions, model_path="kmeans_iris_model.joblib", scaler_path="iris_scaler.joblib"):
    """
    Load saved model and scaler and verify prediction consistency.
    """
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    
    loaded_predictions = loaded_model.predict(X_scaled)
    
    is_identical = np.array_equal(original_predictions, loaded_predictions)
    
    print("Model loaded successfully.")
    if is_identical:
        print("Prediction verification successful.")
    else:
        print("WARNING: Prediction verification failed!")
        
    return loaded_model, loaded_scaler, is_identical


def main():
    # 1. Load Dataset
    df, dataset_name = load_dataset()
    
    # 2. Explore Dataset
    explore_dataset(df)
    
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    
    # 3. Preprocess Data
    X_scaled, scaler, X_raw, y_true = preprocess_data(df, feature_cols, target_col="Species")
    
    # 4. Elbow Method
    inertias = perform_elbow_method(X_scaled, max_k=10, output_img="elbow_method.png")
    
    # 5. K-Means Clustering (K=3)
    kmeans_model, cluster_labels = train_kmeans(X_scaled, n_clusters=3, random_state=42, n_init=10)
    
    # 6. PCA Dimensionality Reduction
    pca_model, pca_df, pc1_var, pc2_var, total_var = apply_pca(X_scaled, cluster_labels, y_true, n_components=2)
    
    # 7. Visualizations
    create_visualizations(pca_df, output_img="iris_clusters_pca.png")
    
    # 8. Cluster Evaluation & Comparison
    inertia, silhouette, ari, crosstab = evaluate_clusters(
        X_scaled, cluster_labels, y_true, kmeans_model, output_csv="cluster_comparison.csv"
    )
    
    # 9. Save Model
    save_model(kmeans_model, scaler, model_path="kmeans_iris_model.joblib", scaler_path="iris_scaler.joblib")
    
    # 10. Load and Verify Model
    loaded_model, loaded_scaler, verification_success = load_and_verify_model(
        X_scaled, cluster_labels, model_path="kmeans_iris_model.joblib", scaler_path="iris_scaler.joblib"
    )
    
    # 11. Final Summary Terminal Output (Section 19 Format)
    print("\n" + "=" * 60)
    print("IRIS FLOWER CLUSTERING")
    print("K-MEANS + PCA")
    print("=" * 60)
    print(f"Dataset Status       : SUCCESS ({dataset_name})")
    print(f"Dataset Samples      : {len(df)}")
    print(f"Features Used        : {len(feature_cols)}")
    print(f"Number of Clusters   : 3")
    print("-" * 60)
    print("PCA RESULTS")
    print("-" * 60)
    print(f"PC1 Explained Variance : {pc1_var:.4f}")
    print(f"PC2 Explained Variance : {pc2_var:.4f}")
    print(f"Total Explained        : {total_var:.4f}")
    print("-" * 60)
    print("CLUSTER EVALUATION")
    print("-" * 60)
    print(f"Inertia               : {inertia:.4f}")
    print(f"Silhouette Score      : {silhouette:.4f}")
    print(f"Adjusted Rand Index   : {ari:.4f}")
    print("-" * 60)
    print(f"Elbow Plot            : elbow_method.png")
    print(f"Cluster Plot          : iris_clusters_pca.png")
    print(f"Comparison File       : cluster_comparison.csv")
    print(f"K-Means Model         : kmeans_iris_model.joblib")
    print(f"Scaler                : iris_scaler.joblib")
    print("-" * 60)
    print(f"MODEL SAVE/LOAD STATUS : {'SUCCESS' if verification_success else 'FAILED'}")
    print(f"PROJECT STATUS         : COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
