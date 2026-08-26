# ============================================================
# 🚀 DAY 35/60 — UNSUPERVISED LEARNING
# Topic: K-Means Clustering — End-to-End
# ============================================================
#
# Learning flow:
# 1. Understand the problem
# 2. Generate unlabeled data
# 3. Visualize the data
# 4. Scale features
# 5. Find optimal K using Elbow Method
# 6. Validate K using Silhouette Score
# 7. Train final K-Means model
# 8. Analyze clusters and centroids
# 9. Visualize final clustering
# 10. Print revision notes
# 11. Generate LinkedIn learning post
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# 2. CREATE AN UNLABELED DATASET
# ------------------------------------------------------------
#
# Important:
# In unsupervised learning, we do NOT provide target labels
# to the model.
#
# make_blobs internally creates labels, but we deliberately
# ignore them using "_".
#
# The model will have to discover the structure itself.
# ------------------------------------------------------------

X, _ = make_blobs(
    n_samples=600,
    centers=4,
    cluster_std=1.5,
    n_features=2,
    random_state=42
)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Number of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")


# Convert to DataFrame for easier inspection

df = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"]
)

print("\nFirst 5 observations:")
print(df.head())


# ------------------------------------------------------------
# 3. VISUALIZE RAW DATA
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Feature_1"],
    df["Feature_2"],
    alpha=0.7
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Raw Unlabeled Data")

plt.grid(alpha=0.2)
plt.show()


# ------------------------------------------------------------
# 4. FEATURE SCALING
# ------------------------------------------------------------
#
# K-Means is distance-based.
#
# If one feature has a much larger numerical scale than
# another feature, it can dominate the distance calculation.
#
# StandardScaler transforms features approximately into:
#
# mean = 0
# standard deviation = 1
#
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

print("Original feature means:")
print(X.mean(axis=0))

print("\nScaled feature means:")
print(X_scaled.mean(axis=0))

print("\nScaled feature standard deviations:")
print(X_scaled.std(axis=0))


# ------------------------------------------------------------
# 5. ELBOW METHOD
# ------------------------------------------------------------
#
# We don't want to blindly choose:
#
# KMeans(n_clusters=4)
#
# Instead, we test multiple K values.
#
# Inertia = Within-Cluster Sum of Squares (WCSS)
#
# Lower inertia means points are closer to their centroids.
#
# But inertia always tends to decrease as K increases.
# Therefore, we look for the "elbow".
# ------------------------------------------------------------

k_values = range(1, 11)

inertias = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)


# Plot elbow curve

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia / WCSS")
plt.title("Elbow Method")

plt.xticks(list(k_values))
plt.grid(alpha=0.2)

plt.show()


# ------------------------------------------------------------
# 6. SILHOUETTE ANALYSIS
# ------------------------------------------------------------
#
# Elbow method is useful, but sometimes the elbow is ambiguous.
#
# Silhouette Score measures:
#
# - How close a point is to its own cluster
# - How far it is from the nearest other cluster
#
# Range:
#
#       -1  --------  0  --------  +1
#       Bad       Overlap       Good
#
# Higher is generally better.
# ------------------------------------------------------------

silhouette_scores = []

# Silhouette score requires at least 2 clusters

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    cluster_labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        cluster_labels
    )

    silhouette_scores.append(score)


# Display silhouette scores

silhouette_results = pd.DataFrame(
    {
        "K": list(range(2, 11)),
        "Silhouette_Score": silhouette_scores
    }
)

print("\n" + "=" * 60)
print("SILHOUETTE ANALYSIS")
print("=" * 60)

print(silhouette_results.to_string(index=False))


# Plot silhouette scores

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")

plt.xticks(range(2, 11))
plt.grid(alpha=0.2)

plt.show()


# ------------------------------------------------------------
# 7. SELECT K
# ------------------------------------------------------------
#
# For this controlled synthetic dataset, we know the data was
# generated around 4 centers.
#
# In a real project, we would NOT have that information.
#
# We would combine:
#
# - Elbow method
# - Silhouette score
# - Domain knowledge
# - Business requirements
#
# Here we use K = 4 for demonstration.
# ------------------------------------------------------------

optimal_k = 4

print("\n" + "=" * 60)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 60)

print(f"Selected K = {optimal_k}")


# ------------------------------------------------------------
# 8. TRAIN FINAL K-MEANS MODEL
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)


# ------------------------------------------------------------
# 9. STORE CLUSTER ASSIGNMENTS
# ------------------------------------------------------------

df["Cluster"] = cluster_labels

print("\n" + "=" * 60)
print("CLUSTER ASSIGNMENTS")
print("=" * 60)

print(df.head(10))


# ------------------------------------------------------------
# 10. MODEL INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL INFORMATION")
print("=" * 60)

print(f"Number of clusters : {kmeans.n_clusters}")
print(f"Iterations         : {kmeans.n_iter_}")
print(f"Inertia / WCSS     : {kmeans.inertia_:.4f}")


# ------------------------------------------------------------
# 11. FINAL SILHOUETTE SCORE
# ------------------------------------------------------------

final_silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

print(f"Silhouette Score   : {final_silhouette:.4f}")


# ------------------------------------------------------------
# 12. CLUSTER COUNTS
# ------------------------------------------------------------

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 60)
print("NUMBER OF POINTS IN EACH CLUSTER")
print("=" * 60)

print(cluster_counts)


# ------------------------------------------------------------
# 13. CENTROIDS
# ------------------------------------------------------------
#
# K-Means learns one centroid for every cluster.
#
# These centroids are in the SCALED feature space because
# we trained the model using X_scaled.
# ------------------------------------------------------------

centroids_scaled = kmeans.cluster_centers_

print("\n" + "=" * 60)
print("CLUSTER CENTROIDS — SCALED SPACE")
print("=" * 60)

centroid_df = pd.DataFrame(
    centroids_scaled,
    columns=["Feature_1", "Feature_2"]
)

centroid_df.index.name = "Cluster"

print(centroid_df)


# ------------------------------------------------------------
# 14. CONVERT CENTROIDS BACK TO ORIGINAL SCALE
# ------------------------------------------------------------
#
# This is useful for interpretation.
#
# The model trained on scaled data, but humans usually want
# to understand the centroids in the original feature space.
# ------------------------------------------------------------

centroids_original = scaler.inverse_transform(
    centroids_scaled
)

centroid_original_df = pd.DataFrame(
    centroids_original,
    columns=["Feature_1", "Feature_2"]
)

centroid_original_df.index.name = "Cluster"

print("\n" + "=" * 60)
print("CLUSTER CENTROIDS — ORIGINAL SPACE")
print("=" * 60)

print(centroid_original_df.round(2))


# ------------------------------------------------------------
# 15. FINAL CLUSTER VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=cluster_labels,
    alpha=0.7
)

plt.scatter(
    centroids_scaled[:, 0],
    centroids_scaled[:, 1],
    marker="X",
    s=250,
    edgecolor="black",
    linewidth=1.5,
    label="Centroids"
)

plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.title("Final K-Means Clustering")

plt.legend()
plt.grid(alpha=0.2)

plt.show()


# ------------------------------------------------------------
# 16. UNDERSTAND WHAT K-MEANS ACTUALLY DID
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("HOW K-MEANS WORKED")
print("=" * 60)

print("""
1. We started with unlabeled data.
2. We selected a number of clusters (K).
3. K-Means initialized K centroids using k-means++.
4. Every point was assigned to its nearest centroid.
5. New centroids were calculated using the mean.
6. Steps 4 and 5 were repeated until convergence.
7. The final model minimized within-cluster squared distances.
""")


# ------------------------------------------------------------
# 17. INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(f"""
The final model created {optimal_k} clusters.

In this synthetic dataset:
- Each cluster represents a group of similar observations.
- The centroid represents the center of each cluster.
- Lower inertia indicates more compact clusters.
- A higher silhouette score generally indicates better
  separation between clusters.

Final Inertia       : {kmeans.inertia_:.4f}
Final Silhouette    : {final_silhouette:.4f}
""")


# ------------------------------------------------------------
# 18. IMPORTANT LIMITATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("K-MEANS LIMITATIONS")
print("=" * 60)

print("""
1. K must usually be specified beforehand.
2. Sensitive to feature scaling.
3. Sensitive to outliers.
4. Works best with compact / roughly spherical clusters.
5. Can struggle with clusters of very different densities.
6. Different initialization can lead to different solutions.
7. Cluster IDs have no inherent meaning.
8. Lower inertia alone does NOT mean better clustering.
9. Elbow method is heuristic rather than a strict rule.
10. Business/domain knowledge is still required to interpret clusters.
""")


# ------------------------------------------------------------
# 19. FINAL MODEL SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame(
    {
        "Metric": [
            "Algorithm",
            "Learning Type",
            "Number of Clusters",
            "Inertia / WCSS",
            "Silhouette Score",
            "Initialization",
            "Iterations"
        ],
        "Value": [
            "K-Means",
            "Unsupervised Learning",
            optimal_k,
            round(kmeans.inertia_, 4),
            round(final_silhouette, 4),
            "k-means++",
            kmeans.n_iter_
        ]
    }
)

print("\n" + "=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))


# ============================================================
# 🧠 DAY 35 REVISION NOTES
# ============================================================

revision_notes = """
============================================================
🧠 DAY 35 — REVISION NOTES: K-MEANS CLUSTERING
============================================================

1. WHAT IS UNSUPERVISED LEARNING?
   Learning patterns or structure from data without target
   labels.

2. WHAT IS CLUSTERING?
   Grouping similar observations together.

3. WHAT IS K-MEANS?
   A centroid-based unsupervised clustering algorithm.

4. WHAT DOES K REPRESENT?
   The number of clusters we want the algorithm to create.

5. CORE K-MEANS PROCESS

       Initialize K centroids
                ↓
       Assign points to nearest centroid
                ↓
       Calculate new centroids
                ↓
       Repeat until convergence

6. CENTROID
   The mean position of all points belonging to a cluster.

7. OBJECTIVE FUNCTION

       J = Σ ||xᵢ - μcᵢ||²

   K-Means tries to minimize the total squared distance
   between observations and their assigned centroids.

8. INERTIA / WCSS
   Measures the total within-cluster squared distance.

   Lower inertia → more compact clusters.

   BUT:
   Increasing K generally decreases inertia, so inertia
   alone cannot determine the best K.

9. ELBOW METHOD
   Run K-Means for several K values and plot inertia.
   Look for the point where additional clusters provide
   diminishing improvement.

10. SILHOUETTE SCORE

       s = (b - a) / max(a, b)

       a = average distance within own cluster
       b = average distance to nearest other cluster

       Close to +1 → strong clustering
       Around 0    → overlapping clusters
       Negative   → potentially poor assignment

11. K-MEANS++
   Better initialization strategy that spreads initial
   centroids across the dataset.

12. FEATURE SCALING
   Important because K-Means relies on distance calculations.
   Large-scale features can dominate smaller-scale features.

13. COMMON LIMITATIONS
   - Need to select K
   - Sensitive to outliers
   - Sensitive to initialization
   - Sensitive to feature scale
   - Poor with irregular cluster shapes
   - Assumes relatively compact clusters

14. REAL-WORLD APPLICATIONS
   - Customer segmentation
   - Image compression
   - Document clustering
   - Market segmentation
   - Pattern discovery

15. K-MEANS VS CLASSIFICATION

   Classification:
       X + known y → predict class

   K-Means:
       X only → discover clusters

16. IMPORTANT SCikit-Learn ATTRIBUTES

       model.labels_
       model.cluster_centers_
       model.inertia_
       model.n_iter_

17. MOST IMPORTANT INTERVIEW ANSWER

   K-Means does NOT guarantee the global optimum.
   It can converge to a local optimum, which is why
   initialization and multiple runs are important.

============================================================
ONE-LINE MEMORY TRICK

K-Means = Assign → Mean → Repeat → Converge

============================================================
"""


print(revision_notes)


# ============================================================
# 💼 LINKEDIN LEARNING POST
# ============================================================

linkedin_post = """
🚀 Day 35/60 — Starting Unsupervised Machine Learning

After completing the supervised learning part of my ML roadmap,
I’ve now started exploring Unsupervised Learning.

Today I learned K-Means Clustering — and the biggest shift in
thinking was that there are no target labels telling the model
what the correct answer is.

Instead, the goal is to discover structure in the data.

What I covered today:

• What unsupervised learning actually means
• Clustering and centroid-based learning
• How the K-Means algorithm works
• Centroids and distance-based assignment
• Within-Cluster Sum of Squares (WCSS) / inertia
• Elbow Method for selecting K
• Silhouette Score for evaluating cluster quality
• K-Means++ initialization
• Why feature scaling matters
• Limitations of K-Means
• Real-world applications such as customer segmentation
  and image compression

The core idea I’m taking away is simple:

K-Means repeatedly assigns points to the nearest centroid,
recalculates the centroid using the mean, and continues until
the clusters converge.

One thing that stood out to me is that clustering is not simply
about getting a low inertia value. Choosing the number of
clusters requires a combination of metrics, visualization, and
domain understanding.

This also made the limitations of K-Means clearer — especially
when dealing with outliers or clusters with irregular shapes.

Next, I’ll be moving to DBSCAN, which approaches clustering from
a completely different perspective using density.

Still learning, still experimenting, and building the
understanding one algorithm at a time.

#MachineLearning #UnsupervisedLearning #KMeans #DataScience
#100DaysOfCode #LearningInPublic
"""

print("\n" + "=" * 60)
print("💼 LINKEDIN POST")
print("=" * 60)

print(linkedin_post)
