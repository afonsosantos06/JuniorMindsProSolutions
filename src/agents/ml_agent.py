"""Machine Learning Cluster Agent: evaluates anomaly probability through unsupervised KMeans clustering with Risk Scoring."""
import json
import numpy as np

from langchain.tools import tool
from sklearn.cluster import KMeans
from src.data_loader import get_store
from src.features import build_ml_features

# Global cache for the fitted model and distance threshold
_kmeans_model = None
_anomaly_threshold = None


def _init_kmeans() -> None:
    """Initialize and fit the KMeans model on the entire dataset."""
    global _kmeans_model, _anomaly_threshold
    store = get_store()
    
    # Build feature matrix for all known historical transactions
    X_list = []
    for _, row in store.transactions.iterrows():
        tx = row.to_dict()
        feats = build_ml_features(tx)
        X_list.append(feats)
        
    X = np.array(X_list)
    
    if len(X) < 5:
        _anomaly_threshold = 9999.0
        return
        
    n_clusters = 3 if len(X) > 10 else 1
    _kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    clusters = _kmeans_model.fit_predict(X)
    centroids = _kmeans_model.cluster_centers_
    
    distances = []
    for i, x in enumerate(X):
        centroid = centroids[clusters[i]]
        dist = np.linalg.norm(x - centroid)
        distances.append(dist)
        
    _anomaly_threshold = np.percentile(distances, 95)


@tool
def check_cluster(transaction_json: str) -> str:
    """
    Evaluates if the transaction is mathematically anomalous using machine learning K-Means clustering.
    Outputs strict Risk Score.
    
    Input: JSON string with the transaction details.
    Output: {"risk_score": 0-100, "trigger": "string", "reasoning": "string"}
    Call this: ALWAYS for EVERY transaction.
    """
    global _kmeans_model, _anomaly_threshold
    
    if _kmeans_model is None and _anomaly_threshold is None:
        try:
            _init_kmeans()
        except Exception as e:
            return json.dumps({
                "risk_score": 30, 
                "trigger": "ML_INIT_ERROR", 
                "reasoning": f"KMeans initialization failed: {str(e)}"
            })
            
    if _kmeans_model is None:
        return json.dumps({
            "risk_score": 30, 
            "trigger": "NO_ML_DATA", 
            "reasoning": "Not enough data to support Clustering analysis."
        })

    tx = json.loads(transaction_json)
    features = np.array(build_ml_features(tx)).reshape(1, -1)
    
    cluster_idx = _kmeans_model.predict(features)[0]
    centroid = _kmeans_model.cluster_centers_[cluster_idx]
    
    dist = np.linalg.norm(features - centroid)
    
    if dist > _anomaly_threshold:
        # Penalize severely for outliers
        risk = min(100, 60 + (dist - _anomaly_threshold) * 20)
        return json.dumps({
            "risk_score": int(risk),
            "trigger": "ML_ANOMALY_OUTLIER",
            "reasoning": f"Mathematical Unsupervised Outlier: distance ({dist:.2f}) exceeds the 95th percentile threshold ({_anomaly_threshold:.2f})."
        })
    else:
        return json.dumps({
            "risk_score": 0,
            "trigger": "ML_INLIER",
            "reasoning": f"Inlier: Belongs to standard behavioral cluster (dist {dist:.2f} <= {_anomaly_threshold:.2f})."
        })
