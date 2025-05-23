import numpy as np

def calculate_ndcg_at_k(relevance_scores, k):
    """
    Calculate nDCG@k for a single query.
    
    Args:
        relevance_scores (list): List of relevance scores in the retrieved order.
        k (int): Number of top items to consider.

    Returns:
        float: nDCG@k score.
    """
    # Limit to top-k
    relevance_scores = relevance_scores[:k]
    
    # Compute DCG@k
    dcg = sum([
        (2**rel - 1) / np.log2(idx + 2)  # log2(idx + 2) because idx is 0-based
        for idx, rel in enumerate(relevance_scores)
    ])
    
    # Compute IDCG@k (Ideal DCG)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(ideal_relevance_scores)
    ])
    
    # Normalize DCG by IDCG
    return dcg / idcg if idcg > 0 else 0.0

# Example usage
relevance_scores = [3, 2, 3, 0, 1]  # Relevance scores of retrieved items
k = 3
ndcg_at_k = calculate_ndcg_at_k(relevance_scores, k)
print(f"nDCG@{k}: {ndcg_at_k:.3f}")
