import numpy as np


def calculate_map_at_k(relevance_scores, k):
    """
    Calculate MAP@k for a single query.
    
    Args:
        relevance_scores (list): List of binary relevance scores (1 for relevant, 0 for irrelevant).
        k (int): Number of top items to consider.

    Returns:
        float: MAP@k score.
    """
    # Limit to top-k
    relevance_scores = relevance_scores[:k]
    
    # Compute Precision@i for each position
    precisions = [
        sum(relevance_scores[:i+1]) / (i+1)  # Precision@i
        for i in range(len(relevance_scores)) if relevance_scores[i] == 1
    ]
    
    # Average Precision@k
    return np.mean(precisions) if precisions else 0.0

# Example usage
relevance_scores = [1, 0, 1, 1, 0]  # Binary relevance scores
k = 3
map_at_k = calculate_map_at_k(relevance_scores, k)
print(f"MAP@{k}: {map_at_k:.3f}")
