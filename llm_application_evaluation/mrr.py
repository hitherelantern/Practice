import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_mrr_at_k(data_samples, k, similarity_threshold=0.8, model=None):
    """
    Calculate MRR@k for a dataset using semantic similarity.

    Args:
        data_samples (dict): Dictionary containing 'contexts' and 'ground_truth'.
        k (int): Number of top retrieved items to consider.
        similarity_threshold (float): Threshold for determining relevance.
        model (SentenceTransformer): Pre-trained embedding model.

    Returns:
        float: MRR@k score for the dataset.
    """
    if model is None:
        raise ValueError("An embedding model is required for semantic similarity.")

    reciprocal_ranks = []
    for i, (contexts, ground_truth) in enumerate(zip(data_samples['contexts'], data_samples['ground_truth'])):
        # Compute embeddings
        context_embeddings = model.encode(contexts)
        ground_truth_embedding = model.encode([ground_truth])
        
        # Compute cosine similarity
        similarities = cosine_similarity(context_embeddings, ground_truth_embedding).flatten()
        
        # Sort contexts by descending similarity
        sorted_indices = np.argsort(similarities)[::-1]
        print(f"sorted indices:{sorted_indices}")
        # sorted_contexts = [contexts[j] for j in sorted_indices]
        sorted_relevance = [1 if similarities[j] > similarity_threshold else 0 for j in sorted_indices]
        print(f"sorted relevance:{sorted_relevance}")
        # Limit to top-k
        sorted_relevance = sorted_relevance[:k]
        
        # Find the rank of the first relevant item
        try:
            first_relevant_rank = sorted_relevance.index(1) + 1  # Convert 0-based to 1-based
            reciprocal_ranks.append(1 / first_relevant_rank)
        except ValueError:
            reciprocal_ranks.append(0)  # No relevant item in top-k
    
    # Compute mean reciprocal rank
    mrr_at_k = np.mean(reciprocal_ranks)
    return mrr_at_k

# Example usage
data_samples = {
    'question': [
        'When was the first super bowl?', 
        'Who won the most super bowls?'
    ],
    'contexts': [
        [
            'The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'
        ], 
        [
            'The Green Bay Packers...Green Bay, Wisconsin.',
            'The Packers compete...Football Conference'
        ]
    ],
    'ground_truth': [
        'The first superbowl was held on January 15, 1967', 
        'The New England Patriots have won the Super Bowl a record six times'
    ]
}

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Calculate MRR@k for k=3
mrr_at_3 = calculate_mrr_at_k(data_samples, k=3, similarity_threshold=0.5, model=model)
print("MRR@3:", mrr_at_3)
