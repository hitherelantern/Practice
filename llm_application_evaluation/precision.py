import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def calculate_precision_at_k(data_samples, k, similarity_threshold=0.8, model=None):
    """
    Calculate Precision@k for a dataset using semantic similarity.
    
    Args:
        data_samples (dict): Dictionary containing 'contexts' and 'ground_truth'.
        k (int): Number of top retrieved items to consider.
        similarity_threshold (float): Threshold for determining relevance.
        model (SentenceTransformer): Pre-trained embedding model.

    Returns:
        dict: Precision@k for each question.
    """
    if model is None:
        raise ValueError("An embedding model is required for semantic similarity.")

    results = {}
    for i, (contexts, ground_truth) in enumerate(zip(data_samples['contexts'], data_samples['ground_truth'])):
        # Limit to top-k contexts
        contexts = contexts[:k]
        
        # Compute embeddings
        context_embeddings = model.encode(contexts)
        ground_truth_embedding = model.encode([ground_truth])
        
        # Compute cosine similarity
        similarities = cosine_similarity(context_embeddings, ground_truth_embedding).flatten()
       
        print(f"similarities{i}:{similarities}")
        
        # Determine relevance (similarity > threshold)
        relevance = [1 if sim > similarity_threshold else 0 for sim in similarities]
        
        # Calculate Precision@k
        precision = sum(relevance) / k
        results[data_samples['question'][i]] = precision
    
    return np.mean(list(results.values())),results





# Example usage
data_samples = {
    'question': [
        'When was the first super bowl?', 
        'Who won the most super bowls?'
    ],
    'answer': [
        'The first superbowl was held on Jan 15, 1967', 
        'The most super bowls have been won by The New England Patriots'
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

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Calculate Precision@k for k = 1 and semantic similarity
precision_at_1 = calculate_precision_at_k(data_samples, k=1, similarity_threshold=0.5, model=model)
print("Precision@1:", precision_at_1)

# Calculate Precision@k for k = 2 and semantic similarity
precision_at_2 = calculate_precision_at_k(data_samples, k=2, similarity_threshold=0.5, model=model)
print("Precision@2:", precision_at_2)
