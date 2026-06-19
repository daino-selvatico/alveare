import numpy as np

def sample(logits: np.ndarray, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> int:
    """
    Samples a token from logits using temperature, top-k, and top-p sampling.
    If temperature is 0, performs greedy sampling (argmax).
    """
    if temperature == 0.0:
        return int(np.argmax(logits))
        
    # Apply temperature
    logits = logits / max(temperature, 1e-5)
    
    # Softmax
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    
    # Top-K
    if top_k > 0:
        top_k = min(top_k, len(probs))
        # Find indices of top_k probabilities
        top_indices = np.argpartition(probs, -top_k)[-top_k:]
        # Get their probabilities and sort them descending
        top_probs = probs[top_indices]
        sort_idx = np.argsort(top_probs)[::-1]
        top_indices = top_indices[sort_idx]
        top_probs = top_probs[sort_idx]
    else:
        top_indices = np.arange(len(probs))
        sort_idx = np.argsort(probs)[::-1]
        top_indices = top_indices[sort_idx]
        top_probs = probs[sort_idx]
        
    # Top-P (nucleus sampling)
    if top_p < 1.0:
        cum_probs = np.cumsum(top_probs)
        # Keep tokens with cumulative probability <= top_p, plus the first one exceeding top_p
        cutoff_idx = np.where(cum_probs > top_p)[0]
        if len(cutoff_idx) > 0:
            top_k_p = cutoff_idx[0] + 1
            top_indices = top_indices[:top_k_p]
            top_probs = top_probs[:top_k_p]
            
    # Normalize probabilities again
    top_probs = top_probs / np.sum(top_probs)
    
    # Sample from top indices
    sampled_token = np.random.choice(top_indices, p=top_probs)
    return int(sampled_token)
