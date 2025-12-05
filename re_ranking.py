# re_ranking.py
# FIXED VERSION - Division safety improved
#
# FIXES APPLIED:
# 1. Safe normalization with proper clamping to avoid NaN

import torch
import torch.nn.functional as F


def re_ranking(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    """
    GPU-Optimized k-reciprocal Encoding Re-Ranking.
    
    Args:
        probFea (numpy.ndarray or torch.Tensor): Query features (M, dim)
        galFea (numpy.ndarray or torch.Tensor): Gallery features (N, dim)
        k1 (int): k-reciprocal nearest neighbors
        k2 (int): k-nearest neighbors for query expansion
        lambda_value (float): Weight for the original distance
        
    Returns:
        final_dist (numpy.ndarray): Re-ranked distance matrix (M, N)
    """
    # 1. Setup & Device Management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure inputs are tensors on the correct device
    if not isinstance(probFea, torch.Tensor):
        query = torch.from_numpy(probFea).to(device)
    else:
        query = probFea.to(device)
        
    if not isinstance(galFea, torch.Tensor):
        gallery = torch.from_numpy(galFea).to(device)
    else:
        gallery = galFea.to(device)

    # FP16 optimization for speed/memory
    query = query.half() if query.is_floating_point() else query
    gallery = gallery.half() if gallery.is_floating_point() else gallery

    num_query = query.shape[0]
    num_gallery = gallery.shape[0]
    all_num = num_query + num_gallery
    
    # Handle edge cases
    if num_query == 0 or num_gallery == 0:
        return np.zeros((num_query, num_gallery))
    
    # Concatenate features
    features = torch.cat([query, gallery], dim=0)

    # 2. Optimized Euclidean Distance
    original_dist = torch.cdist(features, features).float()
    
    # FIXED: Safe normalization with proper clamping
    max_dist_per_col = torch.max(original_dist, dim=0)[0]
    max_dist_per_col = torch.clamp(max_dist_per_col, min=1e-6)  # Ensure non-zero
    original_dist = original_dist / max_dist_per_col
    
    # Keep the query-gallery block for the final result
    original_dist_qg = original_dist[:num_query, num_query:]

    # 3. Initial Ranking (Top-K)
    k_search = min(k1 + 50, all_num)
    _, initial_rank = torch.topk(original_dist, k=k_search, dim=1, largest=False)
    
    initial_rank = initial_rank.long()
    
    nn_k1 = initial_rank[:, :k1]
    
    # Prepare sparse components
    rows = []
    cols = []
    vals = []
    
    for i in range(all_num):
        f_idx = nn_k1[i]
        b_idx = nn_k1[f_idx] 
        
        reciprocal_mask = (b_idx == i).any(dim=1)
        k_reciprocal_idx = f_idx[reciprocal_mask]
        
        if k_reciprocal_idx.shape[0] > 0:
            current_dist = original_dist[i, k_reciprocal_idx]
            current_weights = torch.exp(-current_dist)
            
            rows.append(torch.full_like(k_reciprocal_idx, i))
            cols.append(k_reciprocal_idx)
            vals.append(current_weights)

    # Check if we have any valid reciprocal neighbors
    if not rows:
        # Fallback to original distances
        return original_dist_qg.cpu().numpy()
    
    rows = torch.cat(rows)
    cols = torch.cat(cols)
    vals = torch.cat(vals)
    
    # Create Sparse Matrix V
    V = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), 
        vals, 
        (all_num, all_num)
    ).to_dense()

    # 5. Query Expansion (k2)
    if k2 > 1:
        top_k2_indices = initial_rank[:, :k2]
        T_rows = torch.arange(all_num, device=device).unsqueeze(1).expand_as(top_k2_indices)
        T_vals = torch.ones_like(top_k2_indices, dtype=torch.float32) / k2
        
        T = torch.sparse_coo_tensor(
            torch.stack([T_rows.reshape(-1), top_k2_indices.reshape(-1)]),
            T_vals.reshape(-1),
            (all_num, all_num)
        )
        
        V = torch.sparse.mm(T, V)

    # 6. Vectorized Jaccard Distance
    V_query = V[:num_query]
    V_gallery = V[num_query:]
    
    intersection = torch.mm(V_query, V_gallery.t())
    
    V_q_sq = (V_query ** 2).sum(1).unsqueeze(1)
    V_g_sq = (V_gallery ** 2).sum(1).unsqueeze(0)
    
    # FIXED: Safe division with clamping
    denominator = V_q_sq + V_g_sq - intersection + 1e-6
    jaccard_dist = 1.0 - (intersection / denominator)
    
    # 7. Final Fusion
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist_qg * lambda_value
    
    return final_dist.cpu().numpy()


# Import numpy for the edge case return
import numpy as np