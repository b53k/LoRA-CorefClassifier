import torch

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.uniform_(m.bias)


def bucketize(score, prefix = "S", bins=[0.25, 0.5, 0.75]):
    if score < bins[0]:
        return f"<{prefix}=low>"
    elif score < bins[1]:
        return f"<{prefix}=mid>"
    elif score < bins[2]:
        return f"<{prefix}=high>"
    else:
        return f"<{prefix}=veryhigh>"
    

def compute_doc_score(doc):
    '''
        Computes 3 document-level (not cross-document level) heuristic scores:
        - score1: mention density
        - score2: cluster richness
        - score3: length normalization

        Returns:
            List of raw [score1, score2, score3], each \in [0.0, 1.0] 
    '''

    total_tokens = sum(len(sent['words']) for sent in doc['sentences'])
    total_mentions = sum(len(sent['coref_spans']) for sent in doc['sentences'])

    # Collect all unique cluster IDs
    cluster_ids = set()
    for sent in doc['sentences']:
        for cid, _, _ in sent['coref_spans']:
            cluster_ids.add(cid)
    
    total_clusters = len(cluster_ids)

    score1 = total_mentions / (total_tokens + 1e-6)         # mention density
    score2 = total_clusters / (total_mentions + 1e-6)       # cluster richness
    score3 = min(total_tokens / 669.75, 1.0)                 # doc length normalization: mean word count per doc in training set is 669.75

    score1 = min(score1, 1.0)
    score2 = min(score2, 1.0)

    return [score1, score2, score3]


def extract_mention_repr(input_ids, hidden_states, m_token_id, m_end_token_id, include_scores = False):
    '''
        Extract [CLS, mention1, mention2, mention1*mention2] representation from hidden states....similar to the paper
        Returns a tensor of shape (batch_size, 4 * hidden_dim)
    '''
    
    batch_size = input_ids.size(0)
    reps = []

    for i in range(batch_size):
        ids = input_ids[i]              # (seq_len,)
        hs = hidden_states[i]           # (seq_len, hidden_dim)

        # Get CLS token (assume at index 0)
        cls_vec = hs[0]

        # Find all <m> and </m> positions
        m_starts = (ids == m_token_id).nonzero(as_tuple=True)[0]
        m_ends = (ids == m_end_token_id).nonzero(as_tuple=True)[0]

        if len(m_starts) != 2 or len(m_ends) != 2:
            raise ValueError("Expected exactly two <m> and two </m> markers per example")
        
        # Mention 1
        start1 = m_starts[0].item() + 1
        end1 = m_ends[0].item()
        mention1_vec = hs[start1:end1].sum(dim=0)   # (hidden_dim,)

        # Mention 2
        start2 = m_starts[1].item() + 1
        end2 = m_ends[1].item()
        mention2_vec = hs[start2:end2].sum(dim=0)   # (hidden_dim,)

        # Elementwise product
        mention_product = mention1_vec * mention2_vec
        
        combined = [cls_vec, mention1_vec, mention2_vec, mention_product]

        if include_scores:
            s1, s2, s3 = hs[1], hs[2], hs[3]
            s_sum = s1 + s2 + s3
            combined.extend([s1, s2, s3, s_sum])

        concat = torch.cat(combined, dim=-1)           
        reps.append(concat)

    return torch.stack(reps)    # (batch_size, 4 * hidden_dim)
