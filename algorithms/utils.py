import torch
from data_types import Episode
from typing import List

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = (
        torch.logsumexp(logits, dim=-1)
        - torch.sum(probs * logits, dim=-1)
    )
    return entropy

def cache_old_log_probs(model, episodes: List[Episode], 
                        micro_batch_size: int, pad_token_id: int, device, dtype):
    print(f"Cache old log probability")
    
    old_log_probs = []
    for i in range(0, len(episodes), micro_batch_size):
        j = min(i+micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_max_length = max([
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ])
        batch_token_ids = [
            episode.prefix_token_ids + episode.generated_token_ids + 
            [pad_token_id] * (batch_max_length - (len(episode.prefix_token_ids) + len(episode.generated_token_ids)))
            for episode in batch_episodes
        ]
        
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
        with torch.autocast(device, dtype=dtype):
            logits = model.forward(batch_token_ids[:, :-1]).float()
            
        target = batch_token_ids[:, 1:]
        
        log_probs = - torch.nn.functional.cross_entropy(
            logits.transpose(-1, -2), target, reduction='none', ignore_index=pad_token_id
        )
        
        old_log_probs.extend(log_probs.tolist())
        
    return old_log_probs

def get_rewards(final_rewards: torch.Tensor, token_ids: torch.Tensor, pad_token_id: int):
    # only the last non-pad-token should get reward
    B, S = token_ids.shape
    rewards = torch.zeros_like(token_ids, dtype=final_rewards.dtype)
    pos = torch.arange(S, device=token_ids.device)
    last_pos = torch.where(token_ids != pad_token_id, pos, -1).max(dim=-1).values
    rewards[torch.arange(B, device=token_ids.device), last_pos] = final_rewards
    
    # r_t is given at a_t, a_t is actually token_{t+1}
    return rewards[:, 1:]

def get_returns(rewards: torch.Tensor, gamma: float):
    B, S = rewards.shape
    weights = gamma ** torch.arange(S, dtype=rewards.dtype, device=rewards.device)
    weights = torch.nn.functional.pad(weights, (S-1, 0)).unfold(0, S, 1).flip(0)
    
    returns = (weights[None, :, :] * rewards).sum(dim=1)
    
    return returns