import torch
from data_types import Episode
from typing import List
from algorithms.utils import compute_entropy, cache_old_log_probs, get_returns, get_rewards

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                gamma=1, _lambda=0.01, k=7):
    delta = rewards + gamma * values[:, 1:] - values[:, :-1]
    B, S = delta.shape
    delta_expanded = torch.cat(delta, torch.zeros(B, k-1), dim=1)
    windows = delta_expanded.unfold(dimension=1, size=k, step=1)
    weights = (gamma * _lambda) ** torch.arange(k, dtype=delta_expanded.dtype, device=delta_expanded.device)
    gae = windows @ weights
    
    return gae.detach()
    

def update_policy(
    model,
    value_head,
    optimizer,
    episodes: List[Episode],
    hyper_params: dict,
    device: torch.device,
    dtype: torch.dtype,
): 
    micro_batch_size = hyper_params.get("micro_batch_size", 2)
    pad_token_id = hyper_params.get("pad_token_id")
    max_grad_norm = hyper_params.get("max_grad_norm", 1.0)
    num_epochs = hyper_params.get("num_epochs", 1)
    eps = hyper_params.get("epsilon", 0.2)
    gamma = hyper_params.get("gamma", 0.99)
    _lambda = hyper_params.get("lambda", 0.95)
    gae_k = hyper_params.get("gae_k", 10)
    coeff_vf = hyper_params.get("coeff_vf", 0.5)
    coeff_en = hyper_params.get("coeff_en", 0.01)
    loss = .0
    entropy = .0
    old_log_probs = cache_old_log_probs(model, episodes, micro_batch_size, pad_token_id, device, dtype)

    for k in range(num_epochs):
        loss = .0
        entropy = .0
        print(f"* Epoch {k:>2d}/{num_epochs:>2d} starts:")
        
        for i in range(0, len(episodes), micro_batch_size):
            print(
                f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
                flush=True,
                end=""
            )
            # mini batch loop
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
            batch_masks = [
                [0] * len(episode.prefix_token_ids) +
                [1] * len(episode.generated_token_ids) +
                [0] * (batch_max_length - (len(episode.prefix_token_ids) + len(episode.generated_token_ids)))
                for episode in batch_episodes
            ]
            batch_final_rewards = [episode.reward for episode in batch_episodes]
            
            # move to gpu
            batch_final_rewards = torch.tensor(batch_final_rewards, dtype=torch.float32, device=device)
            batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
            batch_masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)
            
            with torch.autocast(device_type=device, dtype=dtype):
                batch_hiddens = model.forward_hiddens(batch_token_ids) # (B, S+1)
                batch_logits = model.output_proj(batch_hiddens[:, :-1]).float() # (B, S)
                batch_values = value_head(batch_hiddens).float() # (B, S+1)
                
            batch_count = batch_masks.reshape(-1).sum()
            with torch.no_grad:
                batch_token_entropy = compute_entropy(batch_logits)
                entropy += (
                    batch_token_entropy.reshape(-1) * batch_masks.reshape(-1)
                ).sum() / batch_count
                
            B, S, V = batch_logits.shape
            batch_rewards = get_rewards(batch_final_rewards, batch_token_ids, pad_token_id)
            batch_returns = get_returns(batch_rewards, gamma)
            batch_advantages = compute_gae(batch_rewards, batch_values, gamma, _lambda, gae_k)
            
            batch_target = batch_token_ids[:, 1:]
            log_probs = torch.nn.functional.cross_entropy(batch_logits, batch_target, reduction='none', ignore_index=pad_token_id)
            batch_old_log_probs = torch.tensor(old_log_probs[i:j], dtype=torch.float32, device=device)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps)
            
            policy_objective = (torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages) * batch_masks).sum() / batch_count
            vf_loss = 0.5 * (batch_returns - batch_values[:, :-1]) ** 2
            
            objective = policy_objective - coeff_vf * vf_loss + coeff_en * entropy
            
            batch_loss = -objective
            batch_loss.backward()
            
            loss += batch_loss
            
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item()
    }