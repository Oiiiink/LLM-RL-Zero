import torch
from data_types import Episode
from typing import List
from algorithms.utils import get_rewards, get_returns, compute_entropy

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
    gamma = hyper_params.get("gamma", 0.99)
    coeff_vf = hyper_params.get("coeff_vf", 0.5)
    coeff_en = hyper_params.get("coeff_en", 0.01)
    loss = .0
    entropy = .0
    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end=""
        )
        j = min(i+micro_batch_size, len(episodes))
        
        # data preparation 
        batch_episodes = episodes[i:j]
        batch_max_length = max([
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ])
        batch_token_ids = [
            episode.prefix_token_ids + 
            episode.generated_token_ids + 
            [pad_token_id] * (batch_max_length - len(episode.prefix_token_ids) - len(episode.generated_token_ids) )
            for episode in batch_episodes
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids) +
            [1] * len(episode.generated_token_ids) +
            [0] * (batch_max_length - len(episode.prefix_token_ids) - len(episode.generated_token_ids))
            for episode in batch_episodes
        ]  
        batch_final_rewards = [episode.reward for episode in batch_episodes]
        
        # move to GPU
        batch_final_rewards = torch.tensor(batch_final_rewards, dtype=torch.float32, device=device)
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
        batch_masks = torch.tensor(batch_masks[:, 1:], dtype=torch.bool, device=device)
        
        # forward pass
        with torch.autocast(device_type=device.type, dtype=dtype):
            batch_hiddens = model.forward_hiddens(batch_token_ids[:, :-1])
            batch_logits = model.output_proj(batch_hiddens).float()
            batch_values = value_head(batch_hiddens).float()
            
        
        B, S, V = batch_logits.shape
        batch_returns = get_returns(get_rewards(batch_final_rewards, batch_token_ids, pad_token_id), gamma)
        batch_advantages = batch_returns - batch_values.detach()
        
        batch_targets = batch_token_ids[:, 1:]
        batch_log_probs = - torch.nn.functional.cross_entropy(
            batch_logits.transpose(-1, -2), 
            batch_targets,
            reduction='None',
            ignore_index=pad_token_id
        )
        
        batch_count = batch_masks.reshape(-1).sum()
        batch_token_entropy = compute_entropy(batch_logits)
        entropy = (
            batch_token_entropy.reshape(-1) * batch_masks.reshape(-1)
        ).sum() / batch_count
        
        # calculate loss
        assert batch_logits.shape == batch_masks.shape == batch_log_probs.shape, f"{batch_logits.shape} == {batch_masks.shape} == {batch_log_probs.shape}"
        batch_policy_loss = (- batch_advantages * batch_log_probs * batch_masks).reshape(-1).sum() / batch_count
        batch_value_loss = ((0.5 * (batch_values - batch_returns) **2) * batch_masks).reshape(-1).sum() / batch_count
        batch_loss = batch_policy_loss + coeff_vf * batch_value_loss - coeff_en * entropy
        batch_loss.backward()
        
        loss += batch_loss.detach()    
    
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
    )
    optimizer.step()
    optimizer.zero_grad()
    
    return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "entropy": entropy.item(),
        }