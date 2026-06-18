import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50, device="cpu"):
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]

        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        next_token_logits = next_token_logits / temperature

        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)

        sampled_idx = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices.gather(-1, sampled_idx)

        if next_token.item() == tokenizer.eos_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)