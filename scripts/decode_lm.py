import torch

from seq2seq.transformer.transformer import Decoder
from seq2seq.data.screenplay import tokenizer

def make_no_peak_mask(q, k, device=0):
    # Create a look-ahead mask to prevent attending to future tokens
    len_q, len_k = q.size(1), k.size(1)
    mask = torch.triu(
        torch.ones(len_q, len_k, device=device, dtype=torch.bool), diagonal=1
    )
    return mask


def decode(model, max_length, start_tokens=None, gen_len=2000, device="cpu", mode="top_p", temperature=0.8):
    model.eval()
    if start_tokens is None:
        # Start with the beginning of sequence token if no prompt is given
        tgt_tokens = [tokenizer.bos_token_id]
    else:
        tgt_tokens = [tokenizer.bos_token_id] + start_tokens

    for _ in range(gen_len):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)[:, -max_length:]
        tgt_mask = make_no_peak_mask(tgt_tensor, tgt_tensor, device)

        with torch.no_grad():
            output = model(tgt_tensor, tgt_mask=tgt_mask)

        next_token_logits = output[0, -1, :].clone() / temperature

        if mode == "top_k":
            indices_to_remove = (
                next_token_logits < torch.topk(next_token_logits, 20)[0][..., -1, None]
            )
            next_token_logits[indices_to_remove] = -float("inf")

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        elif mode == "top_p":
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float("inf")

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        elif mode == "greedy":
            next_token = torch.argmax(next_token_logits).item()

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

        print(tokenizer.decode(torch.tensor([next_token])), sep="", end="", flush=True)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model configuration from train_lm.py
    vocab_size = len(tokenizer.vocab)
    num_layers = 4
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 512 * 4
    qk_length = 512 // 8
    value_length = 512 // 8
    max_length = 512
    dropout = 0.1

    # Instantiate the model
    model = Decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        qk_length=qk_length,
        max_length=max_length,
        value_length=value_length,
        dropout=dropout,
    ).to(device)

    # Load the trained model weights
    model_path = "screenplay_lm_gpt_latest.pt"
    try:
        # The training script saves a checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        print("Please make sure the model file exists and the path is correct.")
        return
    except KeyError:
        # Fallback for models saved directly as state_dict
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading state dict: {e}")
            return

    model.eval()

    # --- Text Generation ---
    print("Generating text from the language model...")

    # Optional: Provide a starting prompt
    start_prompt = """ANAKIN"""
    start_tokens = tokenizer.encode(start_prompt).tolist()
    print(start_prompt, sep="", end="")
    _generated_text = decode(
        model, max_length, start_tokens=start_tokens, gen_len=2000, device=device, mode="top_k", temperature=0.5
    )
    print()


if __name__ == "__main__":
    main()
