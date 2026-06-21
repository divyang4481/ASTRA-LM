import torch
import argparse
from transformers import GPT2Tokenizer
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.model.config import ModelConfig
from astra_lm.utils.config_utils import load_config_from_yaml

def generate(
    model, tokenizer, prompt, max_new_tokens=30, temperature=0.7, device="cuda"
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs["logits"][:, -1, :]

            # Simple sampling
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--model_config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Starting text"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Load config
    m_cfg = load_config_from_yaml(ModelConfig, args.model_config)

    # Initialize model
    model = DecoderForCausalLM(m_cfg).to(args.device)

    # Load weights
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Generate
    print(f"\nPrompt: {args.prompt}")
    output = generate(model, tokenizer, args.prompt, device=args.device)
    print(f"Generated: {output}\n")

if __name__ == "__main__":
    main()
