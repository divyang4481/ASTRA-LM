import os
import argparse
import numpy as np
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Prepare GPT-2 pretraining data from FineWeb-Edu")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="HuggingFace dataset name")
    parser.add_argument("--name", type=str, default="sample-10BT", help="Dataset configuration/subset name")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--train_tokens", type=int, default=10_000_000, help="Target number of training tokens")
    parser.add_argument("--val_tokens", type=int, default=500_000, help="Target number of validation tokens")
    parser.add_argument("--out_dir", type=str, default="data/fineweb_edu_gpt2_10m", help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = 50256 # GPT-2 default EOS

    print(f"Streaming dataset: {args.dataset} (name: {args.name})")
    ds = load_dataset(args.dataset, name=args.name, split="train", streaming=True)

    def process_split(split_name, target_tokens):
        out_path = os.path.join(args.out_dir, f"{split_name}.npy")
        print(f"Processing {split_name} (target: {target_tokens} tokens)...")

        # Use a list to collect tokens until we hit the target
        all_tokens = []
        tokens_count = 0

        pbar = tqdm(total=target_tokens, unit="tokens")

        for example in ds:
            text = example["text"]
            tokens = tokenizer.encode(text)
            tokens.append(eos_token_id) # Append EOS between documents

            all_tokens.extend(tokens)
            tokens_count += len(tokens)
            pbar.update(len(tokens))

            if tokens_count >= target_tokens:
                break

        pbar.close()

        # Trim to exact target if needed
        all_tokens = all_tokens[:target_tokens]

        print(f"Saving {len(all_tokens)} tokens to {out_path} as int32...")
        arr = np.array(all_tokens, dtype=np.int32)
        np.save(out_path, arr)
        return len(all_tokens)

    # For simplicity in this script, we just stream from the same 'train' split
    # but sequential chunks. In a production setting, one might use different splits.
    # However, many large streaming datasets only have a 'train' split.

    # We'll take validation tokens first, then training tokens.
    val_count = process_split("val", args.val_tokens)
    train_count = process_split("train", args.train_tokens)

    # Save metadata
    meta = {
        "vocab_size": len(tokenizer),
        "tokenizer": args.tokenizer,
        "dataset": args.dataset,
        "subset": args.name,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "eos_token_id": eos_token_id
    }

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Data preparation complete! Metadata saved to {os.path.join(args.out_dir, 'meta.json')}")

if __name__ == "__main__":
    main()
