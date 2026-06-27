import argparse
import torch
import logging
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.utils import load_config_from_yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading config from {args.model_config}")
    config = load_config_from_yaml(ModelConfig, args.model_config)
    config.max_seq_len = args.seq_len # Ensure config matches requested seq_len

    logger.info("Initializing model...")
    model = DecoderForCausalLM(config).to(args.device)

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=args.device)

    logger.info("Running forward pass...")
    outputs = model(input_ids, return_diagnostics=True)
    logits = outputs["logits"]

    logger.info(f"Logits shape: {logits.shape}")
    assert logits.shape == (args.batch_size, args.seq_len, config.vocab_size), "Shape mismatch"
    assert not torch.isnan(logits).any(), "NaN in logits"
    logger.info("Forward OK")

    logger.info("Running backward pass...")
    loss = outputs.get("loss")
    if loss is None:
        loss = logits.mean()
    loss.backward()
    logger.info("Backward OK")

    if "diagnostics" in outputs:
        logger.info("Route stats:")
        for i, diag in enumerate(outputs["diagnostics"]):
            if diag:
                logger.info(f"Layer {i}: {diag}")

    logger.info("Smoke test passed!")

if __name__ == "__main__":
    main()
