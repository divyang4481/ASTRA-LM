import torch
import argparse
import logging
import gc
from astra_lm.model.config import ModelConfig
from astra_lm.model.decoder import DecoderForCausalLM
from astra_lm.utils import load_config_from_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tune_batch_size(model_config_path, seq_len, start_batch=1, max_batch=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.error("CUDA not available. Auto-tuning only makes sense on GPU.")
        return

    config = load_config_from_yaml(ModelConfig, model_config_path)
    logger.info(f"Loading model for tuning: {model_config_path}")
    model = DecoderForCausalLM(config).to(device)

    best_batch = 0

    for b in range(start_batch, max_batch + 1):
        try:
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"Testing batch size: {b} ...")
            input_ids = torch.randint(0, config.vocab_size, (b, seq_len), device=device)
            labels = input_ids.clone()

            # Forward + Backward
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            model.zero_grad()

            best_batch = b
            logger.info(f"Batch size {b} OK.")

        except torch.cuda.OutOfMemoryError:
            logger.info(f"Batch size {b} OOM.")
            break

    if best_batch > 0:
        logger.info(f"Recommended max batch size for seq_len {seq_len}: {best_batch}")
    else:
        logger.error("Could not even run batch size 1. Consider reducing seq_len.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    args = parser.parse_args()
    tune_batch_size(args.model_config, args.seq_len)
