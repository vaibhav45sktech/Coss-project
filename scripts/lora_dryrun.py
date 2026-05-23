"""
LoRA SFT dry-run — proves the SFT JSONL is training-ready.

Loads Qwen/Qwen2.5-0.5B-Instruct on CPU, attaches LoRA adapters via PEFT,
loads 10 samples from data/exports/sft_train.jsonl, configures TRL's
SFTTrainer for 2 training steps, and runs trainer.train().

If this completes without errors, the entire pipeline is training-ready.
This is the acceptance test for the SFT export shape.

Runtime on CPU: 3-7 minutes (depends on machine).
GPU is auto-detected and used if available.

Run:
    python scripts/lora_dryrun.py
"""

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lora_dryrun")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DATA = Path("data/exports/sft_train.jsonl")
N_SAMPLES = 10
MAX_STEPS = 2
OUTPUT_DIR = Path("./tmp_dryrun")


def main() -> None:
    if not SFT_DATA.exists():
        raise SystemExit(f"Training data not found: {SFT_DATA}. Run src.export_sft first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ---- Load tokenizer ----
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # ---- Load model ----
    logger.info(f"Loading model: {MODEL_NAME} (this may take 1-2 min on first run)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # CPU needs float32; GPU could use float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Model loaded. Total params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Attach LoRA ----
    logger.info("Configuring LoRA adapters")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # standard minimal LoRA target
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA attached. Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    # ---- Load and prepare dataset ----
    logger.info(f"Loading first {N_SAMPLES} samples from {SFT_DATA}")
    samples = []
    with SFT_DATA.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_SAMPLES:
                break
            samples.append(json.loads(line))

    # Render each sample's messages through the chat template
    def render(sample: dict) -> dict:
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    rendered = [render(s) for s in samples]
    dataset = Dataset.from_list(rendered)
    logger.info(f"Dataset built: {len(dataset)} samples, columns: {dataset.column_names}")
    logger.info(f"Sample text length (chars): {len(dataset[0]['text'])}")

    # ---- Training config ----
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    # ---- Build SFTTrainer ----
    logger.info("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
        args=training_args,
    )

    # ---- Train ----
    logger.info(f"Starting training for {MAX_STEPS} steps...")
    trainer.train()

    print("\n" + "=" * 60)
    print("  ✅ LoRA dry-run passed — SFT JSONL is training-ready")
    print("=" * 60)
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Samples used:    {N_SAMPLES}")
    print(f"  Training steps:  {MAX_STEPS}")
    print(f"  LoRA trainable:  {trainable:,} params ({100 * trainable / total:.3f}% of total)")
    print(f"  Device:          {device}")
    print(f"  This proves the entire pipeline produces training-compatible data.\n")


if __name__ == "__main__":
    main()
