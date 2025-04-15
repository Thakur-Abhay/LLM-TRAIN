# fine_tune.py
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
from config import MODEL_NAME, TRAINING_PARAMS, OUTPUT_DIR, DATA_PATHS

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    train_data = load_from_disk(DATA_PATHS["train_data"])
    val_data = load_from_disk(DATA_PATHS["val_data"])

    # Load model and apply LoRA
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=TRAINING_PARAMS["lora_r"],
        lora_alpha=TRAINING_PARAMS["lora_alpha"],
        lora_dropout=TRAINING_PARAMS["lora_dropout"],
    )
    model = get_peft_model(model, peft_config)

    # Training configuration
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAINING_PARAMS["batch_size"],
        gradient_accumulation_steps=TRAINING_PARAMS["gradient_accumulation_steps"],
        learning_rate=TRAINING_PARAMS["learning_rate"],
        num_train_epochs=TRAINING_PARAMS["num_epochs"],
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    main()
