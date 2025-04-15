# config.py
MODEL_NAME = "google/flan-t5-small"

TRAINING_PARAMS = {
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "max_input_length": 512,
    "max_summary_length": 128,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
}

DATA_PATHS = {
    "dataset": "../data/dataset",
    "train_data": "../data/dataset/train.csv",
    "val_data": "../data/dataset/val.csv",
}

OUTPUT_DIR = "../outputs"
