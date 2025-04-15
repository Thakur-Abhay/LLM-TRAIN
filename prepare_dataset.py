# prepare_dataset.py
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, DATA_PATHS, TRAINING_PARAMS

# Initialize the tokenizer for the given model (e.g., FLAN-T5 Small)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    # Use 'input' instead of 'text' because the dataset columns are: ['input', 'summary', 'model']
    inputs = tokenizer(
        batch["input"],
        max_length=TRAINING_PARAMS["max_input_length"],
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        batch["summary"],
        max_length=TRAINING_PARAMS["max_summary_length"],
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

def main():
    # Load the dataset (it only contains a "train" split)
    dataset = load_dataset("kritsadaK/EDGAR-CORPUS-Financial-Summarization")

    # Show the keys to confirm the splits; should print only ['train']
    print("Dataset splits available:", list(dataset.keys()))

    # Create a validation split from the original train split (e.g., 90% train / 10% validation)
    dataset = dataset["train"].train_test_split(test_size=0.1)
    train_data = dataset["train"].map(preprocess, batched=True)
    val_data = dataset["test"].map(preprocess, batched=True)

    # Save the datasets to disk (update paths accordingly)
    train_data.save_to_disk(DATA_PATHS["train_data"])
    val_data.save_to_disk(DATA_PATHS["val_data"])
    print("Training and validation data prepared and saved.")

if __name__ == "__main__":
    main()
