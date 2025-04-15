# inference.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from config import MODEL_NAME, OUTPUT_DIR, TRAINING_PARAMS

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model = PeftModel.from_pretrained(base_model, f"{OUTPUT_DIR}/final_model").to(device)

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=TRAINING_PARAMS["max_input_length"],
                       truncation=True).to(device)

    summary_ids = model.generate(**inputs, max_new_tokens=TRAINING_PARAMS["max_summary_length"])
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_text = ("For 2024, XYZ Corp reported revenue growth of 10% due to higher North America sales. "
                   "Operating margins increased to 15%, net debt reduced by 5% to $500 million. "
                   "2025 outlook shows moderate growth.")

    print("Summary:", summarize(sample_text))
