# Abusive-Language-Detection-Model
Multilingual Toxicity Detector A cross-lingual model for identifying abusive content and offensive language. 
Currently in active development; open to contributions.


🚀 Hugging Face Model
The model is hosted and available for inference on Hugging Face: [View Model on Hugging Face](https://huggingface.co/Dvvreddy/70-SPDM)

📂 Project Structure
model1-training.py: The core script used for fine-tuning the transformer model.
tokenizer.json / config.json: Configuration files for reproducing the model's vocabulary and architecture.
scaler.pt / scheduler.pt: Checkpoints for optimization states and gradient scaling.
trainer_state.json: Detailed logs of the training metrics and loss history.


🛠️ Usage
Prerequisites


Bash


pip install transformers torch

Basic Inference

You can load the model directly using the transformers library:

Python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "path_to_your_downloaded_files"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "Your sample comment here"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    prediction = torch.softmax(logits, dim=1)

print(f"Toxicity Score: {prediction[0][1].item():.4f}")
🏗️ Active Development
Note: This model is currently in a "Work in Progress" phase. We are continuously improving accuracy across low-resource languages.

How to Contribute

Report Issues: If you find language-specific bias, please open an issue.

Dataset Expansion: Contributions to multilingual toxicity datasets are highly encouraged.

Refine Model: Feel free to fork the repo and submit a PR with improved training arguments.
