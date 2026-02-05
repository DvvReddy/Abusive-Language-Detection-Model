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


# Install the Hugging Face CLI
brew install huggingface-cli

# (optional) Login with your Hugging Face credentials
hf auth login

# Push your model files
hf upload Dvvreddy/70-SPDM . 




🏗️ Active Development
Note: This model is currently in a "Work in Progress" phase. We are continuously improving accuracy across low-resource languages.

How to Contribute
Report Issues: If you find language-specific bias, please open an issue.
Dataset Expansion: Contributions to multilingual toxicity datasets are highly encouraged.

Refine Model: Feel free to fork the repo and submit a PR with improved training arguments.
