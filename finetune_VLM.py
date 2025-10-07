# Import dependencies

import torch
import os
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers.image_utils import load_image
from datasets import load_dataset
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import wandb


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
).to(device)

processor = AutoProcessor.from_pretrained(model_name)

train_dataset, eval_dataset = load_dataset("HuggingFaceM4/ChartQA", split=["train[:10%]", "val[:10%]"])

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_data(sample):
    return {
        "images": [sample["image"]],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"][0]}],
            },
        ],
    }

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    output_dir="smol-course-smolvlm2-2.2b-instruct-trl-sft-ChartQA",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=25,
    save_strategy="steps",
    save_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=True,
    report_to="wandb",
    max_length=None,
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)
wandb.init(project="smolvlm2-chartqa", name="smolvlm2-2.2b-instruct-trl-sft-ChartQA")
trainer.train()
