import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM3-3B-Base",
    dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "SmolLM3-Custom-SFT/checkpoint-2000",
    dtype=torch.bfloat16
)

# 3. Merge adapter weights with base model
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # Implement fallback strategy or memory optimization

# 4. Save the merged model
merged_model.save_pretrained("SmolLM3-Custom-SFT/merged_model")

print("Merged model saved successfully.")

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained("SmolLM3-Custom-SFT/checkpoint-2000")
tokenizer.save_pretrained("SmolLM3-Custom-SFT/merged_model")

# Push the model and tokenizer to the hub
merged_model.push_to_hub("SmolLM3-CustomSFT_LoRA", organization="ocaetas")
tokenizer.push_to_hub("SmolLM3-CustomSFT_LoRA", organization="ocaetas")
