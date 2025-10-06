from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import wandb  # Optional: for experiment tracking
from datasets import concatenate_datasets

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS")
    else:
        device = "cpu"
        print("Using CPU - you will need to use a GPU to train models")

        # Import required libraries for fine-tuning

    # Initialize Weights & Biases (optional)
    # wandb.init(project="smollm3-finetuning")

    # Load SmolLM3 base model for fine-tuning
    model_name = "HuggingFaceTB/SmolLM3-3B-Base"
    new_model_name = "SmolLM3-Custom-SFT"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    tokenizer.padding_side = "right"  # Padding on the right for generation
    instruct_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    print(f"Model loaded! Parameters: {model.num_parameters():,}")

    # Example: Process GSM8K math dataset
    print("=== PROCESSING GSM8K DATASET 1 ===\n")

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    val = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Original GSM8K example: {gsm8k[0]}")

    # Convert to chat format
    def process_gsm8k(example):
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]}
        ]

        text = instruct_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    gsm8k_processed = gsm8k.map(process_gsm8k)
    val_processed = val.map(process_gsm8k)
    gsm8k_formatted = gsm8k_processed.remove_columns(
        [col for col in gsm8k_processed.column_names if col != "text"]
    )
    val_formatted = val_processed.remove_columns(
        [col for col in val_processed.column_names if col != "text"]
    )
    print(f"Processed GSM8K example: {gsm8k_formatted[0]['text'][:200]}...")

    # Load and prepare training dataset
    print("=== PREPARING DATASET 2 ===\n")

    # Option 1: Use SmolTalk2 (recommended for beginners)
    dataset = load_dataset("HuggingFaceTB/smoltalk2", "SFT")
    train_dataset = dataset["smoltalk_everyday_convs_reasoning_Qwen3_32B_think"]

    # Option 2: Use your own processed dataset from Exercise 2
    # train_dataset = gsm8k_formatted.select(range(500))

    print(f"Training examples: {len(train_dataset)}")
    print(f"Example: {train_dataset[0]}")

    # Prepare the dataset for SFT
    def format_chat_template(example):
        """Format the messages using the chat template"""
        if "messages" in example:
            # SmolTalk2 format
            messages = example["messages"]
        else:
            # Custom format - adapt as needed
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]}
            ]
        
        # Apply chat template
        text = instruct_tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    # Apply formatting
    formatted_dataset = train_dataset.map(format_chat_template)
    formatted_dataset = formatted_dataset.remove_columns(
        [col for col in formatted_dataset.column_names if col != "text"]
    )
    print(f"Formatted example: {formatted_dataset[0]['text'][:200]}...")
    
    # Append both datasets
    formatted_dataset = concatenate_datasets([formatted_dataset, gsm8k_formatted])
    print(f"Total training examples after concatenation: {len(formatted_dataset)}")

    wandb.init(project="smollm3-finetuning")

    # Configure training parameters
    training_config = SFTConfig(
        # Model and data
        output_dir=f"./{new_model_name}",
        dataset_text_field="text",
        max_length=2048,
        
        # Training hyperparameters
        per_device_train_batch_size=4,  # Adjust based on your GPU memory
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=3,  # Start with 1 epoch
        max_steps=20000,  # Limit steps for demo
        
        # Optimization
        warmup_steps=2000,
        weight_decay=0.01,
        optim="adamw_torch",
        
        # Logging and saving
        logging_steps=200,
        save_steps=2000,
        eval_steps=2000,
        save_total_limit=11,
        
        # Memory optimization
        dataloader_num_workers=6,
        group_by_length=True,  # Group similar length sequences
        
        # Hugging Face Hub integration
        push_to_hub=False,  # Set to True to upload to Hub
        hub_model_id=f"ocaetas/{new_model_name}",
        
        # Experiment tracking
        report_to=["wandb"],  # Use wandb for experiment tracking
        run_name=f"{new_model_name}-training",
    )

    print("Training configuration set!")
    print(f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")


    # LoRA configuration with PEFT
    from peft import LoraConfig

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    print("LoRA configuration set!")

    lora_trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,  # dataset with a "text" field or messages + dataset_text_field in config
        eval_dataset=val_formatted,
        args=training_config,
        peft_config=peft_config,  # << enable LoRA
    )

    print("Starting LoRA training…")
    lora_trainer.train()

