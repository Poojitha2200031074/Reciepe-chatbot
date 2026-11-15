import gc
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/recipes.json"
OUTPUT_DIR = "./recipe-llm-finetuned"

def load_and_format_data(tokenizer):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted_samples = []
    for item in raw_data:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful culinary assistant. Suggest a clear, step-by-step recipe using the ingredients provided by the user."
            },
            {
                "role": "user",
                "content": f"Suggest a recipe using these ingredients: {item['ingredients']}"
            },
            {
                "role": "assistant",
                "content": item["recipe"]
            }
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted_samples.append({"text": text})

    dataset = Dataset.from_list(formatted_samples)

    # Dynamic truncation (no wasteful 512-token padding) saves massive CPU RAM
    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=256,
            truncation=True
        )
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        return model_inputs

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

def main():
    print(f"Loading base model and tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    # bfloat16 uses half the RAM of float32 on modern CPUs; float16 for CUDA
    dtype = torch.float16 if use_cuda else torch.bfloat16
    print(f"Hardware detected: {'CUDA GPU' if use_cuda else 'CPU'} (dtype={dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if use_cuda else None
    )

    # Gradient checkpointing drastically reduces peak RAM during backward pass
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_dataset = load_and_format_data(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,               # 6 epochs is plenty for 8 examples (12 total steps)
        per_device_train_batch_size=1,    # Batch size 1 prevents RAM spikes
        gradient_accumulation_steps=4,    # Accumulates gradients safely without extra memory
        learning_rate=3e-4,
        logging_steps=2,
        save_strategy="no",
        fp16=use_cuda,
        bf16=not use_cuda,
        use_cpu=not use_cuda,
        dataloader_pin_memory=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    gc.collect()
    print("Starting lightweight fine-tuning on custom recipe dataset...")
    trainer.train()

    print(f"Saving fine-tuned model and tokenizer to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")

if __name__ == "__main__":
    main()