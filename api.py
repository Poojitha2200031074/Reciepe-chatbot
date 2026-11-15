import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_DIR = "./recipe-llm-finetuned"

app = FastAPI(
    title="Local Recipe LLM API",
    description="API serving a locally fine-tuned LLM for ingredient-based recipe suggestions.",
    version="1.0.0"
)

class RecipeRequest(BaseModel):
    ingredients: str = Field(..., example="Egg, Onions", description="Comma-separated list of ingredients")
    max_new_tokens: int = Field(default=300, ge=50, le=800)
    temperature: float = Field(default=0.4, ge=0.1, le=1.5)

class RecipeResponse(BaseModel):
    ingredients: str
    recipe: str
    model_used: str

# Global variables for model and tokenizer
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
active_model_label = ""

@app.on_event("startup")
def load_model():
    global tokenizer, model, active_model_label
    dtype = torch.float16 if device == "cuda" else torch.float32

    if os.path.exists(FINETUNED_DIR):
        print(f"Loading fine-tuned LoRA model from {FINETUNED_DIR} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, torch_dtype=dtype
        ).to(device)
        model = PeftModel.from_pretrained(base_model, FINETUNED_DIR).to(device)
        active_model_label = f"{BASE_MODEL_NAME} + LoRA (Fine-Tuned)"
    else:
        print(f"Fine-tuned directory not found. Loading base model {BASE_MODEL_NAME} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, torch_dtype=dtype
        ).to(device)
        active_model_label = f"{BASE_MODEL_NAME} (Base)"

    model.eval()
    print("Model loaded and ready to serve requests!")

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device, "model": active_model_label}

@app.post("/generate-recipe", response_model=RecipeResponse)
def generate_recipe(request: RecipeRequest):
    if not request.ingredients.strip():
        raise HTTPException(status_code=400, detail="Ingredients cannot be empty.")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful culinary assistant. Suggest a clear, step-by-step recipe using the ingredients provided by the user."
        },
        {
            "role": "user",
            "content": f"Suggest a recipe using these ingredients: {request.ingredients.strip()}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    recipe_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return RecipeResponse(
        ingredients=request.ingredients.strip(),
        recipe=recipe_text,
        model_used=active_model_label
    )