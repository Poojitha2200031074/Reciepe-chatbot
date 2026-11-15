# Local LLM Recipe Chatbot (Fine-Tuned Qwen2.5 + FastAPI + Streamlit)

An end-to-end local AI pipeline that fine-tunes an open-source Large Language Model (`Qwen/Qwen2.5-0.5B-Instruct`) using LoRA on a custom recipes dataset, serves it via a **FastAPI** REST endpoint, and provides an interactive **Streamlit** chatbot UI.

---

## Prerequisites
- **OS:** Windows 10/11, Linux (Ubuntu/Debian), or macOS
- **Python:** Python 3.10 or 3.11
- **Hardware:** Standard laptop with at least 8 GB RAM (works on CPU; automatically accelerates if an NVIDIA CUDA GPU is available)
- **Internet:** Required once during setup to download Python packages and the ~1 GB base model from Hugging Face. Everything runs 100% locally afterward.

---

## 1. Setup & Installation

Create the project directory, place all project files inside it (`data/recipes.json`, `train.py`, `api.py`, `app.py`, `requirements.txt`), and set up a virtual environment.

### Windows (Command Prompt / PowerShell)
```bash
mkdir recipe-chatbot
cd recipe-chatbot
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt