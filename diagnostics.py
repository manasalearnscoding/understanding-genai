from huggingface_hub import list_repo_files
files = list_repo_files("meta-llama/Llama-2-13b-chat")
model_files = [f for f in files if any(ext in f for ext in ['pytorch_model', 'safetensors', '.bin'])]
print("Model files found:", model_files)