from model import GPTModel
from utils import token_ids_to_text, text_to_token_ids, generate, model_configs, BASE_CONFIG
import torch
from instruction_fine_tuning import format_input
import tiktoken
import json


tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
torch.manual_seed(123)
CHOOSE_MODEL = "gpt2-medium (355M)"
base_config = BASE_CONFIG.copy()
base_config.update(model_configs[CHOOSE_MODEL])
base_config.update({"context_length": 1024, "qkv_bias": True})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(base_config)
model.to(device)
model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
##model.eval()


file_path = "instruction-data.json"
with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=base_config["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
)
print(input_text)
print(f"\nCorrect response:\n>> {entry['output']}")
print(f"\nModel response:\n>> {response_text.strip()}")
print("-------------------------------------")