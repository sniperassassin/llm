import torch
from model import GPTModel
from utils import BASE_CONFIG, model_configs, text_to_token_ids, token_ids_to_text, generate
import tiktoken

torch.manual_seed(123)
model_name = "gpt2-medium (355M)"
new_config = BASE_CONFIG.copy()
new_config.update({
    "context_length": 1024,
    "qkv_bias": True
})
new_config.update(model_configs[model_name])
model = GPTModel(new_config)
tokenizer = tiktoken.get_encoding("gpt2")

# Load pretrained model for inference.
model.load_state_dict(torch.load("gptModel-355M.pth"))
model.eval()

# Load the model weights when we want to train the model further.
# checkpoint = torch.load("checkpoint.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.train()  # if resuming training

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=new_config["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))