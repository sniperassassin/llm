import json
import os
import urllib
import tiktoken
import torch
from functools import partial
from torch.utils.data import DataLoader, Dataset
from model import GPTModel
from train_model import train_model_simple
import time
import re
from utils import model_configs, BASE_CONFIG

class InstructionDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data

            # Pre-tokenize texts
            self.encoded_texts = []
            for entry in data:
                instruction_plus_input = format_input(entry)
                response_text = f"\n\n### Response:\n{entry['output']}"
                full_text = instruction_plus_input + response_text
                self.encoded_texts.append(
                    tokenizer.encode(full_text)
                )

        def __getitem__(self, index):
            return self.encoded_texts[index]

        def __len__(self):
            return len(self.data)
        
def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # The book originally contained this unnecessary "else" clause:
    #else:
    #    with open(file_path, "r", encoding="utf-8") as file:
    #        text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

if __name__ == "__main__":
    model_name = "gpt2-medium (355M)"
    NEW_CONFIG = BASE_CONFIG.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = download_and_load_file(file_path, url)
    
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    model = GPTModel(NEW_CONFIG)
    model.to(device)
    # Load with error handling
    try:
        checkpoint = torch.load("gptModel-355M.pth", map_location=device)
        model.load_state_dict(checkpoint)
        print("Pretrained weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    #checkpoint = torch.load("gpt2-medium355M-instruction-sft.pth", map_location=device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #start_epoch = checkpoint['epoch']
    #previous_losses = checkpoint['train_losses']

    num_epochs = 1
    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    file_name = f"{re.sub(r'[ ()]', '', model_name) }-instruction-sft.pth"
    # Save complete checkpoint for resuming training
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tokens_seen': tokens_seen,
        'config': NEW_CONFIG,
        'model_name': model_name,
        'execution_time_minutes': execution_time_minutes,
        'batch_size': batch_size,
        'learning_rate': 0.00005
    }
    
    torch.save(checkpoint, file_name)
    print(f"Complete checkpoint saved as {file_name}")
    
    # Also save just model weights for inference
    model_only_file = f"{re.sub(r'[ ()]', '', model_name)}-instruction-sft-model-only.pth"
    torch.save(model.state_dict(), model_only_file)
    print(f"Model weights only saved as {model_only_file}")
