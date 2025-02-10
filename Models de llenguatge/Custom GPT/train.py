import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.optim import Adam
from transformers import BertTokenizer, Trainer, TrainingArguments
from gpt import GPTModel, create_dataloader_v1
from gpt_train import calc_loss_batch

def read_file(path): # Llegir el fitxer .txt
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

text = read_file('kamasutra.txt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Inicialitzar tokenizer
    
dataloader = create_dataloader_v1(text)

GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = GPTModel(GPT_CONFIG_124M).to(device)

optimizer = Adam(model.parameters(), lr=0.003)

criterion = nn.CrossEntropyLoss()

def main():
    epochs = 3
    total_loss = 0.0

    for epoch in range(epochs):
        model.train()

        for batch, (input_batch, target_batch) in enumerate(tqdm(dataloader)):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # Forward
            outputs = model(input_batch)

                # Reshape logits and target data to match the CrossEntropyLoss requirements
            batch_size, seq_len, vocab_size = outputs.shape
            logits = outputs.view(batch_size * seq_len, vocab_size)  # Shape: [batch_size * seq_len, vocab_size]
            targets = target_batch.view(-1)  # Shape: [batch_size * seq_len]

            loss = criterion(logits, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()   

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
