# %%
from huggingface_hub import login
token = "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"
login(token=token)

# %%
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-mamba-7b")
tokenizer.pad_token = tokenizer.eos_token

# %%
device = 'cuda:5'

# %%
from peft import LoraConfig, get_peft_model
import torch

lora_config = LoraConfig(
    r=4,                     # Rank of the LoRA update matrices
    lora_alpha=32,            # Scaling factor
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],  # Mamba projection layers
    lora_dropout=0.1,         # Dropout probability
    bias="none",              # Whether to include bias terms
    task_type="CAUSAL_LM"     # Type of task
)

# Prepare model for LoRA fine-tuning
model = get_peft_model(model, lora_config).to(device)
print("LoRA model ready for fine-tuning.")
model.print_trainable_parameters()

# %%
epochs = 30
batch_size = 8

# %%
# Load CNN/DailyMail dataset
import pandas as pd
train_data = pd.read_csv('cnn_dailymail/train.csv')
valid_data = pd.read_csv('cnn_dailymail/validation.csv')

# Custom Dataset Class
from torch.utils.data import DataLoader

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]["article"]
        target_text = self.data.iloc[idx]["highlights"]
        input_ids = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return [input_ids.input_ids[0], input_ids.attention_mask[0], labels.input_ids[0]]

train_dataset = TextDataset(tokenizer, train_data)
valid_dataset = TextDataset(tokenizer, valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# %%
learning_rate = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Add padding token ID to model config
model.config.pad_token_id = tokenizer.pad_token_id

# %%
# Initialize mixed precision scaler
scaler = GradScaler()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Enable mixed precision for the forward pass
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        
        # Update model parameters (with scaled gradients)
        scaler.step(optimizer)
        
        # Update the scaler for the next iteration
        scaler.update()
        
        total_loss += loss.item()
        if i % 20 == 0:
            print(f"Batch {i} loss: {loss.item()}")

        if i % 100 == 0:
            # Generate output after a few steps for inspection
            output = model.generate(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), max_new_tokens=256)
            print("INPUT: ", tokenizer.decode(batch[0][0], skip_special_tokens=True))
            print("OUTPUT: ", tokenizer.decode(output[0], skip_special_tokens=True))

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Enable mixed precision for the validation step
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            val_loss += loss.item()

    print(f"Validation Loss after epoch {epoch+1}: {val_loss/len(valid_loader)}")


# %%
# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_falcon_mamba_summary')
tokenizer.save_pretrained('./fine_tuned_falcon_mamba_tokenizer_summary')
