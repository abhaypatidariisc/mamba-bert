# %%
from huggingface_hub import login
token = "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"
login(token=token)

# %%
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

# #set tokenizer padding_side
# tokenizer.padding_side = "left"

# %%
device = 'cuda:5'

# %%
from peft import LoraConfig, get_peft_model
import torch

lora_config = LoraConfig(
    r=4,                     # Rank of the LoRA update matrices
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],  # Projection layers
    lora_dropout=0.1,         # Dropout probability
    bias="none",              # Whether to include bias terms
    task_type="CAUSAL_LM"     # Type of task
)

# Prepare model for LoRA fine-tuning
model = get_peft_model(model, lora_config).to(device)
print("LoRA model ready for fine-tuning.")
model.print_trainable_parameters()

# %%
epochs = 5
batch_size = 8

# %%
#load data from csv
import pandas as pd
train_data = pd.read_csv('cnn_dailymail/train.csv')
valid_data = pd.read_csv('cnn_dailymail/validation.csv')

# train_data.head()

# %%
#convert to dataloader
from torch.utils.data import DataLoader

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data.iloc[idx]["article"]
        output = self.data.iloc[idx]["highlights"]
        input_ids = self.tokenizer(input, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(output, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return [input_ids.input_ids[0], input_ids.attention_mask[0], labels.input_ids[0]]

train_dataset = TextDataset(tokenizer, train_data)
valid_dataset = TextDataset(tokenizer, valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# %%
learning_rate = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# add pading token in config
model.config.pad_token_id = tokenizer.pad_token_id

# %%
batch_train = next(iter(train_loader))
batch_valid = next(iter(valid_loader))

# %%
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # print(input_ids.shape, labels.shape, outputs.shape)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if i % 20 == 0:
            print(f"Batch {i} loss: {loss.item()}")

        if i % 100 == 0:
            #genearte output
            output = model.generate(input_ids=batch_train[0].to(device), attention_mask=batch_train[1].to(device), max_new_tokens=512)
            print(tokenizer.decode(output[0], skip_special_tokens=True))

            output = model.generate(input_ids=batch_valid[0].to(device), attention_mask=batch_valid[1].to(device), max_new_tokens=512)
            print(tokenizer.decode(output[0], skip_special_tokens=True))

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")


    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    print(f"Validation Loss after epoch {epoch+1}: {val_loss/len(valid_loader)}")

# %%
# Saving the model after training
model.save_pretrained('./fine_tuned_lora_model_summary')
tokenizer.save_pretrained('./fine_tuned_lora_tokenizer_summary')