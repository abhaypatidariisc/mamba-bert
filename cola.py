# %%
from huggingface_hub import login
token = "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"
login(token=token)

# %%
# Load model and tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = AutoModelForSequenceClassification.from_pretrained("tiiuae/falcon-mamba-7b", num_labels=2)

# %%
device = 'cuda:5'

# %%
from peft import LoraConfig, get_peft_model
import torch

# Falcon-Mamba LoRA configuration
lora_config = LoraConfig(
    r=4,                     # Rank of the LoRA update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],  # Falcon-Mamba projection layers
    lora_dropout=0.1,        # Dropout probability
    bias="none",             # Exclude biases from LoRA updates
    task_type="SEQ_CLS"      # Task type: Sequence Classification
)

# Prepare model for LoRA fine-tuning
model = get_peft_model(model, lora_config).to(device)
print("LoRA model ready for fine-tuning.")
model.print_trainable_parameters()

# %%
import numpy as np
def get_data_loader(data_path, batch_size, tokenizer, shuffle=True, max_len=50):
    """
    Create a DataLoader for a TSV dataset.
    """
    data = np.loadtxt(data_path, delimiter='\t', dtype=str)
    X, y = data[:, -1], data[:, 1]
    X = tokenizer.batch_encode_plus(
        X.tolist(), max_length=max_len, truncation=True, padding='max_length')
    X, mask = X['input_ids'], X['attention_mask']
    X = torch.tensor(np.array(X))
    mask = torch.tensor(np.array(mask))
    y = torch.tensor(np.array(y, dtype=int))
    dataset = torch.utils.data.TensorDataset(X, mask, y)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return data_loader

# %%
from datasets import load_dataset
from torch.utils.data import DataLoader

# Set the padding token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

batch_size = 128
learning_rate = 2e-5
epochs = 10

# Load dataset
train_loader = get_data_loader(
        'data/in_domain_train.tsv', batch_size, tokenizer)
val_loader = get_data_loader(
        'data/in_domain_dev.tsv', batch_size, tokenizer, shuffle=False)

# %%
# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Add padding token to model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# %%
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_acc = (outputs.logits.argmax(dim=1) == labels).float().mean()
        correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
        
        total_loss += loss.item()
        if i % 20 == 0:
            print(f"Batch {i} loss: {loss.item()}, accuracy: {train_acc.item()}")

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
    print(f"Accuracy after epoch {epoch+1}: {correct/len(train_loader.dataset)}")

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

    print(f"Validation Loss after epoch {epoch+1}: {val_loss/len(val_loader)}")
    print(f"Validation accuracy after epoch {epoch+1}: {correct/len(val_loader.dataset)}")

# %%
# Save the fine-tuned model
model.save_pretrained('./fine_tuned_lora_model_falcon_cola')
tokenizer.save_pretrained('./fine_tuned_lora_tokenizer_falcon_cola')

# %%
