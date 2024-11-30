# %%
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Hugging Face Hub login
token = "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"
login(token=token)

# %%
# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set device
device = 'cuda:5'

# %%
# Dataset loader
class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# %%
# Load data from a TSV file
def load_data(file_path, tokenizer, max_len, batch_size, shuffle=True):
    data = np.loadtxt(file_path, delimiter='\t', dtype=str)
    texts = data[:, -1]  # Texts in the last column
    labels = data[:, 1].astype(int)  # Labels in the second column
    dataset = TextClassificationDataset(tokenizer, texts, labels, max_length=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

batch_size = 32
max_len = 128
train_loader = load_data('data/in_domain_train.tsv', tokenizer, max_len, batch_size)
val_loader = load_data('data/in_domain_dev.tsv', tokenizer, max_len, batch_size, shuffle=False)

# %%
# Optimizer and criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Add padding token to tokenizer
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# %%
# Training loop
epochs = 10
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

    train_acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Validation loop
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            val_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# %%
# Save the fine-tuned model
model.save_pretrained('./fine_tuned_bert_cls')
tokenizer.save_pretrained('./fine_tuned_bert_tokenizer_cls')
