# %%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from datasets import load_dataset

# Dataset class
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

# Fine-tuning function
def fine_tune_bert(model, train_loader, val_loader, epochs=3, learning_rate=5e-5, device="cuda"):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    model.to(device)
    for epoch in range(epochs):
        # Training loop
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

# Load IMDB dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Prepare DataLoaders
train_data = TextClassificationDataset(
    tokenizer,
    dataset["train"]["text"],
    dataset["train"]["label"],
    max_length=128,
)
test_data = TextClassificationDataset(
    tokenizer,
    dataset["test"]["text"],
    dataset["test"]["label"],
    max_length=128,
)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Fine-tune the model
fine_tune_bert(model, train_loader, test_loader)

#save model
model.save_pretrained('./fine_tuned_lora_model_imdb_bert')
tokenizer.save_pretrained('./fine_tuned_lora_tokenizer_imdb_bert')