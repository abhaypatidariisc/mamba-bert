# %%
from huggingface_hub import login
token = "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"
login(token=token)

# %%
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b")
model = AutoModelForSequenceClassification.from_pretrained("tiiuae/falcon-mamba-7b", num_labels=5)
tokenizer.pad_token = tokenizer.eos_token

# %%
device = 'cuda:5'

# %%
from peft import LoraConfig, get_peft_model
import torch

# Configure LoRA with Falcon-Mamba specific projection layers
lora_config = LoraConfig(
    r=4,                     # Rank of the LoRA update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],  # Falcon-Mamba layers
    lora_dropout=0.1,        # Dropout probability
    bias="none",             # Whether to include bias terms
    task_type="SEQ_CLS"      # Task type is sequence classification
)

# Prepare model for LoRA fine-tuning
model = get_peft_model(model, lora_config).to(device)
print("LoRA model ready for fine-tuning.")
model.print_trainable_parameters()

# %%
# Load data from CSV
import pandas as pd
train_data = pd.read_csv('Corona_NLP_train.csv', encoding="ISO-8859-1")
valid_data = pd.read_csv('Corona_NLP_test.csv')

# Drop unnecessary columns
train_data = train_data.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])
valid_data = valid_data.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])

# Map sentiment to labels
sentiment_mapping = {
    "Extremely Negative": 0,
    "Negative": 1,
    "Neutral": 2,
    "Positive": 3,
    "Extremely Positive": 4
}
train_data["Label"] = train_data["Sentiment"].map(sentiment_mapping)
valid_data["Label"] = valid_data["Sentiment"].map(sentiment_mapping)

train_data.head()

# %%
batch_size = 50

# %%
# Convert to DataLoader
from torch.utils.data import DataLoader

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['OriginalTweet']
        label = self.data.iloc[idx]['Label']
        
        inputs = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        labels = torch.tensor(label, dtype=torch.long)
        
        return [inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze(), labels]

train_dataset = TextDataset(tokenizer, train_data)
valid_dataset = TextDataset(tokenizer, valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Add padding token to config
model.config.pad_token_id = tokenizer.pad_token_id
epochs = 10

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
        for batch in valid_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

    print(f"Validation Loss after epoch {epoch+1}: {val_loss/len(valid_loader)}")
    print(f"Validation accuracy after epoch {epoch+1}: {correct/len(valid_loader.dataset)}")

# %%
# Saving the model after training
model.save_pretrained('./fine_tuned_lora_model_falcon')
tokenizer.save_pretrained('./fine_tuned_lora_tokenizer_falcon')

# %%
