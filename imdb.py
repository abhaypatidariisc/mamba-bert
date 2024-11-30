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
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load IMDB dataset
dataset = load_dataset("imdb")

# Set tokenizer padding token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Preprocess the data
def preprocess_data(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

# Tokenize the data
max_length = 128
train_data = dataset["train"].map(lambda x: preprocess_data(x, tokenizer, max_length), batched=True)
test_data = dataset["test"].map(lambda x: preprocess_data(x, tokenizer, max_length), batched=True)

# Convert to PyTorch tensors and create DataLoaders
def format_data(data):
    input_ids = torch.stack([x["input_ids"] for x in data])
    attention_mask = torch.stack([x["attention_mask"] for x in data])
    labels = torch.tensor(data["label"])
    return torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

train_dataset = format_data(train_data)
test_dataset = format_data(test_data)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Add padding token to model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# %%
# Training loop
epochs = 10
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
model.save_pretrained('./fine_tuned_lora_model_imdb')
tokenizer.save_pretrained('./fine_tuned_lora_tokenizer_imdb')

# %%
