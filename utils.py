import torch
import matplotlib.pyplot as plt

def train(model, train_data, val_data, optimizer, criterion, device, epochs=10, is_rnn = False):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()

        train_loss = 0
        train_acc = 0
        train_length = 0

        for i, batch in enumerate(train_data):
            x, mask, y = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output = model(x, mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (output.argmax(1) == y).sum().item()
            train_length += len(y)
            # print(f"Epoch {epoch} Batch {i} loss: {loss.item()}")
        val_loss, val_acc = evaluate(model, val_data, criterion, device)
        train_accs.append(train_acc/train_length)
        val_accs.append(val_acc)
        train_losses.append(train_loss/len(train_data))
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: Train loss: {train_loss/len(train_data)}, Train acc: {train_acc/train_length}, Val loss: {val_loss}, Val acc: {val_acc}")
    model = 'rnn' if is_rnn else 'LoRA'
    plot_accuracy(train_accs, val_accs, model)
    plot_loss(train_losses, val_losses, model)

def train_distil(teacher_model, student_model, train_data, val_data, optimizer, criterion, distil_criterion, device, epochs=10, alpha=0.5):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        student_model.train()

        train_loss = 0
        train_acc = 0
        train_length = 0

        for batch in train_data:
            x, mask, y = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            output = student_model(x, mask)
            # output = torch.argmax(output, dim=1)
            teacher_output = teacher_model(x, mask)
            teacher_output = torch.nn.functional.softmax(teacher_output, dim=1)

            student_loss = criterion(output, y)
            distillation_loss = distil_criterion(output, teacher_output)
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (output.argmax(1) == y).sum().item()
            train_length += len(y)
        val_loss, val_acc = evaluate(student_model, val_data, criterion, device)
        train_accs.append(train_acc/train_length)
        val_accs.append(val_acc)
        train_losses.append(train_loss/len(train_data))
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: Train loss: {train_loss/len(train_data)}, Train acc: {train_acc/train_length}, Val loss: {val_loss}, Val acc: {val_acc}")
    plot_accuracy(train_accs, val_accs, 'Distil')
    plot_loss(train_losses, val_losses, 'Distil')

def evaluate(model, data, criterion, device):
    model.eval()
    total_loss = 0
    accuracy = 0
    data_size = 0
    with torch.no_grad():
        for batch in data:
            x, mask, y = batch
            x, y = x.to(device), y.to(device)
            mask = mask.to(device)
            output = model(x, mask)
            loss = criterion(output, y)
            accuracy += (output.argmax(1) == y).sum().item()
            data_size += len(y)
            total_loss += loss.item()
    return total_loss / len(data), accuracy / data_size
    
def plot_accuracy(train_acc, val_acc, model_name):
    plt.figure()
    x = range(1, len(train_acc)+1)
    plt.plot(x, train_acc, label='Train Accuracy')
    plt.plot(x, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(model_name + ' - Accuracy')
    plt.legend()
    plt.savefig(model_name + ' - accuracy.png')

def plot_loss(train_loss, val_loss, model_name):
    plt.figure()
    x = range(1, len(train_loss)+1)
    plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model_name + ' - Loss')
    plt.legend()
    plt.savefig(model_name + ' - loss.png')