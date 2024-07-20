import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, save_checkpoint=False, checkpoint_dir="checkpoints", log_interval=100):
    """
    Trains a PyTorch model.
    
    Parameters:
    - model: torch.nn.Module, the model to train
    - train_loader: DataLoader, the data loader for the training data
    - val_loader: DataLoader, the data loader for the validation data
    - loss_fn: loss function, e.g., torch.nn.CrossEntropyLoss
    - optimizer: optimizer, e.g., torch.optim.SGD
    - num_epochs: int, number of epochs to train
    - device: torch.device, the device to train on (e.g., 'cuda' or 'cpu')
    - save_checkpoint: bool, whether to save model checkpoints (default: False)
    - checkpoint_dir: str, directory to save checkpoints (default: "checkpoints")
    - log_interval: int, how many batches to wait before logging training status (default: 100)
    """
    
    model.to(device)
    if save_checkpoint and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % log_interval == log_interval - 1:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / log_interval:.4f}')
                running_loss = 0.0
        
        #validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        
        if save_checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

    print('Training complete')



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#randomdata
X_train = torch.randn(800, 10)
y_train = torch.randint(0, 2, (800,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_val = torch.randn(200, 10)
y_val = torch.randint(0, 2, (200,))
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#randommodel
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#checkgpu (need to add mps fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#call function
train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10, device=device, save_checkpoint=True, checkpoint_dir="checkpoints", log_interval=10)
