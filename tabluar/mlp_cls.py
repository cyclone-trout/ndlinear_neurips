import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

from tabular_data import TabularDataPipeline
from ndlinear import NdLinear

import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # This will print whether GPU or CPU is being used

class LinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        
        x = self.fc(x)
        return x
    
class NdLinearModel(nn.Module):
    def __init__(self, ndlinear_input_dims, ndlinear_hidden_dims, num_classes):
        super(NdLinearModel, self).__init__()
        self.ndlinear1 = NdLinear(ndlinear_input_dims, ndlinear_hidden_dims)
        self.ndlinear2 = NdLinear(ndlinear_hidden_dims, ndlinear_hidden_dims)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        hidden_dim = math.prod(ndlinear_hidden_dims)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.ndlinear1(x)
        x = self.relu1(x)
        x = self.ndlinear2(x)
        x = self.relu2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
print("--- Conducting Classification Task on Tabular Data ---")
cls_csv_path = "datasets/cardio_disease.csv"
pipeline_classification = TabularDataPipeline(cls_csv_path, task="classification")
train_data, test_data = pipeline_classification.prepare_dataloader(batch_size=32)


cls_num_classes = 2
# Linear Model
cls_input_dim = 11
cls_hidden_dim = 128
cls_linear_model = LinearModel(cls_input_dim, cls_hidden_dim, cls_num_classes)
print("Number of Params for Linear Model: ", sum(p.numel() for p in cls_linear_model.parameters() if p.requires_grad))

# NdLinear Model
cls_ndlinear_input_dims = [11, 1]
cls_ndlinear_hidden_dims = [11, 64]
cls_ndlinear_model = NdLinearModel(cls_ndlinear_input_dims, cls_ndlinear_hidden_dims, cls_num_classes)
print("Number of Params for NdLinear Model: ", sum(p.numel() for p in cls_ndlinear_model.parameters() if p.requires_grad))


cls_criterion = nn.CrossEntropyLoss()

# -------------------------
# Training & Evaluation Functions
# -------------------------

experiment_times = 5
experiments_train_loss = []
experiments_test_acc = []

for exp_idx in range(experiment_times):
    # Initialize models from scratch for each experiment
    cls_linear_model = LinearModel(cls_input_dim, cls_hidden_dim, cls_num_classes).to(device)
    cls_ndlinear_model = NdLinearModel(cls_ndlinear_input_dims, cls_ndlinear_hidden_dims, cls_num_classes).to(device)
    
    # Reinitialize optimizers for the new models
    cls_linear_optimizer = optim.AdamW(cls_linear_model.parameters(), lr=0.0001)
    cls_ndlinear_optimizer = optim.AdamW(cls_ndlinear_model.parameters(), lr=0.0001)
    
    num_epochs = 40
    cls_linear_train_loss = []
    cls_ndlinear_train_loss = []
    cls_linear_test_acc = []
    cls_ndlinear_test_acc = []

    for epoch in range(num_epochs):
        
        cls_linear_running_loss = 0.0
        cls_ndlinear_running_loss = 0.0
        
        for batch_X, batch_y in train_data:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Training for Linear Model
            cls_linear_optimizer.zero_grad()
            cls_linear_outputs = cls_linear_model(batch_X)
            targets = batch_y.argmax(dim=1)
            cls_linear_loss = cls_criterion(cls_linear_outputs, targets)
            cls_linear_loss.backward()
            cls_linear_optimizer.step()
            cls_linear_running_loss += cls_linear_loss.item()
            
            # Training for NdLinear Model
            cls_ndlinear_optimizer.zero_grad()
            cls_ndlinear_outputs = cls_ndlinear_model(batch_X)
            targets = batch_y.argmax(dim=1)
            cls_ndlinear_loss = cls_criterion(cls_ndlinear_outputs, targets)
            cls_ndlinear_loss.backward()
            cls_ndlinear_optimizer.step()
            cls_ndlinear_running_loss += cls_ndlinear_loss.item()
        
        avg_cls_linear_loss = cls_linear_running_loss / len(train_data)
        avg_cls_ndlinear_loss = cls_ndlinear_running_loss / len(train_data)
        
        cls_linear_train_loss.append(avg_cls_linear_loss)
        cls_ndlinear_train_loss.append(avg_cls_ndlinear_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Linear Loss: {avg_cls_linear_loss:.4f}, NdLinear Loss: {avg_cls_ndlinear_loss:.4f}")
        
        cls_linear_accuracy = 0.0
        cls_ndlinear_accuracy = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_data:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # testing Base model
                cls_linear_outputs = cls_linear_model(batch_X)
                cls_linear_accuracy += (torch.argmax(cls_linear_outputs, dim=1) == torch.argmax(batch_y, dim=1)).float().sum()
                
                # testing revised model
                cls_ndlinear_outputs = cls_ndlinear_model(batch_X)
                cls_ndlinear_accuracy += (torch.argmax(cls_ndlinear_outputs, dim=1) == torch.argmax(batch_y, dim=1)).float().sum()
        
        cls_linear_accuracy = cls_linear_accuracy / len(test_data.dataset)
        cls_ndlinear_accuracy = cls_ndlinear_accuracy / len(test_data.dataset)
        
        cls_linear_test_acc.append(cls_linear_accuracy)
        cls_ndlinear_test_acc.append(cls_ndlinear_accuracy)
        
        print(f"Linear Accuracy: {cls_linear_accuracy.item()*100:.2f}% | NdLinear Accuracy: {cls_ndlinear_accuracy.item()*100:.2f}%")
        
    experiments_train_loss.append((cls_linear_train_loss, cls_ndlinear_train_loss))
    experiments_test_acc.append((cls_linear_test_acc, cls_ndlinear_test_acc))



# Assuming experiments_train_loss and experiments_test_acc are available.
# Each entry in experiments_train_loss is a tuple: (cls_linear_train_loss, cls_ndlinear_train_loss)
# Each entry in experiments_test_acc is a tuple: (cls_linear_test_acc, cls_ndlinear_test_acc)

# Determine the number of epochs from one experiment (assuming all experiments have the same number)
num_epochs = len(experiments_train_loss[0][0])
epochs = np.arange(1, num_epochs + 1)

# --- Compute Mean Training Loss ---

# Convert the training loss lists to numpy arrays for easier averaging
train_loss_linear_all = np.array([exp[0] for exp in experiments_train_loss])
train_loss_ndlinear_all = np.array([exp[1] for exp in experiments_train_loss])

# Compute mean training loss for each epoch (averaging across the 50 experiments)
mean_train_loss_linear = np.mean(train_loss_linear_all, axis=0)
mean_train_loss_ndlinear = np.mean(train_loss_ndlinear_all, axis=0)

# Plot the mean training loss for both models
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_train_loss_linear, label='Linear Model Train Loss')
plt.plot(epochs, mean_train_loss_ndlinear, label='NdLinear Model Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Training Loss over 50 Experiments')
plt.legend()
plt.savefig('training_loss_classification.png')
plt.close()

# --- Compute Mean Test Accuracy ---

# Convert the test accuracy lists to numpy arrays for easier averaging
test_acc_linear_all = np.array([exp[0] for exp in experiments_test_acc])
test_acc_ndlinear_all = np.array([exp[1] for exp in experiments_test_acc])

# Compute mean test accuracy for each epoch (averaging across the 50 experiments)
mean_test_acc_linear = np.mean(test_acc_linear_all, axis=0)
mean_test_acc_ndlinear = np.mean(test_acc_ndlinear_all, axis=0)

# Plot the mean test accuracy for both models (multiplied by 100 to get percentage values)
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_test_acc_linear * 100, label='Linear Model Test Accuracy')
plt.plot(epochs, mean_test_acc_ndlinear * 100, label='NdLinear Model Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Mean Test Accuracy over 50 Experiments')
plt.legend()
plt.savefig('test_accuracy_classification.png')
plt.close()

print("Plots saved successfully as 'training_loss_classification.png' and 'test_accuracy_classification.png'.")