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
reg_csv_path = "datasets/delivery_time.csv"
pipeline_classification = TabularDataPipeline(reg_csv_path, task="regression")
train_data, test_data = pipeline_classification.prepare_dataloader(batch_size=32)


reg_output_dim = 1  # For regression, output is a continuous value
# Linear Model
reg_input_dim = 14
reg_hidden_dim = 128
reg_linear_model = LinearModel(reg_input_dim, reg_hidden_dim, reg_output_dim)
print("Number of Params for Linear Model: ", sum(p.numel() for p in reg_linear_model.parameters() if p.requires_grad))

# NdLinear Model
reg_ndlinear_input_dims = [14, 1]
reg_ndlinear_hidden_dims = [32, 64]
reg_ndlinear_model = NdLinearModel(reg_ndlinear_input_dims, reg_ndlinear_hidden_dims, reg_output_dim)
print("Number of Params for NdLinear Model: ", sum(p.numel() for p in reg_ndlinear_model.parameters() if p.requires_grad))

reg_criterion = nn.MSELoss()

# -------------------------
# Training & Evaluation Functions
# -------------------------

experiment_times = 5
experiments_train_loss = []
experiments_test_loss = []

for exp_idx in range(experiment_times):
    reg_linear_model = LinearModel(reg_input_dim, reg_hidden_dim, reg_output_dim).to(device)
    reg_ndlinear_model = NdLinearModel(reg_ndlinear_input_dims, reg_ndlinear_hidden_dims, reg_output_dim).to(device)
    
    reg_linear_optimizer = optim.AdamW(reg_linear_model.parameters(), lr=0.0002)
    reg_ndlinear_optimizer = optim.AdamW(reg_ndlinear_model.parameters(), lr=0.0002)

    num_epochs = 40
    reg_linear_train_loss = []
    reg_ndlinear_train_loss = []
    reg_linear_test_loss = []
    reg_ndlinear_test_loss = []

    for epoch in range(num_epochs):
        
        reg_linear_running_loss = 0.0
        reg_ndlinear_running_loss = 0.0
        
        for batch_X, batch_y in train_data:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # training base model
            reg_linear_optimizer.zero_grad()
            reg_linear_outputs = reg_linear_model(batch_X)
            reg_linear_loss = reg_criterion(reg_linear_outputs.squeeze(-1), batch_y)
            reg_linear_loss.backward()
            reg_linear_optimizer.step()
            reg_linear_running_loss += reg_linear_loss.item()
            
            # training revised model
            reg_ndlinear_optimizer.zero_grad()
            reg_ndlinear_outputs = reg_ndlinear_model(batch_X)
            reg_ndlinear_loss = reg_criterion(reg_ndlinear_outputs.squeeze(-1), batch_y)
            reg_ndlinear_loss.backward()
            reg_ndlinear_optimizer.step()
            reg_ndlinear_running_loss += reg_ndlinear_loss.item()
        
        avg_reg_linear_loss = reg_linear_running_loss / len(train_data)
        avg_reg_ndlinear_loss = reg_ndlinear_running_loss / len(train_data)
        
        reg_linear_train_loss.append(avg_reg_linear_loss)
        reg_ndlinear_train_loss.append(avg_reg_ndlinear_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Linear Loss: {avg_reg_linear_loss:.4f}, NdLinear Loss: {avg_reg_ndlinear_loss:.4f}")
        
        reg_linear_eval_loss = 0.0
        reg_ndlinear_eval_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_data:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
            
                # testing Base model
                reg_linear_outputs = reg_linear_model(batch_X)
                reg_linear_eval_loss += reg_criterion(reg_linear_outputs.squeeze(-1), batch_y).item()
                
                # testing revised model
                reg_ndlinear_outputs = reg_ndlinear_model(batch_X)
                reg_ndlinear_eval_loss += reg_criterion(reg_ndlinear_outputs.squeeze(-1), batch_y).item()
        
        reg_linear_eval_loss = reg_linear_eval_loss / len(test_data)
        reg_ndlinear_eval_loss = reg_ndlinear_eval_loss / len(test_data)
        
        print(len(test_data.dataset) / len(test_data))
        
        reg_linear_test_loss.append(reg_linear_eval_loss)
        reg_ndlinear_test_loss.append(reg_ndlinear_eval_loss)
        
        print(f"Test Loss -> Linear: {reg_linear_eval_loss:.4f} | NdLinear: {reg_ndlinear_eval_loss:.4f}")
        
    experiments_train_loss.append((reg_linear_train_loss, reg_ndlinear_train_loss))
    experiments_test_loss.append((reg_linear_test_loss, reg_ndlinear_test_loss))



# -------------------------
# Plotting the Results
# -------------------------

# Determine the number of epochs (assuming all experiments have the same number)
num_epochs = len(experiments_train_loss[0][0])
epochs = np.arange(1, num_epochs + 1)

# --- Compute Mean Training Loss ---
train_loss_linear_all = np.array([exp[0] for exp in experiments_train_loss])
train_loss_ndlinear_all = np.array([exp[1] for exp in experiments_train_loss])
mean_train_loss_linear = np.mean(train_loss_linear_all, axis=0)
mean_train_loss_ndlinear = np.mean(train_loss_ndlinear_all, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_train_loss_linear, label='Linear Model Train Loss')
plt.plot(epochs, mean_train_loss_ndlinear, label='NdLinear Model Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Training Loss over 50 Experiments (Regression)')
plt.legend()
plt.savefig('training_loss_regression.png')
plt.close()

# --- Compute Mean Test Loss ---
test_loss_linear_all = np.array([exp[0] for exp in experiments_test_loss])
test_loss_ndlinear_all = np.array([exp[1] for exp in experiments_test_loss])
mean_test_loss_linear = np.mean(test_loss_linear_all, axis=0)
mean_test_loss_ndlinear = np.mean(test_loss_ndlinear_all, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_test_loss_linear, label='Linear Model Test Loss')
plt.plot(epochs, mean_test_loss_ndlinear, label='NdLinear Model Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mean Test Loss over 50 Experiments (Regression)')
plt.legend()
plt.savefig('test_loss_regression.png')
plt.close()

print("Plots saved successfully as 'training_loss_regression.png' and 'test_loss_regression.png'.")