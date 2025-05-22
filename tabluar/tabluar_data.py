import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class TabularDataPipeline:
    def __init__(self, csv_path, task='classification'):
        """
        Args:
            csv_path (str): Path to the CSV file.
            task (str): Either 'classification' or 'regression'. Default is 'classification'.
        """
        self.csv_path = csv_path
        task = task.lower()
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")
        self.task = task
        
    def prepare_dataloader(self, batch_size=32):
        # Load CSV data
        df = pd.read_csv(self.csv_path)
        data = df.values
        
        # Split features and labels (last column is assumed to be the label)
        X = data[:, :-1]
        y = data[:, -1]
        
        # Process labels based on task type
        if self.task == 'classification':
            unique_labels, y_indices = np.unique(y, return_inverse=True)
            y = np.eye(len(unique_labels))[y_indices]  # One-hot encoding
        else: # Regression
            y = y.astype(np.float32)
        
        # Shuffle indices and split into training (80%) and testing (20%) sets
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        train_size = int(0.8 * n_samples)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Convert NumPy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Create DataLoader objects for training and testing sets
        train_data = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                batch_size=batch_size, shuffle=True)
        test_data = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                               batch_size=batch_size, shuffle=False)
        
        return train_data, test_data

if __name__ == "__main__":
    # For classification (labels will be one-hot encoded)
    cls_csv_path = "datasets/cardio_disease.csv"
    
    pipeline_classification = TabularDataPipeline(cls_csv_path, task="classification")
    train_data, test_data = pipeline_classification.prepare_dataloader(batch_size=32)
    
    for batch_X, batch_y in train_data:
        print("Classification - Train batch X shape:", batch_X.shape)
        print("Classification - Train batch y shape:", batch_y.shape)
        break
    
    
    # For regression (labels remain as continuous values)
    reg_csv_path = "datasets/delivery_time.csv"
    
    pipeline_regression = TabularDataPipeline(reg_csv_path, task="regression")
    train_data, test_data = pipeline_regression.prepare_dataloader(batch_size=32)
    
    for batch_X, batch_y in train_data:
        print("Regression - Train batch X shape:", batch_X.shape)
        print("Regression - Train batch y shape:", batch_y.shape)
        break