import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ndlinear import NdLinear

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path, feat_len=96, pred_len=24):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    data = df.values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    sequences, targets = [], []
    for i in range(len(data) - feat_len - pred_len):
        sequences.append(data[i:i + feat_len, :-1])
        targets.append(data[i + feat_len:i + feat_len + pred_len, -1])

    return np.array(sequences), np.array(targets), scaler


class NdTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, custom_ffn, **kwargs):
        kwargs['batch_first'] = True
        super(NdTransformerEncoderLayer, self).__init__(d_model, nhead, **kwargs)
        self.nd_ffn = custom_ffn

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.nd_ffn(x)
        return x


class NdFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1, activation='relu'):
        super(NdFeedForward, self).__init__()
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layer1 = NdLinear((input_dim, 1), (hidden_dim, 1))
        self.activation = activations[activation]
        self.dropout = nn.Dropout(dropout)
        self.layer2 = NdLinear((hidden_dim, 1), (input_dim, 1))

    def forward(self, x):
        x_dims = list(x.shape)
        x = x.reshape(x_dims[0] * x_dims[1], x_dims[2], 1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = x.reshape(x_dims[0], x_dims[1], x_dims[2])
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=2, num_layers=3,
                 hidden_dim=256, dropout=0.1, activation='relu', target_len=24):
        super(TimeSeriesTransformer, self).__init__()
        hidden_dim = hidden_dim
        model_dim = model_dim
        ff_layer = NdFeedForward(model_dim, hidden_dim=hidden_dim, dropout=dropout, activation=activation)
        encoder_layer = NdTransformerEncoderLayer(model_dim, num_heads, custom_ffn=ff_layer, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, target_len)
        self.embedding = nn.Linear(input_dim, model_dim)
        self.target_len = target_len

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output[:, -1, :])
        return output


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0
    predictions = []
    targets_lst = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            targets_lst.append(targets.cpu().numpy())
    predictions = np.vstack(predictions)
    targets_lst = np.vstack(targets_lst)
    mse = mean_squared_error(targets_lst, predictions)
    mae = mean_absolute_error(targets_lst, predictions)
    return val_loss / len(data_loader), mse, mae


def log_metrics(best_mse, best_mae):
    logger.info(f"Best MSE: {best_mse}")
    logger.info(f"Best MAE: {best_mae}")


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, filepath='best_model.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    best_mse = float('inf')
    best_mae = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        val_loss, mse, mae = evaluate_model(model, val_loader, criterion, device)
        logger.info({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'eval_loss': val_loss,
            'mse': mse,
            'mae': mae
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mse = mse
            best_mae = mae
            torch.save(model.state_dict(), filepath)
            logger.info(f"Model checkpoint saved at epoch {epoch + 1}")

    log_metrics(best_mse, best_mae)


def main(args):
    args.lr = 0.015  # Adjusted learning rate for the experiment
    train_data, train_targets, scaler = load_data(args.file_path, args.feat_len, args.pred_len)
    train_size = int(0.5 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_targets)),
        [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = train_data.shape[-1]
    model = TimeSeriesTransformer(input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
                                  num_layers=args.n_layers, activation=args.activation,
                                  model_dim=args.model_dim, target_len=args.pred_len)
    logger.info(f"NdLinear model: {model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {total_params}")

    train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, filepath=args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type=str, default='data/ett/ETTm2.csv', help='Path to CSV data file')
    parser.add_argument('--feat-len', type=int, default=24, help='Sequence length for input data')
    parser.add_argument('--pred-len', type=int, default=12, help='Length of prediction to make')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension of feedforward layers')
    parser.add_argument('--model-dim', type=int, default=32, help='Model dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'tanh', 'sigmoid', 'gelu'])
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--n-layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--model-path', type=str, default='best_model.pth', help='File path to save the best model')
    args = parser.parse_args()

    main(args)
