"""
Usage Example:

python cnn_img_classification.py \
--batch_size 64 \
--learning_rate 0.001 \
--epochs 20 \
--data_dir './data' \
--output_file 'training_results.pdf'
"""

import argparse
import logging
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ndlinear import NdLinear


def get_args():
    parser = argparse.ArgumentParser(description="HyperMLP Training Script")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for the CIFAR-10 dataset')
    parser.add_argument('--output_file', type=str, default='training_results.pdf',
                        help='Output file for saving training results')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def get_compute_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        logging.info("CUDA and MPS not available; using CPU instead.")
        return torch.device("cpu")

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def load_data(transform, data_dir, batch_size):
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class HyperVision(nn.Module):
    def __init__(self, input_shape, hidden_size, num_classes):
        super(HyperVision, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.ndlinear = NdLinear((64, 8, 8), hidden_size)
        final_dim = math.prod(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, num_classes),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.ndlinear(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def initialize_model(device, model_class, *args):
    model = model_class(*args).to(device)
    return model

def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(trainloader)

def evaluate(model, testloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def plot_and_save(losses, accuracies, params, epochs, filename, model_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, label=f"{model_name} (Params: {params})", linestyle="solid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracies, label=f"{model_name} Accuracy", linestyle="solid")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    args = get_args()
    setup_logging()

    device = get_compute_device()
    transform = get_transform()
    trainloader, testloader = load_data(transform, args.data_dir, args.batch_size)

    # Initialize HyperVision model and optimizer
    hyper_vision = initialize_model(device, HyperVision, (3, 32, 32), (64, 8, 8), 10)
    optimizer_hyper = get_optimizer(hyper_vision, args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    hyper_losses, hyper_acc = [], []
    params_hyper = count_parameters(hyper_vision)
    logging.info(f"# Parameters - HyperVision: {params_hyper}")

    for epoch in range(args.epochs):
        loss_hyper = train(hyper_vision, trainloader, criterion, optimizer_hyper, device)
        hyper_losses.append(loss_hyper)

        acc_hyper = evaluate(hyper_vision, testloader, device)
        hyper_acc.append(acc_hyper)

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"HyperVision Loss: {loss_hyper:.4f}, Acc: {acc_hyper:.2f}%")

    plot_and_save(hyper_losses, hyper_acc, params_hyper, args.epochs, args.output_file, "HyperVision")

    # Initialize CNN model and optimizer
    cnn = initialize_model(device, CNN, 10)
    optimizer_cnn = get_optimizer(cnn, args.learning_rate)

    cnn_losses, cnn_acc = [], []
    params_cnn = count_parameters(cnn)
    logging.info(f"# Parameters - CNN: {params_cnn}")

    for epoch in range(args.epochs):
        loss_cnn = train(cnn, trainloader, criterion, optimizer_cnn, device)
        cnn_losses.append(loss_cnn)

        acc_cnn = evaluate(cnn, testloader, device)
        cnn_acc.append(acc_cnn)

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"CNN Loss: {loss_cnn:.4f}, Acc: {acc_cnn:.2f}%")

    plot_and_save(cnn_losses, cnn_acc, params_cnn, args.epochs, "cnn_" + args.output_file, "CNN")

if __name__ == "__main__":
    main()
    