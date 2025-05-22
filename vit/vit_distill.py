"""
Usage Example:
    torchrun --nnodes 1 --nproc_per_node=4 \
    src/vit_distill.py \
    --num_epochs 30 \
    --num_transformers 6 \
    --dataset CIFAR100 \
    --batch_size 256 \
    --lr 2.75e-4
"""
import argparse
import json
import logging
from datetime import datetime

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Resize
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from vit import DistillableViT, DistillWrapper

logging.basicConfig(level=logging.INFO)


def init_cuda_dist():
    """Initialize CUDA and distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank


def setup_logger(args, local_rank):
    """Set up the logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"{args.dataset}_training.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if local_rank == 0:
        logger.info(f"Starting training with configuration: {vars(args)}")

    return logger


def get_dataset_transforms(selected_dataset):
    """Return dataset class, normalization values, and transformations based on dataset."""
    if selected_dataset == "CIFAR10":
        norm_val = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        num_classes = 10
        DatasetClass = datasets.CIFAR10
    else:
        norm_val = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        num_classes = 100
        DatasetClass = datasets.CIFAR100

    train_transform = transforms.Compose(
        [
            Resize((224, 224)),
            transforms.RandomCrop(224, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*norm_val, inplace=True),
        ]
    )
    val_transform = transforms.Compose(
        [
            Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(*norm_val),
        ]
    )

    return DatasetClass, num_classes, train_transform, val_transform


def create_dataloaders(DatasetClass, train_transform, val_transform, batch_size, data_root):
    """Create data loaders for training and validation."""
    train_data = DatasetClass(root=data_root, train=True, download=True, transform=train_transform)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    test_data = DatasetClass(root=data_root, train=False, download=True, transform=val_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size * 8, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_sampler


def build_models(num_classes, input_dim, num_transformers, temperature_value, alpha_value):
    """Build and return teacher and student models."""
    teacher = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    num_features = teacher.heads.head.in_features
    teacher.heads.head = nn.Linear(num_features, num_classes)

    student = DistillableViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=input_dim,
        depth=num_transformers,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.3
    )

    distiller = DistillWrapper(
        student=student,
        teacher=teacher,
        temperature=temperature_value,
        alpha=alpha_value,
        hard=False,
    )

    return distiller


def train_epoch(train_loader, distiller, optimizer, scheduler, scaler, device):
    """Train the model for one epoch and return average loss."""
    distiller.train()
    running_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc="Training", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):
            loss = distiller(imgs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(test_loader, student, device):
    """Validate the model and return top-1 and top-5 accuracy."""
    student.eval()
    correct_1 = correct_5 = total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast("cuda"):
                outputs = student(imgs)

            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()

            total += labels.size(0)

    accuracy_1 = 100.0 * correct_1 / total
    accuracy_5 = 100.0 * correct_5 / total
    return accuracy_1, accuracy_5


def save_checkpoint_and_results(selected_dataset, num_transformers, epoch_result, best_accuracy, distiller, local_rank,
                                logger, timestamp):
    """Save model checkpoint and log results if necessary."""
    if local_rank == 0:
        logger.info(epoch_result)

        save_filename = None
        is_best = False

        if selected_dataset == "CIFAR10" and epoch_result["accuracy@1"] > best_accuracy:
            best_accuracy = epoch_result["accuracy@1"]
            save_filename = f"torchvt_{selected_dataset.lower()}_{timestamp}_{num_transformers}tf.pth"
            is_best = True
        elif selected_dataset == "CIFAR100" and epoch_result["accuracy@5"] > best_accuracy:
            best_accuracy = epoch_result["accuracy@5"]
            save_filename = f"torchvt_{selected_dataset.lower()}_{timestamp}_{num_transformers}tf.pth"
            is_best = True

        if is_best:
            torch.save(distiller.state_dict(), save_filename)
            logger.info(
                f"Best model saved with Accuracy@1: {epoch_result['accuracy@1']:.2f}%, Accuracy@5: {epoch_result['accuracy@5']:.2f}% as {save_filename}")

    return best_accuracy


def train(args):
    device, local_rank = init_cuda_dist()
    logger = setup_logger(args, local_rank)
    DatasetClass, num_classes, train_transform, val_transform = get_dataset_transforms(args.dataset)
    train_loader, test_loader, train_sampler = create_dataloaders(DatasetClass, train_transform, val_transform,
                                                                  args.batch_size, args.data_root)

    distiller = build_models(num_classes, args.input_dim, args.num_transformers, args.temperature, args.alpha)
    distiller.to(device)
    distiller = nn.parallel.DistributedDataParallel(distiller, device_ids=[local_rank], find_unused_parameters=True)

    params_student = sum(p.numel() for p in distiller.module.student.parameters() if p.requires_grad)
    logger.info(f"Ndlinear Parameter Count: {params_student}")

    total_steps = args.num_epochs * len(train_loader)
    optimizer = optim.AdamW(distiller.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)
    scaler = GradScaler()

    best_accuracy = 0.0
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        avg_loss = train_epoch(train_loader, distiller, optimizer, scheduler, scaler, device)
        accuracy_1, accuracy_5 = validate(test_loader, distiller.module.student, device)

        epoch_result = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy@1": accuracy_1,
            "accuracy@5": accuracy_5,
        }
        results.append(epoch_result)
        best_accuracy = save_checkpoint_and_results(args.dataset, args.num_transformers, epoch_result, best_accuracy,
                                                    distiller, local_rank, logger, timestamp)

    if local_rank == 0:
        results_filename = f"torchvt_{args.dataset.lower()}_{timestamp}_{args.num_transformers}tf.json"
        with open(results_filename, "w") as fp:
            json.dump(results, fp, indent=4)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_transformers', type=int, default=6, help='Number of transformers to use in the model')
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100'],
                        help='Choose between CIFAR10 or CIFAR100')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=2.75e-4)
    parser.add_argument('--input_dim', type=int, default=128, help='Input dimensions.')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for the dataset')

    args = parser.parse_args()
    train(args)