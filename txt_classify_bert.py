import torch
import torch.nn as nn
from ndlinear import NdLinear
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import argparse


class NdClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NdClassificationHead, self).__init__()
        self.ndlinear = NdLinear(input_dims=(input_dim // 8, 8), hidden_size=(2, 2))
        self.linear = nn.Linear(4, output_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] // 8, 8)
        x = self.ndlinear(x)
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)


class CoLADataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_cola_data(file_path, max_size=3000):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['source', 'label', 'nothing', 'sentence'])
    train_val_sentences, test_sentences, train_val_labels, test_labels = train_test_split(
        data['sentence'][:max_size], data['label'][:max_size], test_size=0.2, random_state=42
    )
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_val_sentences, train_val_labels, test_size=0.25, random_state=42
    )
    return train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels


def prepare_cola_data_loaders(tokenizer, train_sentences, val_sentences, test_sentences, train_labels, val_labels,
                              test_labels, batch_size):
    MAX_LEN = 128
    train_dataset = CoLADataset(train_sentences.to_numpy(), train_labels.to_numpy(), tokenizer, MAX_LEN)
    val_dataset = CoLADataset(val_sentences.to_numpy(), val_labels.to_numpy(), tokenizer, MAX_LEN)
    test_dataset = CoLADataset(test_sentences.to_numpy(), test_labels.to_numpy(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_epoch(model, data_loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    auc_roc = roc_auc_score(true_labels, predictions)
    return accuracy, auc_roc


def test_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    auc_roc = roc_auc_score(true_labels, predictions)
    return accuracy, auc_roc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels = load_cola_data(
        args.file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_ndlinear = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    hidden_size = model_ndlinear.config.hidden_size
    num_labels = model_ndlinear.config.num_labels
    model_ndlinear.classifier = NdClassificationHead(hidden_size, num_labels)
    ndlinear_classifier_params = count_parameters(model_ndlinear.classifier)
    print(f"Number of parameters in model_ndlinear.classifier: {ndlinear_classifier_params}")
    device = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    optimizer_ndlinear = AdamW(model_ndlinear.parameters(), lr=args.learning_rate)
    model_ndlinear.to(device)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    train_loader_ndlinear, val_loader_ndlinear, test_loader_ndlinear = prepare_cola_data_loaders(
        tokenizer, train_sentences, val_sentences, test_sentences, train_labels, val_labels, test_labels,
        args.batch_size
    )

    best_ndlinear_accuracy = 0.0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model_ndlinear, train_loader_ndlinear, optimizer_ndlinear, device, scaler)
        val_accuracy, val_auc_roc = eval_model(model_ndlinear, val_loader_ndlinear, device)
        print(
            f"Model NdLinear - Train loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, AUC ROC: {val_auc_roc:.4f}")

        if val_accuracy > best_ndlinear_accuracy:
            best_ndlinear_accuracy = val_accuracy
            torch.save(model_ndlinear.state_dict(), 'best_txt_bert_ndlinear_model_cola.pth')

    model_ndlinear.load_state_dict(torch.load('best_txt_bert_ndlinear_model_cola.pth'))
    test_accuracy_ndlinear, test_auc_roc_ndlinear = test_model(model_ndlinear, test_loader_ndlinear, device)
    print(f"Model NdLinear - Test Accuracy: {test_accuracy_ndlinear:.4f}, AUC ROC: {test_auc_roc_ndlinear:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--file_path', type=str, default='data/CoLA/train.tsv')
    args = parser.parse_args()
    main(args)