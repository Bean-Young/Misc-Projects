import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import torch.nn as nn

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config['device'])
        self.optimizer = optimizer
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=2)

    def train(self):
        os.makedirs(self.config['log_path'], exist_ok=True)
        result_log = os.path.join(self.config['log_path'], self.config['model_name']+'_'+self.config['dataset_name']+"_metrics.txt")

        with open(result_log, 'w') as f:
            for epoch in range(self.config['num_epochs']):
                self.model.train()
                for batch in self.train_loader:
                    inputs = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                acc, f1, prec, recall = self.evaluate()
                line = f"Epoch {epoch+1}, Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}"
                print(line)
                f.write(line + '\n')

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        return (
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro'),
            precision_score(all_labels, all_preds, average='macro'),
            recall_score(all_labels, all_preds, average='macro')
        )

   