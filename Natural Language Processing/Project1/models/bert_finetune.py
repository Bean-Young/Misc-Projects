import torch
import torch.nn as nn
from transformers import AutoModel

class BERTFinetune(nn.Module):
    def __init__(self, config, vocab_size=None):
        super(BERTFinetune, self).__init__()
        # 本地加载预训练模型
        self.bert = AutoModel.from_pretrained(
            "./NLP-Class/Project1/bert_local/",
            local_files_only=True
        )
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(hidden, config['num_classes'])

    def forward(self, x):
        outputs = self.bert(input_ids=x)
        pooled = outputs.last_hidden_state[:, 0, :]
        out = self.dropout(pooled)
        return self.fc(out)
