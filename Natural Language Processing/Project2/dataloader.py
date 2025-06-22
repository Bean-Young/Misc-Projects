import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, test_size=0.2, split='train', random_state=42):
        self.data = pd.read_excel(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 划分训练集和验证集
        self.train_data, self.val_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        
        # 根据 split 加载训练集或验证集
        if split == 'train':
            self.data = self.train_data
        elif split == 'val':
            self.data = self.val_data
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.iloc[idx]['中文原句']
        target_text = self.data.iloc[idx]['日文翻译']
        
        source_encoding = self.tokenizer.encode_plus(
            source_text,
            max_length=self.max_length,
            padding='max_length',  # 确保启用padding
            truncation=True,
            return_tensors='pt',
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False  # GPT-2通常不需要token type ids
        )
                
        target_encoding = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def create_dataloader(file_path, tokenizer, batch_size=64, max_length=128, split='train', test_size=0.2):
    dataset = TranslationDataset(file_path, tokenizer, max_length, test_size, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
