import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, GPT2Tokenizer
from dataloader import create_dataloader
from tqdm import tqdm
from model.bert import BertTranslationModel
from model.gpt import GPT2TranslationModel
from model.transformer import TransformerTranslationModel
from model.new import HybridTranslationModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", unit="batch", ncols=100):
        src = batch['input_ids'].to(device)
        tgt = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        
        # 调用模型的前向传播
        output = model(src, tgt[:, :-1])  # Remove last token for input to decoder
        tgt = tgt[:, 1:]  # Shift the target to align with the decoder output

        # 计算损失
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch", ncols=100):
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = model(src, tgt[:, :-1])  # Remove last token for input to decoder
            tgt = tgt[:, 1:]  # Shift the target to align with the decoder output

            loss = criterion(output.reshape(-1, output.shape[-1]), tgt.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 选择BERT、GPT 或 Transformer模型
    model_choice = "new"  # 可以选择 "bert", "gpt", "transformer"
    
    if model_choice == "bert":
        model = BertTranslationModel().to(device)
        tokenizer = BertTokenizer.from_pretrained('/home/yyz/NLP-Class/Project2/bert')
    elif model_choice == "transformer":
        model = TransformerTranslationModel().to(device)
        tokenizer = BertTokenizer.from_pretrained('/home/yyz/NLP-Class/Project2/bert')  # 可以根据需要调整
    elif model_choice == "new":
        model = HybridTranslationModel().to(device)
        tokenizer = BertTokenizer.from_pretrained('/home/yyz/NLP-Class/Project2/bert')

    # 创建数据加载器
    file_path = '/home/yyz/NLP-Class/Project2/data/cff79186692165b02d3b94d650c11d55.xlsx'
    train_dataloader = create_dataloader(file_path, tokenizer, split='train')
    val_dataloader = create_dataloader(file_path, tokenizer, split='val')
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.src_pad_idx)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        
        # 评估模型
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
    
    # 保存模型
    model_save_path = '/home/yyz/NLP-Class/Project2/result/translation_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
