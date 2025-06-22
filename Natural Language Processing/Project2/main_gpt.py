import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, GPT2Tokenizer
from dataloader import create_dataloader
from tqdm import tqdm
from model.gpt import GPT2TranslationModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def train(model, dataloader, optimizer, device):
    """训练函数 - 针对GPT模型优化"""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", unit="batch", ncols=100):
        src = batch['input_ids'].to(device)
        tgt = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # 直接获取损失值（GPT模型内部处理损失计算）
        loss = model(src, tgt)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """评估函数 - 针对GPT模型优化"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch", ncols=100):
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            
            # 直接获取损失值
            loss = model(src, tgt)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 使用GPT模型
    model_choice = "gpt"
    print(f"Initializing {model_choice.upper()} model...")
    
    # 初始化GPT模型
    model = GPT2TranslationModel().to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('/home/yyz/NLP-Class/Project2/gpt2')
    
    # 确保设置了pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    
    # 创建数据加载器
    file_path = '/home/yyz/NLP-Class/Project2/data/cff79186692165b02d3b94d650c11d55.xlsx'
    print("Creating dataloaders...")
    train_dataloader = create_dataloader(file_path, tokenizer, split='train')
    val_dataloader = create_dataloader(file_path, tokenizer, split='val')
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 10
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # 训练
        train_loss = train(model, train_dataloader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = '/home/yyz/NLP-Class/Project2/result/gpt_translation_model.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path} with val loss: {val_loss:.4f}")
    
    print("\nTraining completed!")

if __name__ == '__main__':
    main()