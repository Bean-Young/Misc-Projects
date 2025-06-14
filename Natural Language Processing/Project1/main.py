import yaml
import torch
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
from data.dataloader import LTCRDataset
from data.preprocess import load_data

#  导入所有模型
from models.textcnn import TextCNN
from models.textrnn import TextRNN
from models.fasttext import FastText
from models.textrcnn import TextRCNN
from models.textrnn_att import TextRNNAtt
from models.dpcnn import DPCNN
from models.transformer import TransformerModel
from models.bert_finetune import BERTFinetune
from models.han import HAN
from models.cnn_trans_rnn import CNNTransRNN

def get_model(name, config, vocab_size):
    if name == "TextCNN":
        return TextCNN(config, vocab_size)
    elif name == "TextRNN":
        return TextRNN(config, vocab_size)
    elif name == "FastText":
        return FastText(config, vocab_size)
    elif name == "TextRCNN":
        return TextRCNN(config, vocab_size)
    elif name == "TextRNNAtt":
        return TextRNNAtt(config, vocab_size)
    elif name == "DPCNN":
        return DPCNN(config, vocab_size)
    elif name == "Transformer":
        return TransformerModel(config, vocab_size)
    elif name == "BERTFinetune":
        return BERTFinetune(config, vocab_size)
    elif name == "HAN":
        return HAN(config, vocab_size)
    elif name == "CNNTransRNN":
        return CNNTransRNN(config, vocab_size)  
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    with open("./NLP-Class/Project1/configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    tokenizer = BertTokenizer.from_pretrained(
        "/home/yyz/NLP-Class/Project1/bert_local/",
        local_files_only=True
    )

    # 加载完整训练集（包含 label 0/1/2）与验证集
    train_texts, val_texts, train_labels, val_labels = load_data(config)

    print(f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples.")
    #  去除验证集中的 uncertain（label=2）
    filtered_val = [(t, l) for t, l in zip(val_texts, val_labels) if l in [0, 1]]
    val_texts, val_labels = zip(*filtered_val)

    train_dataset = LTCRDataset(train_texts, train_labels, tokenizer, config['max_seq_len'])
    val_dataset = LTCRDataset(val_texts, val_labels, tokenizer, config['max_seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    vocab_size = tokenizer.vocab_size
    model = get_model(config['model_name'], config, vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))

    from trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, config, optimizer)
    trainer.train()

    #  保存最后一轮训练后的模型权重
    final_model_path = os.path.join(config['save_path'], config['model_name']+'_'+config['dataset_name']+'_final_model.pt')
    torch.save(model.state_dict(), final_model_path)

    #  加载模型进行测试（使用清洗后的验证集）
    print("\nEvaluating final model on validation set...")
    model.load_state_dict(torch.load(final_model_path))
    model.eval()
    device = torch.device(config['device'])
    model.to(device)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(all_labels, all_preds, digits=4)
    print(report)

    #  保存报告
    with open(os.path.join(config['log_path'], config['model_name']+'_'+config['dataset_name']+"_test_report.txt"), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
