import torch.optim as optim
import pandas as pd
import torch
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from transformer import *

# Vocab 用法见 http://man.hubwiz.com/docset/torchtext.docset/Contents/Resources/Documents/vocab.html#build-vocab-from-iterator
def build_vocab(path, vocab_path):
    special_symbols = [
        "<unk",
        "<pad>",
        "<SOS>",
        "\n",
        "\t",
        " ",
        "<EOS>",
        "<S1>",
        "<S2>",
    ]
    special_symbols += ["dis", "sym", "pro", "equ", "dru", "ite", "bod", "dep", "mic"]
    with open(path) as r:
        text = r.read()
    text = list(text)
    vocab = Vocab(Counter(text), specials=special_symbols)
    torch.save(vocab, vocab_path)
    return vocab


def build_batch(df: pd.DataFrame, vocab: Vocab):
    batch = pd.Series(df["text"]).apply(
        lambda x: torch.tensor([vocab.stoi[chr] for chr in list(x)])
    )
    pad = vocab.stoi["<pad>"]
    batch = pad_sequence(batch, batch_first=True, padding_value=pad)
    batch_mask = batch == pad
    label = pd.Series(df["entities"]).apply(build_label, vocab=vocab)
    target = label.apply(lambda x: x[1:])
    label = label.apply(lambda x: x[0:-1])
    label = pad_sequence(label, batch_first=True, padding_value=pad)
    target = pad_sequence(target, batch_first=True, padding_value=pad)
    label_mask = label == pad
    return batch, batch_mask, label, label_mask, target


def build_label(lis: list, vocab: Vocab):
    res = [vocab.stoi["<SOS>"]]
    for item in lis:
        res += [vocab.stoi[chr] for chr in list(item["entity"])] + [
            vocab.stoi["<S1>"],
            vocab.stoi[item["type"]],
            vocab.stoi["<S2>"],
        ]
    res.pop()
    return torch.tensor(res + [vocab.stoi["<EOS>"]])


def train(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for batch, batch_mask, label, label_mask, target in train_loader:
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(batch, label, batch_mask, label_mask)
            # 计算损失
            loss = criterion(outputs, target)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def pre_data(train_file, batch_size=1, save_path=None):
    df = pd.read_json(train_file)
    batch, batch_mask, label, label_mask, target = build_batch(df, vocab)
    batch_mask = batch_mask.unsqueeze(-2)
    label_mask = len_mask2(label_mask, label_mask.size(-1))
    sub_mask = subsequent_mask(label_mask.size(-1))
    label_mask = label_mask.masked_fill(sub_mask, True)
    dataset = TensorDataset(batch, batch_mask, label, label_mask, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if save_path is not None:
        torch.save(dataloader, save_path)
    return dataloader
    
    

train_file = "/work/CMeEE-V2/CMeEE-V2_train.json"
vocab_path = "/work/CMeEE-V2/vocab.pt"
train_loader_file = "/work/mnt/train_loader.pt"
# vocab = build_vocab(train_file, vocab_path)
vocab=torch.load(vocab_path, weights_only=False)
vocab_dim = len(vocab)
# train_loader = pre_data(train_file, batch_size=1, save_path=train_loader_file)
train_loader=torch.load(train_loader_file, weights_only=False)
model = Transformer(src_vocab=vocab_dim, tgt_vocab=vocab_dim, N=2, 
                        d_model=512, d_ff=2048, h=8, dropout=0.1)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
optimizer = optim.SGD(model.parameters(), lr=0.01)
train(model, train_loader, criterion, optimizer, epochs=5)


