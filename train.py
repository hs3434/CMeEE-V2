import torch.optim as optim
import pandas as pd
import torch
import time
import os
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from tensorboardX import SummaryWriter
from transformer import len_mask2, subsequent_mask, Transformer
from log import My_logger
path = os.path.dirname(__file__)
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
    special_symbols += ["dis", "sym", "pro",
                        "equ", "dru", "ite", "bod", "dep", "mic"]
    with open(path, encoding="utf-8") as r:
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


def train(model, train_loader, criterion, optimizer, epochs=5,
          device=None, model_file="model.pt", logger=My_logger(),
          writer=SummaryWriter()):
    global_step = 0
    for epoch in range(epochs):
        n = 0
        start = time.time()
        total = 0
        for batch, batch_mask, label, label_mask, target in train_loader:
            if device is not None:
                torch.cuda.empty_cache()
                batch = batch.to(device)
                label = label.to(device)
                batch_mask = batch_mask.to(device)
                label_mask = label_mask.to(device)
                target = target.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(batch, label, batch_mask, label_mask)
            # 计算损失
            loss = criterion(outputs.transpose(1, -1), target)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            if n == 0:
                text = g_gpu_memory() + f", Epoch [{epoch+1}/{epochs}]"
                writer.add_text("test", text)
                logger.info(text)
            n += 1
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"batch {n}, Loss: {loss.item():.4f}, {localtime}")
            writer.add_scalar("loss", loss.item(), global_step+n)
            writer.add_scalar('Average batch spend', (time.time()-start)/n, global_step+n)
            total += loss.item()
        global_step += n
        end = time.time()
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {(total/n):.4f}, spend: {end-start}s')
        writer.add_scalar('Average Loss', total/n, epoch)
        torch.save(model.state_dict(), model_file)
    return model


def pre_data(train_file, save_path=None):
    df = pd.read_json(train_file)
    batch, batch_mask, label, label_mask, target = build_batch(df, vocab)
    batch_mask = batch_mask.unsqueeze(-2)
    label_mask = len_mask2(label_mask, label_mask.size(-1))
    sub_mask = subsequent_mask(label_mask.size(-1))
    label_mask = label_mask.masked_fill(sub_mask, True)
    dataset = TensorDataset(batch, batch_mask, label, label_mask, target)
    if save_path is not None:
        torch.save(dataset, save_path)
    return dataset


def gpu_memory(device):
    props = torch.cuda.get_device_properties(device)
    return props.total_memory


def list_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU{i}显存为: {gpu_memory(i) / (1024 ** 3):.2f} GB")
    else:
        print("没有找到GPU！")


def g_gpu_memory(device=0):
    gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    all = (gpu_memory(device) - gpu_memory_allocated) / (1024 ** 3)
    text = f"显存占用: {gpu_memory_allocated:.2f}/{all:.2f} GB"
    return text


def inference(text, model, vocab: Vocab, stop=1000):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = torch.tensor([vocab.stoi[v] for v in text]).unsqueeze(0).to(device)
    tgt = torch.tensor(vocab.stoi["<SOS>"]).unsqueeze(
        0).unsqueeze(0).to(device)
    model = model.to(device)
    while stop:
        stop -= 1
        prob = model(input, tgt, None, None)
        _, next_word = torch.max(prob[:, -1], dim=1)
        next_word = next_word.data[0]
        if vocab.stoi["<EOS>"] == next_word:
            break
        tgt = torch.cat(
            [tgt, torch.empty(1, 1).type_as(input.data).fill_(next_word)],
            dim=1
        )
    out = "".join([vocab.itos[v] for v in tgt[0, :]])
    return out


if __name__ == "__main__":
    # 参数设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_file = os.path.join(path, "CMeEE-V2_train.json")
    vocab_path = os.path.join(path, "vocab.pt")
    dataset_file = os.path.join(path, "train_loader.pt")
    model_file = os.path.join(path, "model_params.pt")
    batch_size = 5  # batch_size越大，平均单个样本所需的训练时间越大。
    load_vocab = True
    load_data = True
    load_model = True
    logger = My_logger(level=20)
    logger.add_handler("file", level=20, file="log.txt")
    writer = SummaryWriter('./runs/train')
    # 预处理
    mark = time.time()
    if load_vocab:
        vocab = torch.load(vocab_path, weights_only=False)
        text = f"载入词表花费 {time.time()-mark:.3f} s"
        writer.add_text('初始化', text)
        logger.info(text)
    else:
        vocab = build_vocab(train_file, vocab_path)
        text = f"创建词表花费 {time.time()-mark:.3f} s"
        writer.add_text('初始化', text)
        logger.info(text)
    vocab_dim = len(vocab)
    mark = time.time()
    if load_data:
        dataset = torch.load(dataset_file, weights_only=False)
        text = f"载入数据集花费 {time.time()-mark:.3f} s"
        writer.add_text('初始化', text)
        logger.info(text)
    else:
        dataset = pre_data(train_file, save_path=dataset_file)
        text = f"创建数据集花费 {time.time()-mark:.3f} s"
        writer.add_text('初始化', text)
        logger.info(text)
    mark = time.time()
    model = Transformer(src_vocab=vocab_dim, tgt_vocab=vocab_dim, N=2,
                        d_model=128, d_ff=512, h=4, dropout=0.1, device=device)
    if load_model:
        model.load_state_dict(torch.load(model_file))
        text = f"载入模型花费 {time.time()-mark:.3f} s"
        writer.add_text('初始化', text)
        logger.info(text)
    else:
        text = f"创建模型花费 {time.time()-mark:.3f} s"
        writer.add_text('初始化', text)
        logger.info(text)
    logger.info(g_gpu_memory())
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 训练
    model = train(model, train_loader, criterion, optimizer, epochs=5,
                  device=device, model_file=model_file, logger=logger, writer=writer)
