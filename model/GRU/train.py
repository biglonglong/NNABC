import torch
from torch import nn
from model import GRU
import torch.utils.data as Data
from data import TextDataset
import jieba
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import os

model_save_dir = './model/GRU/model'
time_step = 50
jieba.setLogLevel(logging.ERROR)


def train_val_data_process():
    ROOT_TRAIN = r'.\data\train.txt'

    is_chinese = lambda c: u'\u4e00' <= c <= u'\u9fa5'
    is_digit = lambda c: u'\u0030' <= c <= u'\u0039'
    is_alpha = lambda c: (u'\u0041' <= c <= u'\u005a') or (u'\u0061' <= c <= u'\u007a')
    is_punct = lambda c: c in ('，', '。', '：', '？', '"', '"', '！', '；', '、', '《', '》', '——')
    is_valid = lambda c: is_chinese(c) or is_digit(c) or is_alpha(c) or is_punct(c)

    with open(ROOT_TRAIN, 'r', encoding='utf-8') as f:
        text = f.read()
        text = jieba.lcut(text)
        text = list(filter(is_valid, text))

    vocab = np.array(sorted(set(text)))

    train_data = TextDataset(text, vocab, time_step=time_step)

    train_data, val_data = Data.random_split(train_data, lengths=[round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=8,
                                       shuffle=True,
                                       num_workers=2,
                                       drop_last=True)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=8,
                                       shuffle=True,
                                       num_workers=2,
                                       drop_last=True)

    return len(vocab), train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_model_wts = copy.deepcopy(model.state_dict())

    train_avg_loss_all = []
    train_acc_all = []
    val_avg_loss_all = []
    val_acc_all = []
    best_acc = 0.0

    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1:2d}/{num_epochs} -> ')
        train_num = 0
        train_loss = 0.0
        train_corrects = 0
        val_num = 0
        val_loss = 0.0
        val_corrects = 0
        total_train_batches = len(train_dataloader)

        hs = None
        for step, (b_x, b_y) in enumerate(train_dataloader):
            batch_x = b_x.to(device)
            batch_y = b_y.to(device)

            model.train()
            output, hs = model(batch_x, hs)

            if hs is not None:
                hs = hs.detach()

            pre_label = torch.argmax(output, dim=1)

            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_corrects += torch.sum(pre_label == batch_y.data)
            train_num += batch_x.size(0)

            if (step + 1) % 10 == 0 or step == total_train_batches - 1:
                progress = (step + 1) / total_train_batches * 100
                print(f'\rTraining {progress:5.1f}%', end='', flush=True)

        print(' ' * 20 + '\r', end='')
        for step, (b_x, b_y) in enumerate(val_dataloader):
            batch_x = b_x.to(device)
            batch_y = b_y.to(device)

            model.eval()
            output, _ = model(batch_x, None)
            pre_label = torch.argmax(output, dim=1) 

            loss = criterion(output, batch_y)

            val_loss += loss.item() * batch_x.size(0)
            val_corrects += torch.sum(pre_label == batch_y.data)
            val_num += batch_x.size(0)

        train_avg_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_avg_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f'Train Loss: {train_avg_loss_all[-1]:.4f} | '
              f'Train Acc: {train_acc_all[-1]:.4f} | '
              f'Val Loss: {val_avg_loss_all[-1]:.4f} | '
              f'Val Acc: {val_acc_all[-1]:.4f} | '
              f'Time: {(time_use//60):02.0f}m{(time_use%60):02.0f}s')
        print('-' * 90)

    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
                "train_avg_loss_all": train_avg_loss_all,
                "val_avg_loss_all": val_avg_loss_all,
                "train_acc_all": train_acc_all,
                "val_acc_all": val_acc_all}
    )

    os.makedirs(model_save_dir, exist_ok=True)

    torch.save(best_model_wts, model_save_dir + '/best.pth')

    return train_process


def loss_acc_matplot(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'], train_process['train_avg_loss_all'], 'ro-', label='train_loss')
    plt.plot(train_process['epoch'], train_process['val_avg_loss_all'], 'bs-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation Loss')

    plt.subplot(1,2,2)
    plt.plot(train_process['epoch'], train_process['train_acc_all'], 'ro-', label='train_acc')
    plt.plot(train_process['epoch'], train_process['val_acc_all'], 'bs-', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(model_save_dir + '/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    num_epochs = 1
    vocab_size, train_dataloader, val_dataloader = train_val_data_process()
    model = GRU(vocab_size, 512, 2, 0.5)

    if os.path.exists(model_save_dir + '/best.pth'):
        model.load_state_dict(torch.load(model_save_dir + '/best.pth'))
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs)
    loss_acc_matplot(train_process)