import torch
from torch import nn
from model import GRU
import torch.utils.data as Data
from data import TextDataset
import jieba
import logging
import time
import os
import numpy as np

jieba.setLogLevel(logging.ERROR)


def test_data_process():
    ROOT_TRAIN = r'.\data\神雕侠侣.txt'

    is_chinese = lambda c: u'\u4e00' <= c <= u'\u9fa5'
    is_digit = lambda c: u'\u0030' <= c <= u'\u0039'
    is_alpha = lambda c: (u'\u0041' <= c <= u'\u005a') or (u'\u0061' <= c <= u'\u007a')
    is_punct = lambda c: c in ('，', '。', '：', '？', '"', '"', '！', '；', '、', '《', '》', '——')
    is_valid = lambda c: is_chinese(c) or is_digit(c) or is_alpha(c) or is_punct(c)

    with open(ROOT_TRAIN, 'r', encoding='gbk') as f:
        text = f.read()
        text = jieba.lcut(text)
        text = list(filter(is_valid, text))

    vocab = np.array(sorted(set(text)))

    test_data = TextDataset(text, vocab, time_step=50)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=0)

    return len(vocab), test_dataloader


if __name__ == '__main__':
    vocab_size, test_dataloader = test_data_process()
    index = 4
    t_x, t_y = test_dataloader.dataset.__getitem__(index)

    class_names = test_dataloader.dataset.int2word

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GRU(vocab_size)
    model_path = './model/GRU/model/best.pth'
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 '{model_path}'")
        print("请先运行训练脚本生成模型文件。")
        exit(1)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    with torch.no_grad():
        test_data_x = t_x.unsqueeze(0).to(device)

        model.eval()
        output, _ = model(test_data_x, None)

        probabilities = torch.softmax(output, dim=1)
        pre_label = torch.argmax(probabilities, dim=1) 
        pred_class = class_names[pre_label.item()]
        confidence_score = torch.max(probabilities, dim=1)[0].item()

        print(f"预测结果: {pred_class}")
        print(f"置信度: {confidence_score:.4f}")
        print(f"所有类别概率: {probabilities.cpu().numpy()}")

        true_class = class_names[t_y.item()]
        if pre_label.item() == t_y:
            print(f"✅  {true_class} - CORRECT")
        else:
            print(f"❌  {true_class} → {pred_class} - ERROR")
