import torch
from torch import nn
from model import ResNet_18
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import time
import os

def test_data_process():
    test_data = FashionMNIST(root='./data', 
                          train = False,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_num = 0
    test_corrects = 0
    test_acc =0.0

    since = time.time()
    with torch.no_grad():
        for t_x, t_y in test_dataloader: 
            test_data_x = t_x.to(device)
            test_data_y = t_y.to(device)

            model.eval()
            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1) 

            test_corrects += torch.sum(pre_label == test_data_y.data)
            test_num += test_data_x.size(0)
            

    test_acc = test_corrects.double().item() / test_num
    time_use = time.time() - since
    
    print('='*60)
    print(f'📊 TEST RESULTS')
    print('='*60)
    print(f'✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
    print(f'🔢 Total Samples: {test_num}')
    print(f'✔️ Correct Predictions: {test_corrects}')
    print(f'⏱️ Test Time: {(time_use//60):02.0f}m{(time_use%60):02.0f}s')
    print('='*60)


def test_model_detail_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    class_label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("="*80)
    print("🔍 DETAILED TEST RESULTS")
    print("="*80)
    
    with torch.no_grad():
        for index, (t_x, t_y) in enumerate(test_dataloader): 
            test_data_x = t_x.to(device)
            test_data_y = t_y.to(device)

            model.eval()
            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1) 

            pred_class = class_label[pre_label.item()]
            true_class = class_label[test_data_y.item()]

            if pre_label == test_data_y.item():
                print(f"✅ Sample {index+1:4d}: {true_class} - CORRECT")
            else:
                print(f"❌ Sample {index+1:4d}: {true_class} → {pred_class} - ERROR")

if __name__ == '__main__':
    test_dataloader = test_data_process()

    model = ResNet_18(1, 10)

    model_path = './model/ResNet-18/model/best.pth'
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 '{model_path}'")
        print("请先运行训练脚本生成模型文件。")
        exit(1)
    model.load_state_dict(torch.load(model_path))

    # test_model_process(model, test_dataloader)
    test_model_detail_process(model, test_dataloader)
