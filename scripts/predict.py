import torch
from model import MODEL   # mark
from torchvision import transforms
from PIL import Image
import numpy as np




if __name__ == '__main__':
    image = Image.open('demo.jpg')
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
    transform_demo = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
    image_transform = transform_demo(image)

    class_names = []    # mark

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MODEL()   # mark
    model_path = './model/MODEL/model/best.pth'     # mark
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 '{model_path}'")
        print("请先运行训练脚本生成模型文件。")
        exit(1)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    with torch.no_grad():
        image_transform = image_transform.unsqueeze(0).to(device)
        
        model.eval()
        output = model(image_transform)

        probabilities = torch.softmax(output, dim=1)
        pre_label = torch.argmax(probabilities, dim=1)
        pred_class = class_names[pre_label.item()]
        confidence_score = torch.max(probabilities, dim=1)[0].item()
    
        print(f"预测结果: {pred_class}")
        print(f"置信度: {confidence_score:.4f}")
        print(f"所有类别概率: {probabilities.cpu().numpy()}")
    





