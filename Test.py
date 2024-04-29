import os.path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision.models
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import random


def read_images(image_path):
    image_names = []
    for name in os.listdir(image_path):
        image_file = os.path.join(image_path, name)
        image = Image.open(image_file).convert('RGB')
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ]
        )
        image = transform(image)
        image_names.append([image, name])
    return image_names


# params
img_height, img_weight = 512, 512
mix_up_num = 400
num_classes = 3
batch_size = 32
weight_decay = 1e-3
image_path = './Testing Set'
model_path = './ResMix.pth'

image_names = read_images(image_path)
print("Data OK---------------------------------------------------------------")
model = torchvision.models.resnet50(pretrained=True)
#print(model)
model.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

device = torch.device('cuda:0')
model = model.to(device)
model.cuda()
CELoss = torch.nn.CrossEntropyLoss()
pred_prob, pred_label, names = [], [], []


def softmax(x):
    """
    计算 softmax 函数

    参数：
    x -- 一个一维或多维数组

    返回值：
    softmax(x) -- 一个与 x 维度相同的数组，其中每个元素都经过 softmax 转换后的值
    """
    # 计算指数值
    exp_x = np.exp(x)
    # 对每个维度进行求和
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    # 计算 softmax
    softmax_x = exp_x / sum_exp_x
    return softmax_x


model.eval()  # 设置模型为评估模式
with torch.no_grad():
    for input, name in image_names:
        names.append(name)
        input = input.cuda()
        input = torch.unsqueeze(input, dim=0)
        output = model(input)
        output = torch.squeeze(output, dim=0)
        output = output.detach().cpu().numpy()
        output = softmax(output)
        pred_prob.append(output.tolist())
        pred_label.append(output.argmax())

pred0, pred1, pred2 = list(zip(*pred_prob))
# 创建最终结果DataFrame
final_results = pd.DataFrame({
    'case': names,
    'class': pred_label,
    'P0': pred0,
    'P1': pred1,
    'P2': pred2
})

# 将最终结果保存到CSV文件
final_results.to_csv('final_test_quality_predictions.csv', index=False)
