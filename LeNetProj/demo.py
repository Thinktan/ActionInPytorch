import torch
import torch.nn as nn
from model import LeNet
from data import data_train, data_test
from torch.utils.data import DataLoader
from utils import test_one_case
import matplotlib.pyplot as plt

model_path = "./model.pth"  # 假设模型保存在model.pth文件中
save_info = torch.load(model_path)  # 载入模型
model = LeNet()
model.load_state_dict(save_info["model"])  # 载入模型参数
model.eval()  # 切换模型到测试状态

index = 1341

img, target = data_test[index]
img = img.unsqueeze(0)
# print('img.shape: ', img.shape)

outputs = model(img)
_, predicted = outputs.max(1)
print('output: ', outputs)
print('predicted: ', predicted)


plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(img.numpy().squeeze(), cmap='gray_r')
print('target: ', target)
plt.show()