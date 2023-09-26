import torch
import torch.nn as nn
from model import LeNet
from data import data_test
from torch.utils.data import DataLoader


# save_info = { # 保存的信息
#    "optimizer": optimizer.state_dict(), # 优化器的状态字典
#    "model": model.state_dict(), # 模型的状态字典
# }
if __name__ == '__main__':
    data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    model_path = "./model.pth"  # 假设模型保存在model.pth文件中
    save_info = torch.load(model_path)  # 载入模型
    model = LeNet()  # 定义LeNet模型
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    model.load_state_dict(save_info["model"])  # 载入模型参数
    model.eval()  # 切换模型到测试状态

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭计算图
        for batch_idx, (inputs, targets) in enumerate(data_test_loader):
            # print('inputs.shape: ', inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('predicted: %d, target: %d' % (predicted[0], targets[0]))

            print(batch_idx,  'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))