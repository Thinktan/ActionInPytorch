

import torch
import torch.nn as nn
from model import LeNet
from data import data_train
from torch.utils.data import DataLoader

if __name__ == '__main__':
    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)

    model = LeNet()
    model.train() # switch to train state

    print(model)

    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print(optimizer)
    # exit(0)

    for num_epoch in range(20):
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(data_train_loader):
            optimizer.zero_grad()
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, targets) # 计算损失
            loss.backward() # 反向传播计算梯度
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1) # 预测的tem
            total += targets.size(0) # 总预测次数
            correct += predicted.eq(targets).sum().item() # 预测正确次数

            print(num_epoch, batch_idx, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # 保存模型
    save_info = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }

    save_path = './model.pth'
    torch.save(save_info, save_path)
    print('save succ!!!')



