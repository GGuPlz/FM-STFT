import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from FMmodel import * 
from FMdataset import *
 
#定义超参数
num_epochs = 5
num_classes = 25
batch_size = 32
learning_rate = 0.01
# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
train_dataset = FMDataset(r'data\Indoor Foundation\indoor_day1_4MHz_4096_train.pt', r'data\Indoor Foundation\indoor_day1_4MHz_4096_train_label.pt', 128, 0.75)
test_dataset = FMDataset(r'data\Indoor Foundation\indoor_day1_4MHz_4096_test.pt', r'data\Indoor Foundation\indoor_day1_4MHz_4096_test_label.pt', 128, 0.75)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

model = ResNeXt50_32x4d(num_classes = num_classes).to(device)
model.load_state_dict(torch.load('model.ckpt'))

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 向后优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i+1 == 500:
            print(outputs.size())
            print(outputs[0])
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            print(labels)
            print(loss)
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item())) 
 
# 测试模型
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
 
# 保存模型
torch.save(model,'save.pt')

