from sympy import Sum
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FMdataset import *

from FMmodel import *

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)




train_data = FMDataset(r'data\Indoor Foundation\indoor_day1_4MHz_4096_train.pt', r'data\Indoor Foundation\indoor_day1_4MHz_4096_train_label.pt', 64, 0.75)
test_data = FMDataset(r'data\Indoor Foundation\indoor_day1_4MHz_4096_test.pt', r'data\Indoor Foundation\indoor_day1_4MHz_4096_test_label.pt', 64, 0.75)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FM_Pnet()
model = model.to(device)
model.apply(init_weights)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 10

work_dir = './result'
writer = SummaryWriter("{}/logs".format(work_dir))


for i in range(epoch):
    print("-------epoch  {} -------".format(i+1))
    # 训练步骤
    model.train()
    for step, [imgs, targets] in enumerate(train_dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        
        #for name, parms in model.named_parameters():	
         #   print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		  #  ' -->grad_value:',parms.grad)
        loss = loss_fn(outputs, targets)
 
        # 优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        train_step = len(train_dataloader)*i+step+1
        if train_step % 100 == 0:
            print("train time：{}, Loss: {}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)
           # print(outputs)
            #print(outputs.size())
            #print(targets)
            #print(targets.size())
 
    # 测试步骤
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() #argmax(1)表示把outputs矩阵中的最大值输出
            total_accuracy = total_accuracy + accuracy
 
    print("test set Loss: {}".format(total_test_loss))
    print("test set accuracy: {}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss", total_test_loss, i)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_data), i)
 
    torch.save(model, "{}/module_{}.pth".format(work_dir,i+1))
    print("saved epoch {}".format(i+1))
 
writer.close()
