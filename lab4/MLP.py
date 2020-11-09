import pickle
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.datasets as datasets
import torch.utils.data as data 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 将train loss，accuracy记录到tensorboard中
writer = SummaryWriter("./lab4")

class ReShape():
    """
    用于将高维数据转化为一维数据
    """
    def __call__(self, tensor):
        # print(type(sample))
        return tensor.reshape(-1)

class MLP(nn.Module):
    """
    多层感知机模型结构
    
    Arguments:
        hidden_num: 隐含层单元个数
    """
    def __init__(self, hidden_num):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(85, hidden_num)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_num, 10)
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return self.softmax(x)

def validation(model, test_loader):
    """
    用于测试准确率，便于选取最优模型
    
    Arguments:
        model: 用于测试的模型
        test_loader: 验证集dataloader
    
    Return:
        准确率
    """
    right_num = 0
    batch_size=test_loader.batch_size
    for i, (images, labels) in enumerate(test_loader):
        output = model(images)
        right_num +=(torch.argmax(output, dim=1)==labels).float().sum()
    return right_num / (len(test_loader)*batch_size)

def train(model, optimizer, loss_func, train_loader, test_loader, train_name, save_model=False, epoch=3, num_iter=3000):
    """
    用于单个模型的训练
    
    Arguments:
        model: 用于训练的模型
        optimizer: 优化器
        loss_func: loss函数
        train_loader: 训练集dataloder
        test_loader: 测试机dataloader
        train_name: 模型标记，用于tensroboard记录
        epoch: 循环次数
        num_iter: 测试和记录的频率
    
    Return:
        训练过程结果
    """
    iter = 0
    accuracy = []
    loss = 0
    max_acc = 0 
    # 用于learning rate的更新，在迭代次数较大时选用较小的lr
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70000], gamma=0.1)
    for e in range(epoch):
        for i, (images, labels) in enumerate(train_loader):
            model.train()  # 切换到train模式，由于MLP未选用特定层，故此处无实际作用
            iter += 1
            output = model(images)
            iter_loss = loss_func(output, labels)  # 计算loss
            iter_loss.backward()  
            loss += iter_loss
            
            optimizer.step()  # 参数更新
            optimizer.zero_grad()  # 梯度归零
            scheduler.step()  # 学习率更新
            
            if iter % 3000 == 0: 
                model.eval()  # 切换到test模式
                with torch.no_grad():
                    iter_accuracy = validation(model, test_loader)
                    loss = loss/3000
                    accuracy.append({'iter_number': iter, 'loss': loss, 'accuracy':iter_accuracy})
                    print({'iter_number': iter, 'loss': loss, 'accuracy':iter_accuracy})
                    writer.add_scalar(train_name+"_loss", loss, iter)  # tensroboard记录loss
                    writer.add_scalar(train_name+"_accuracy", iter_accuracy, iter)  # tensorboard记录accuracy
                    if iter_accuracy > max_acc:
                        max_acc = iter_accuracy
                        if save_model:
                            torch.save(model.state_dict(), "best_model.pth")
                    loss = 0
    return accuracy

class MNIST(data.dataset):
	def __init__(self, path, train=True, transform=None):
		super(MNIST).__init__()

		# get filename and data
		self.train = "Train" if train else "Test"
		data_file = osp.join(path, self.train+"Samples.csv")
		label_file = osp.join(path, self.train+"Labels.csv")
		self.data = pd.read_csv(data_file, header=None)
		self.labels = pd.read_csv(label_file, header=None)
		assert len(data) == len(labels)

		# get transform
		self.transform = transform

	def __len__(self):
		return len(data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx].to_numpy()
		label = torch.self.labels.iloc[idx].to_numpy()
		label = torch.from_numpy(label)[0]
		if self.transform is not None:
			sample = self.transform(sample)
		return sample, label 


if __name__ == '__main__':
    # Step 1: 只保留inference(), 运行inference() 
    # Step 2: 根据tensorboard曲线寻找最优配置, 并注释inference()
    # Step 3: 根据最优配置训练模型, 并将模型保存到.pth文件中
    # inference()

    transform = transforms.Compose([
        transforms.ToTensor(),
            ReShape()
        ])
    train_set = MNIST("./data/", train=True, transform=transform)
    test_set = MNIST("./data/", train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)
    
    net = MLP(30)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    train(net, optimizer, loss_func, train_loader=train_loader, 
                                 test_loader=test_loader, train_name='best', epoch=8, save_model=True)

