import pu
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import multiprocessing
import time
import numpy as np

BATCH_SIZE = 512#1 # 512大概需要2G的显存
EPOCHS = 2 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
def test(model, device, test_loader):
    model.eval()
    test_loss =0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction = 'sum') # 将一批的损失相加
            pred = output.max(1, keepdim = True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100.* correct / len(test_loader.dataset)
            ))

def train(model, device, train_loader, optimizer,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        #if 100. * batch_idx / len(train_loader) >=2:
        #    break

def myGPU(pip,c2g,g2c,stop):
    device_gpu=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    device_cpu=torch.device("cpu")
    while True:
        if not c2g.empty():
            c2g.get()
            x=pip.recv()
            x=x.to(device_gpu)
            x=F.relu(x)
            x=x.to(device_cpu)
            pip.send(x)
            g2c.put('ready')

        if not stop.empty():
            stop.get()
            break
            

def myReLU(x):
    pipA.send(x.detach())
    cgQueue.put("ready")
    while True:
        if not gcQueue.empty():
            gcQueue.get()
            x=pipA.recv()
            return x

class myNN(nn.Module):

    def __init__(self):
        super().__init__()
        #1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5) 
        self.conv2 = nn.Conv2d(10, 20, 3) 
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        in_size = x.size(0)
        out= self.conv1(x) # 1* 10 * 24 *24
        out = myReLU(out)
        out = F.max_pool2d(out, 2, 2) # 1* 10 * 12 * 12
        out = self.conv2(out) # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(in_size, -1) # 1 * 2000
        out = self.fc1(out) # 1 * 500
        out = F.relu(out)
        out = self.fc2(out) # 1 * 10
        out = F.log_softmax(out, dim = 1)
        return out
    
#shared_data=multiprocessing.Array('d',np.array(torch.rand(10,24,24,32).numpy().tolist()))

pipA,pipB=multiprocessing.Pipe()
cgQueue=multiprocessing.Queue()
gcQueue=multiprocessing.Queue()
stop_sign=multiprocessing.Queue()

def main():
    print("[INFO]  Data-loading")
    ###########################################  loading  ################################
    # train
    pth='./data/'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(pth, train = True, download = True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1037,), (0.3081,))
                ])),
    batch_size = BATCH_SIZE, shuffle = True)

    # test
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(pth, train = False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])),
    batch_size = BATCH_SIZE, shuffle = True)
    ############################################  multiprocess  #########################################
    '''    pipA,pipB=multiprocessing.Pipe()
    cgQueue=multiprocessing.Queue()
    gcQueue=multiprocessing.Queue()
    stop_sign=multiprocessing.Queue()'''
    p = multiprocessing.Process(target = myGPU, args = (pipB,cgQueue,gcQueue,stop_sign))
    p.start()
    ##################################################         begin myNN  ########################################
    model = myNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    print('[INFO]  Data loaded, myNN beginning')
    for epoch in range(1, EPOCHS + 1):
        train(model,  DEVICE, train_loader, optimizer,epoch)
        test(model, DEVICE, test_loader)
    #####################################################    close subprocess   #####################################
    stop_sign.put("stop")

    p.join()
    cgQueue.close()
    gcQueue.close()
    stop_sign.close()





if __name__=='__main__':
    main()
    
