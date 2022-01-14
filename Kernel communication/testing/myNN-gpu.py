import pu
import pickle
import torch
from torch import nn,optim
import torch.nn.functional as F

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden1 = nn.Linear(784, 256) # 隐藏层
        self.hidden2 = nn.Linear(256,64)
        self.act = nn.ReLU()
        self.output = nn.Linear(64, 10)  # 输出层
 
 
    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        x=self.hidden1(x)
        x=self.hidden2(x)
        x = self.act(x)
        return self.output(x)

def main():
    while(True):
        thread_recv,thread_send,read_queue,write_queue,stop_queue,udp_socket,my_socket=pu.socket_ini(0)
        
        while True:
            if not read_queue.empty():
                x = pickle.loads(read_queue.get())
                x=F.relu(x)
                write_queue.put(pickle.dumps(x))
                break
        pu.socket_del(thread_recv,thread_send,read_queue,write_queue,stop_queue,udp_socket,my_socket)




if __name__=='__main__':
    main()
    